"""No network step may start after an earlier research subcall used its budget."""
import json
from email.message import Message
from types import SimpleNamespace

import pytest

from stock_multi_agent.models.research_contracts import RequestBudget
from stock_multi_agent.services import request_telemetry as telemetry
from stock_multi_agent.services.content_fetcher import SafeContentFetcher
from stock_multi_agent.services.embeddings import OllamaEmbeddingProvider
from stock_multi_agent.services.request_telemetry import BudgetExceeded, request_scope, task_deadline_scope
from stock_multi_agent.services.web_research import (
    DisabledCompanyNameResolver, DisabledIssuerDiscoveryProvider, FreeSearchProvider,
    WebResearchError, WebResearchService,
)


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=100.0)
    monkeypatch.setattr(telemetry.time, 'perf_counter', lambda: state.now)
    monkeypatch.setenv('STOCK_AGENT_REQUEST_TELEMETRY_ENABLED', 'false')
    return state


@pytest.mark.parametrize('expired_stage', ['issuer', 'resolver'])
def test_discovery_chain_stops_between_subcalls(clock, expired_stage):
    calls = []
    def operation(name, result):
        def run(*args, **kwargs):
            calls.append(name)
            if name == expired_stage:
                clock.now += 2
            return result
        return run
    service = WebResearchService(
        issuer_provider=SimpleNamespace(name='issuer', search=operation('issuer', [])),
        company_name_resolver=SimpleNamespace(resolve=operation('resolver', 'Company')),
        provider=SimpleNamespace(name='search', search=operation('search', [])))
    with request_scope(RequestBudget(deadline_seconds=1)):
        with pytest.raises(BudgetExceeded):
            service.search('FPT annual report', ticker='FPT')
    assert calls == (['issuer'] if expired_stage == 'issuer' else ['issuer', 'resolver'])


@pytest.mark.parametrize('raises', [False, True])
def test_free_search_does_not_try_next_provider_after_expiry(clock, raises):
    calls = []
    def first(*args):
        calls.append('first')
        clock.now += 2
        if raises:
            raise WebResearchError('Temporary failure')
        return []
    def second(*args):
        calls.append('second')
        return []
    provider = FreeSearchProvider([SimpleNamespace(name='first', search=first),
                                   SimpleNamespace(name='second', search=second)])
    with request_scope(RequestBudget(deadline_seconds=1)):
        with pytest.raises(BudgetExceeded):
            provider.search('annual report', 3)
    assert calls == ['first']


def test_fetcher_clamps_each_redirect_timeout(clock, monkeypatch):
    fetcher = SafeContentFetcher(resolver=lambda _: ['8.8.8.8'])
    headers = Message()
    headers['Location'] = '/next'
    headers['Content-Type'] = 'text/html'
    calls = []
    def open_once(url, timeout, max_bytes):
        calls.append(timeout)
        clock.now += 1
        return (302, headers, b'') if len(calls) == 1 else (200, headers, b'<p>Report</p>')
    monkeypatch.setattr(fetcher, '_open_once', open_once)
    with request_scope(RequestBudget(deadline_seconds=3)):
        assert fetcher.fetch('https://issuer.example/report').status_code == 200
    assert calls == [3, 2]


def test_fetcher_does_not_redirect_after_budget_expires(clock, monkeypatch):
    fetcher = SafeContentFetcher(resolver=lambda _: ['8.8.8.8'])
    calls = []
    def open_once(*args):
        calls.append(True)
        clock.now += 2
        headers = Message()
        headers['Location'] = '/next'
        return 302, headers, b''
    monkeypatch.setattr(fetcher, '_open_once', open_once)
    with request_scope(RequestBudget(deadline_seconds=1)):
        with pytest.raises(BudgetExceeded):
            fetcher.fetch('https://issuer.example/report')
    assert calls == [True]


def test_expired_dns_resolution_does_not_start_http(clock, monkeypatch):
    calls = []
    def resolver(hostname):
        clock.now += 2
        return ['8.8.8.8']
    fetcher = SafeContentFetcher(resolver=resolver)
    monkeypatch.setattr(fetcher, '_open_once', lambda *args: calls.append(True))
    with request_scope(RequestBudget(deadline_seconds=1)):
        with pytest.raises(BudgetExceeded):
            fetcher.fetch('https://issuer.example/report')
    assert calls == []


def test_embedding_transport_uses_remaining_task_time(clock, monkeypatch):
    import stock_multi_agent.services.embeddings as module
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({'embeddings': [[1, 0]]}).encode()
    def open_once(request, timeout):
        calls.append(timeout)
        return Response()
    monkeypatch.setattr(module, 'urlopen', open_once)
    with request_scope(RequestBudget(deadline_seconds=5)):
        with task_deadline_scope(.25):
            assert len(OllamaEmbeddingProvider().embed(['test'])) == 1
    assert calls == [.25]


def test_late_embedding_is_not_published(clock, monkeypatch):
    import stock_multi_agent.services.embeddings as module
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            clock.now += 2
            return json.dumps({'embeddings': [[1, 0]]}).encode()
    monkeypatch.setattr(module, 'urlopen', lambda *args, **kwargs: Response())
    with request_scope(RequestBudget(deadline_seconds=1)):
        with pytest.raises(BudgetExceeded):
            OllamaEmbeddingProvider().embed(['test'])


def test_orchestrator_does_not_continue_after_discovery_overruns(clock):
    from stock_multi_agent.services.research_orchestrator import ResearchOrchestrator
    calls = []
    def search(*args, **kwargs):
        calls.append('search')
        clock.now += 2
        return []
    orchestrator = ResearchOrchestrator(discovery=SimpleNamespace(search=search))
    with request_scope(RequestBudget(deadline_seconds=1)):
        with pytest.raises(BudgetExceeded):
            orchestrator.research('FPT annual report', ticker='FPT')
    assert calls == ['search']
