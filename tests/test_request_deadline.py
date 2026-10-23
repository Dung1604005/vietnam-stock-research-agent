"""Offline deadline regressions: no real waits or upstream requests."""
from __future__ import annotations

import sys
import types

import pytest
from pydantic import BaseModel

from stock_multi_agent.models.research_contracts import RequestBudget
from stock_multi_agent.providers.market_provider import VnstockMarketDataProvider
from stock_multi_agent.providers.resilience import RateLimiter, ResiliencePolicy, ResilientExecutor
from stock_multi_agent.services import request_telemetry as telemetry
from stock_multi_agent.services.llm_gateway import LLMGateway
from stock_multi_agent.services.request_telemetry import (
    BudgetExceeded, check_deadline, observed, remaining_seconds, request_scope, task_deadline_scope,
)


class Clock:
    def __init__(self):
        self.now = 100.0
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.waits.append(duration)
        self.now += duration


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(telemetry.time, 'perf_counter', value)
    monkeypatch.setenv('STOCK_AGENT_REQUEST_TELEMETRY_ENABLED', 'false')
    return value


def executor(clock, **policy):
    defaults = dict(max_attempts=3, min_request_interval_seconds=0, base_delay_seconds=0)
    defaults.update(policy)
    return ResilientExecutor(
        ResiliencePolicy(**defaults), sleeper=clock.sleep,
        rate_limiter=RateLimiter(defaults['min_request_interval_seconds'], clock=clock, sleeper=clock.sleep),
    )


def test_expired_request_never_calls_operation(clock):
    worker = executor(clock)
    calls = []
    with request_scope(RequestBudget(deadline_seconds=1)) as metrics:
        clock.now += 1
        assert metrics.remaining_seconds() == 0
        with pytest.raises(BudgetExceeded):
            worker.run(lambda: calls.append('upstream'))
        assert calls == []
        assert metrics.api_calls == 0
        assert worker.health().consecutive_failures == 0


def test_rate_limit_wait_is_bounded_and_no_operation_starts(clock):
    worker = executor(clock, min_request_interval_seconds=10)
    calls = []
    worker.run(lambda: calls.append('first'))
    with request_scope(RequestBudget(deadline_seconds=2)) as metrics:
        with pytest.raises(BudgetExceeded):
            worker.run(lambda: calls.append('forbidden'))
        assert calls == ['first']
        assert clock.waits == [2]
        assert metrics.api_calls == 0
        assert worker.rate_limiter.total_wait_seconds == 2


def test_deadline_is_rechecked_after_custom_rate_limiter(clock):
    class SlowLimiter:
        def acquire(self):
            clock.now += 3

    worker = ResilientExecutor(rate_limiter=SlowLimiter())
    calls = []
    with request_scope(RequestBudget(deadline_seconds=1)):
        with pytest.raises(BudgetExceeded):
            worker.run(lambda: calls.append('forbidden'))
    assert calls == []
    assert worker.attempts_total == 0


def test_retry_sleep_cannot_consume_more_than_request_time(clock):
    worker = executor(clock, base_delay_seconds=10)
    calls = []

    def failing():
        calls.append('first')
        raise ConnectionError('temporary')

    with request_scope(RequestBudget(deadline_seconds=2)) as metrics:
        with pytest.raises(BudgetExceeded):
            worker.run(failing)
        assert calls == ['first']
        assert clock.waits == [2]
        assert metrics.api_calls == 1 and metrics.retries == 0
        assert worker.health().consecutive_failures == 0
        assert worker.health().retries_total == 0


def test_operation_that_exceeds_deadline_is_not_retried(clock):
    worker = executor(clock)
    calls = []

    def slow_failure():
        calls.append('upstream')
        clock.now += 2
        raise TimeoutError('timeout')

    with request_scope(RequestBudget(deadline_seconds=1)):
        with pytest.raises(BudgetExceeded):
            worker.run(slow_failure)
    assert calls == ['upstream']
    assert clock.waits == []


def test_success_after_deadline_cannot_start_downstream_work(clock):
    worker = executor(clock)

    def slow_success():
        clock.now += 2
        return 'late'

    with request_scope(RequestBudget(deadline_seconds=1)):
        with pytest.raises(BudgetExceeded):
            worker.run(slow_success)
        assert worker.successful_operations == 0


def test_explicit_budget_failure_is_not_retried_or_breaker_failure(clock):
    worker = executor(clock)
    calls = []

    def operation():
        calls.append('upstream')
        raise BudgetExceeded('tool budget')

    with request_scope():
        with pytest.raises(BudgetExceeded):
            worker.run(operation)
    assert calls == ['upstream']
    assert worker.health().consecutive_failures == 0
    assert clock.waits == []


def test_task_deadline_scope_narrows_and_restores_request(clock):
    with request_scope(RequestBudget(deadline_seconds=20)) as metrics:
        with task_deadline_scope(2):
            assert metrics.remaining_seconds() == 2
            with task_deadline_scope(50):
                assert remaining_seconds() == 2
            clock.now += 3
            with pytest.raises(BudgetExceeded):
                check_deadline()
        assert metrics.remaining_seconds() == 17
        check_deadline()
    assert remaining_seconds() is None


def test_finalization_and_snapshot_remain_available_after_deadline(clock):
    @observed('finalize')
    def finalize():
        return {'partial': True}

    @observed('tool', 'tool')
    def tool():
        pytest.fail('expired tool must not run')

    with request_scope(RequestBudget(deadline_seconds=1)) as metrics:
        clock.now += 2
        with pytest.raises(BudgetExceeded):
            tool()
        assert finalize() == {'partial': True}
        assert metrics.snapshot()['nodes'][-1]['node'] == 'finalize'


def test_provider_does_not_swallow_deadline_and_try_fallback_source(clock, monkeypatch):
    calls = []

    class Quote:
        def __init__(self, **kwargs):
            calls.append(kwargs['source'])

        def history(self, **kwargs):
            clock.now += 2
            raise TimeoutError('transport timeout')

    monkeypatch.setitem(sys.modules, 'vnstock.api.quote', types.SimpleNamespace(Quote=Quote))
    monkeypatch.setitem(sys.modules, 'vnstock.api.financial', types.SimpleNamespace(Finance=object))
    provider = VnstockMarketDataProvider(enable_cache=False, resilience_policy=ResiliencePolicy(min_request_interval_seconds=0))
    with request_scope(RequestBudget(deadline_seconds=1)):
        with pytest.raises(BudgetExceeded):
            provider.get_market_data('FPT')
    assert calls == ['VCI']


def test_provider_clamps_timeout_when_supported(clock):
    timeouts = []

    def transport(*, timeout):
        timeouts.append(timeout)
        return 'ok'

    with request_scope(RequestBudget(deadline_seconds=2)):
        clock.now += .5
        assert VnstockMarketDataProvider._call_upstream(transport) == 'ok'
    assert timeouts == [1.5]


def test_provider_refresh_does_not_delete_cache_after_expiry(clock):
    deletes = []
    provider = VnstockMarketDataProvider(enable_cache=False)
    provider.cache = types.SimpleNamespace(delete=lambda key: deletes.append(key))
    with request_scope(RequestBudget(deadline_seconds=1)):
        clock.now += 2
        with pytest.raises(BudgetExceeded):
            provider.refresh_market_data('FPT')
    assert deletes == []


class Review(BaseModel):
    assessment: str


class FakeLLM:
    timeout_seconds = 45

    def __init__(self, clock, *, fails=False, delay=0):
        self.clock = clock
        self.fails = fails
        self.delay = delay
        self.calls = []

    def parse_structured(self, system, payload, output_model):
        self.calls.append(self.timeout_seconds)
        self.clock.now += self.delay
        if self.fails:
            raise TimeoutError('temporary')
        return output_model(assessment='Dữ liệu cần theo dõi.')


def test_gateway_checks_expiry_even_for_cached_reviews(clock):
    provider = FakeLLM(clock)
    gateway = LLMGateway(provider)
    with request_scope(RequestBudget(deadline_seconds=1)):
        gateway.parse_structured('review', {}, Review)
        clock.now += 2
        with pytest.raises(BudgetExceeded):
            gateway.parse_structured('review', {}, Review)
    assert len(provider.calls) == 1


def test_gateway_clamps_transport_to_remaining_request_and_task(clock):
    provider = FakeLLM(clock)
    gateway = LLMGateway(provider)
    with request_scope(RequestBudget(deadline_seconds=20)):
        with task_deadline_scope(.25):
            gateway.parse_structured('review', {}, Review)
    assert provider.calls == [.25]
    assert provider.timeout_seconds == 45


def test_gateway_does_not_retry_expired_transport(clock):
    provider = FakeLLM(clock, fails=True, delay=2)
    gateway = LLMGateway(provider)
    with request_scope(RequestBudget(deadline_seconds=1)) as metrics:
        with pytest.raises(BudgetExceeded):
            gateway.parse_structured('review', {}, Review)
        assert metrics.llm_calls == 1
    assert provider.calls == [1]
    assert gateway.health()['consecutive_failures'] == 0
