"""Author-created parser/filter regressions, not investment performance evidence."""
from types import SimpleNamespace

import pytest

from conftest import conflict_market_data
from stock_multi_agent.chat.query_parser import QueryParser
from stock_multi_agent.chat.router import IntentRouter, RouteStep
from stock_multi_agent.models.analysis import Signal
from stock_multi_agent.models.chat import ConversationState, QueryIntent
from stock_multi_agent.models.screening import ScreenSpec, ScreenStatus
from stock_multi_agent.services.screening_service import ScreeningService
from stock_multi_agent.services.universe_service import UniverseService
from stock_multi_agent.tools.technical_indicators import TechnicalIndicatorToolkit


@pytest.mark.parametrize('prompt,expected', [
    ('Tìm cho tôi 10 cổ phiếu có RSI > 30', {'rsi_gt': 30.}),
    ('TIM CHO TOI 10 CO PHIEU CO RSI > 30', {'rsi_gt': 30.}),
    ('Tìm 10 mã RSI>=30', {'rsi_gte': 30.}),
    ('Tìm 10 mã RSI ≤ 70', {'rsi_lte': 70.}),
    ('Lọc 10 cổ phiếu RSI lớn hơn 30,5', {'rsi_gt': 30.5}),
    ('Lọc 10 mã RSI không thấp hơn 30', {'rsi_gte': 30.}),
    ('Tìm 10 mã RSI dưới 70', {'rsi_lt': 70.}),
    ('Tìm 10 mã RSI từ 30 đến 70', {'rsi_gte': 30., 'rsi_lte': 70.}),
    ('Tìm 10 mã RSI > 30 và < 70', {'rsi_gt': 30., 'rsi_lt': 70.}),
    ('Tìm 10 mã RSI14 = 50', {'rsi_eq': 50.}),
    ('Tìm 10 mã P/E < 15 và ROE >= 20', {'pe_lt': 15., 'roe_gte': 20.}),
])
def test_numeric_screen_retains_operator_count_and_route(prompt, expected):
    context = ConversationState(active_tickers=['FPT'], last_intent=QueryIntent.STOCK_ANALYSIS)
    query = QueryParser(use_configured_provider=False).parse(prompt, context)
    assert query.intent == QueryIntent.STOCK_DISCOVERY
    assert query.filters == expected
    assert query.requested_result_count == 10
    assert query.tickers == [] and not query.needs_clarification
    assert IntentRouter().plan(query).steps == [RouteStep.SCREENING]


@pytest.mark.parametrize('prompt', [
    'Tìm 10 mã RSI > 101', 'Tìm 10 mã RSI > 70 và RSI < 30',
    'Tìm 10 mã RSI > 30 hoặc P/E < 15', 'Tìm 101 cổ phiếu RSI > 30',
    'Tìm 10 mã RSI21 > 30', 'Tìm 10 mã RSI > 30 và RSI <= 30',
    'Top 101 mã RSI > 30', 'Tìm 1000 mã RSI > 30',
])
def test_invalid_or_unsupported_conditions_require_clarification(prompt):
    query = QueryParser(use_configured_provider=False).parse(prompt)
    assert query.needs_clarification and query.clarification_question
    assert IntentRouter().plan(query).steps == [RouteStep.CHAT]


def test_theory_stays_in_knowledge_route():
    query = QueryParser(use_configured_provider=False).parse('RSI > 30 nghĩa là gì?')
    assert query.intent == QueryIntent.THEORY
    assert IntentRouter().plan(query).steps == [RouteStep.KNOWLEDGE]


def test_numeric_screen_does_not_call_llm_for_arithmetic(monkeypatch):
    import stock_multi_agent.nlu.semantic_parser as module
    class Forbidden:
        def parse_structured(self, *args): raise AssertionError('No LLM needed for numeric predicate')
    monkeypatch.setattr(module, 'configured_semantic_provider', lambda: Forbidden())
    query = QueryParser(use_configured_provider=True).parse('Tìm cho tôi 10 cổ phiếu có RSI > 30')
    assert query.filters == {'rsi_gt': 30.} and not query.llm_attempted


class FilterAnalysis:
    def __init__(self, tickers):
        self.data = {ticker: conflict_market_data(ticker) for ticker in tickers}
        self.calls = []

    def analyze_technical(self, ticker, *args, **kwargs):
        self.calls.append(ticker)
        # A SELL policy score must not silently remove neutral RSI matches.
        return SimpleNamespace(market_data=self.data[ticker], technical_result=SimpleNamespace(
            signal=Signal.SELL, score=1.), trace=[])

    def fetch_data(self, *args, **kwargs):
        raise AssertionError('RSI filtering must reuse the already evaluated price snapshot')


def test_rsi_screen_finds_requested_ten_from_whole_bounded_universe_without_refetch():
    tickers = [f'A{i:02d}' for i in range(18)]
    analysis = FilterAnalysis(tickers)
    spec = ScreenSpec(universe_key='default', filters={'rsi_gt': 30.},
        eligible_signals=list(Signal), minimum_technical_score=0., refresh_stale_data=False)
    run = ScreeningService(analysis, UniverseService({'default': tickers})).run_progressive(spec, '3 months', 10)
    assert len(run.selected_tickers) == 10
    assert analysis.calls == tickers
    assert run.pass_count == 18
    assert all(item.filter_values['rsi'] > 30 and item.technical_signal == Signal.SELL for item in run.candidates)


@pytest.mark.parametrize('operator,threshold,expected_count', [('gt', 100., 0), ('gte', 100., 3), ('lt', 100., 0), ('lte', 100., 3), ('eq', 100., 3)])
def test_strict_and_inclusive_thresholds_use_full_precision(monkeypatch, operator, threshold, expected_count):
    monkeypatch.setattr(TechnicalIndicatorToolkit, 'rsi', lambda self, values, period=14: 100.)
    tickers = ['FPT', 'TCB', 'VNM']
    spec = ScreenSpec(universe_key='default', filters={f'rsi_{operator}': threshold},
        eligible_signals=list(Signal), minimum_technical_score=0., refresh_stale_data=False)
    run = ScreeningService(FilterAnalysis(tickers), UniverseService({'default': tickers})).run_progressive(spec, '3 months', 3)
    assert len(run.selected_tickers) == expected_count


def test_end_to_end_chat_screen_returns_measured_rsi_and_sources(monkeypatch):
    from test_phase_e_screening import ScreenProvider
    from stock_multi_agent.services.analysis_service import AnalysisService
    from stock_multi_agent.services.chat_service import ChatService
    tickers = [f'A{i:02d}' for i in range(15)]
    provider = ScreenProvider(tickers)
    service = ChatService(analysis_service=AnalysisService(provider=provider),
        query_parser=QueryParser(use_configured_provider=False), universe={'default': tickers})
    response = service.handle('Tìm cho tôi 10 cổ phiếu có RSI > 30', ConversationState(), execution_mode='QUICK')
    assert response.metadata['semantic_understanding']['intent'] == 'STOCK_DISCOVERY'
    assert response.metadata['requested_result_count'] == 10
    assert response.metadata['returned_result_count'] == 10
    assert len(response.tickers) == len(response.tables[0].rows) == 10
    assert all(row['RSI14'] > 30 for row in response.tables[0].rows)
    assert response.source_details and response.data_timestamp
    assert 'RSI14 > 30' in response.message
    assert response.metadata['screen_spec']['minimum_technical_score'] == 0
