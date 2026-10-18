"""Offline fixtures: test mechanics, not financial accuracy or live Colab quality."""
from datetime import date

import pytest

from conftest import conflict_market_data
from stock_multi_agent.models.chart import ChartRequest, ChartType
from stock_multi_agent.models.chat import ConversationState
from stock_multi_agent.models.data import FinancialStatementObservation, FinancialMetricObservation
from stock_multi_agent.services.chart_service import ChartService
from stock_multi_agent.chat.query_parser import QueryParser
from stock_multi_agent.models.chat import QueryIntent
from stock_multi_agent.services.agent_llm import AgentLLMAdvisor
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.chat_service import ChatService
from stock_multi_agent.services.llm_gateway import LLMGateway


@pytest.mark.parametrize('kind', [ChartType.PRICE, ChartType.CANDLESTICK, ChartType.PRICE_MA, ChartType.PRICE_VOLUME,
    ChartType.RSI, ChartType.MACD, ChartType.BOLLINGER_BANDS, ChartType.DRAWDOWN, ChartType.VOLATILITY, ChartType.RETURNS])
def test_price_charts_are_renderable_with_source_and_time(kind):
    service = ChartService()
    data = conflict_market_data()
    artifact = service.build_artifact(ChartRequest(ticker='FPT', chart_type=kind), data)
    figure = service.render_figure(artifact)
    assert figure.data and artifact.source_details and artifact.data_timestamp == data.as_of
    assert artifact.explanation


def test_short_price_history_is_not_replaced_by_neutral_indicator():
    data = conflict_market_data()
    data.bars = data.bars[-5:]
    artifact = ChartService().build_artifact(ChartRequest(ticker='FPT', chart_type=ChartType.PRICE), data)
    assert len(artifact.records) == 5
    assert all(row['rsi'] is None and row['ma20'] is None for row in artifact.records)


def test_drawdown_and_volatility_follow_declared_formulas():
    import numpy as np
    data = conflict_market_data()
    artifact = ChartService().build_artifact(ChartRequest(ticker='FPT', chart_type=ChartType.DRAWDOWN), data)
    prices = np.array([bar.close for bar in data.bars])
    assert artifact.records[-1]['drawdown'] == pytest.approx((prices[-1] / prices.max() - 1) * 100)
    expected = np.std(np.diff(np.log(prices))[-20:], ddof=1) * np.sqrt(252) * 100
    assert artifact.records[-1]['volatility'] == pytest.approx(expected)


@pytest.mark.parametrize('text,kinds', [
    ('Vẽ Bollinger FPT', [ChartType.BOLLINGER_BANDS]),
    ('Vẽ drawdown FPT', [ChartType.DRAWDOWN]),
    ('Vẽ biểu đồ biến động FPT', [ChartType.VOLATILITY]),
    ('Vẽ doanh thu lợi nhuận FPT', [ChartType.REVENUE_PROFIT]),
    ('Vẽ tăng trưởng doanh thu FPT', [ChartType.GROWTH]),
    ('Vẽ RSI và MACD FPT', [ChartType.RSI, ChartType.MACD]),
    ('Vẽ dòng tiền FPT', [ChartType.CASH_FLOW]),
    ('Vẽ ROE ROA FPT', [ChartType.PROFITABILITY]),
    ('Vẽ MA100 MA200 FPT', [ChartType.PRICE_MA]),
])
def test_nlu_and_multi_chart_selection(text, kinds):
    query = QueryParser(use_configured_provider=False).parse(text)
    assert query.intent == QueryIntent.STOCK_CHART and query.tickers == ['FPT']
    assert [request.chart_type for request in ChartService().requests_from_query(query)] == kinds


def test_financial_charts_keep_units_frequencies_and_reject_conflicting_values():
    data = conflict_market_data()
    data.financial_statement_observations = [FinancialStatementObservation(statement='income_statement',
        metric=metric, value=value, unit=unit, period_label=str(year), period_end=date(year, 12, 31),
        frequency='annual', source_id=f'fixture:{year}') for year, metric, value, unit in [
            (2024, 'revenue', 100, 'VND'), (2024, 'net_profit', 10, 'VND'),
            (2025, 'revenue', 200, 'USD'), (2025, 'revenue', 300, 'USD')]]
    service = ChartService()
    artifact = service.build_artifact(ChartRequest(ticker='FPT', chart_type=ChartType.REVENUE_PROFIT), data)
    assert len(artifact.records) == 2 and artifact.data_timestamp == date(2024, 12, 31)
    assert any('xung đột' in warning for warning in artifact.warnings)
    assert len(service.render_figure(artifact).data) == 2
    with pytest.raises(ValueError, match='Không có quan sát'):
        service.build_artifact(ChartRequest(ticker='FPT', chart_type=ChartType.GROWTH), data)


class Provider:
    def get_market_data(self, ticker, include_financials=True):
        return conflict_market_data(ticker)


class SpecialistModel:
    model = 'offline-fixture'
    def __init__(self): self.payloads = []
    def parse_structured(self, system, payload, output_model):
        self.payloads.append((system, payload))
        if output_model.__name__ == '_TechnicalIndicatorPlan':
            return output_model.model_validate({'selected_indicators': ['ATR'], 'rationale': 'Đo biên độ giá từ dữ liệu đã có.', 'cited_source_ids': payload['allowed_source_ids'][:1]})
        return output_model.model_validate({'assessment': 'Dữ liệu cần được đối chiếu với kỳ báo cáo và rủi ro.',
            'supporting_observations': ['Doanh thu được xem cùng lợi nhuận.', 'Dữ liệu đã có nguồn để đối chiếu.', 'Rủi ro được tách khỏi độ bất định dữ liệu.'],
            'counterarguments': ['Thiếu lịch sử dài để kết luận bền vững.', 'Dự báo vẫn có thể sai.'],
            'missing_information': ['Cần đối chiếu báo cáo mới nhất.'], 'cited_source_ids': payload['allowed_source_ids'][:1]})


def test_colab_compatible_specialist_reviews_run_once_after_conflict_without_mutating_scores(monkeypatch):
    monkeypatch.setenv('STOCK_AGENT_LLM_SPECIALIST_ROLES', 'technical,fundamental,risk')
    model = SpecialistModel()
    advisor = AgentLLMAdvisor(provider=LLMGateway(model), enabled=True)
    analysis = AnalysisService(provider=Provider(), llm_advisor=advisor)
    reference = AnalysisService(provider=Provider()).analyze('FPT', allow_llm_planning=False)
    service = ChatService(analysis_service=analysis, universe={'default': ['FPT']})
    response = service.handle('Phân tích FPT', ConversationState())
    roles = [payload['specialist_role'] for _, payload in model.payloads if 'specialist_role' in payload]
    assert roles.count('fundamental') == 1 and roles.count('risk') == 1
    assert response.metadata['request_metrics']['llm_calls'] == 3
    assert all(review['status'] == 'GROUNDED' for review in response.metadata['llm_specialist_reviews'])
    assert response.metadata['risk_score'] == reference.risk_result.risk_score
    assert response.metadata['fundamental_score'] == reference.fundamental_result.score
    assert 'Rủi ro đến từ đâu?' in [table.title for table in response.tables]
    assert response.metadata['analysis_explanation']
    fnd = next(payload['facts'] for _, payload in model.payloads if payload.get('specialist_role') == 'fundamental')
    assert fnd['reported_metrics'] and 'metric_periods' in fnd


def test_batch_analysis_makes_no_specialist_llm_calls():
    model = SpecialistModel()
    advisor = AgentLLMAdvisor(provider=LLMGateway(model), enabled=True)
    advisor.roles = {'technical', 'fundamental', 'risk'}
    AnalysisService(provider=Provider(), llm_advisor=advisor).analyze('FPT', allow_llm_planning=False)
    assert model.payloads == []


@pytest.mark.parametrize('query,intent,role', [
    ('Phân tích cơ bản FPT', QueryIntent.FUNDAMENTAL_ANALYSIS, 'fundamental'),
    ('Đánh giá rủi ro FPT', QueryIntent.RISK_ANALYSIS, 'risk'),
])
def test_specific_specialist_requests_have_reviews(monkeypatch, query, intent, role):
    monkeypatch.setenv('STOCK_AGENT_LLM_SPECIALIST_ROLES', 'fundamental,risk')
    model = SpecialistModel()
    service = ChatService(analysis_service=AnalysisService(provider=Provider(), llm_advisor=AgentLLMAdvisor(
        provider=LLMGateway(model), enabled=True)), universe={'default': ['FPT']})
    response = service.handle(query, ConversationState())
    assert response.metadata['parsed_intent'] == intent.value
    assert any(review['agent'] == role and review['status'] == 'GROUNDED' for review in response.metadata['llm_specialist_reviews'])


def test_llm_failure_keeps_scores_and_does_not_masquerade_as_generated_review(monkeypatch):
    monkeypatch.setenv('STOCK_AGENT_LLM_SPECIALIST_ROLES', 'fundamental,risk')
    class Broken:
        def parse_structured(self, *args): raise RuntimeError('offline')
    service = ChatService(analysis_service=AnalysisService(provider=Provider(), llm_advisor=AgentLLMAdvisor(
        provider=LLMGateway(Broken()), enabled=True)), universe={'default': ['FPT']})
    response = service.handle('Phân tích FPT', ConversationState())
    assert 'fundamental_score' in response.metadata and 'risk_score' in response.metadata
    assert all(review['status'] == 'UNAVAILABLE' for review in response.metadata['llm_specialist_reviews'])
    assert 'LLM không tạo được nhận định' in response.message
