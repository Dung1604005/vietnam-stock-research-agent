from __future__ import annotations

from datetime import date

import pytest

from conftest import conflict_market_data
from conftest import InMemoryProvider
from stock_multi_agent.agents.data_agent import DataAgent
from stock_multi_agent.agents.fundamental_agent import FundamentalAgent
from stock_multi_agent.agents.risk_agent import RiskAgent
from stock_multi_agent.agents.technical_agent import TechnicalAgent
from stock_multi_agent.chat.response import ResponseComposer
from stock_multi_agent.models.data import DataKind, Instrument, IssuerEvent, PortfolioPosition, SourceReference
from stock_multi_agent.models.data import FinancialStatementObservation
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.providers.market_provider import VnstockMarketDataProvider
from stock_multi_agent.services.chart_service import ChartService
from stock_multi_agent.tools.financial_data import FinancialToolkit
from stock_multi_agent.tools.market_data import MarketDataTool
from stock_multi_agent.tools.technical_indicators import TechnicalIndicatorToolkit


def _state(horizon: str = "3 months") -> AgentState:
    return AgentState(
        ticker="FPT", investment_horizon=horizon, user_goal="Analyse FPT", market_data=conflict_market_data(),
    )


def test_horizon_changes_technical_windows_and_evidence_has_lineage() -> None:
    toolkit = TechnicalIndicatorToolkit()
    short = TechnicalAgent(toolkit).run(_state("1 month"))["technical_result"]
    long = TechnicalAgent(toolkit).run(_state("1 year"))["technical_result"]
    assert short.horizon_profile == "short"
    assert long.horizon_profile == "long"
    assert "SMA10" in short.tools_used
    assert "SMA150" in long.tools_used
    assert short.evidence[0].source_ids == ["test:ohlcv:FPT"]
    assert short.evidence[0].as_of is not None


def test_technical_uses_only_aligned_vnindex_sessions_for_relative_return() -> None:
    state = _state()
    state.benchmark_data = conflict_market_data("VNINDEX")
    result = TechnicalAgent(TechnicalIndicatorToolkit()).run(state)["technical_result"]
    assert result.diagnostics["relative_benchmark_return"] == pytest.approx(0.0)
    assert any(item.label == "Relative return versus VNINDEX" for item in result.evidence)


def test_data_agent_keeps_issuer_result_when_optional_benchmark_is_unavailable() -> None:
    data = conflict_market_data()
    state = _state()
    state.required_data.append("benchmark")
    update = DataAgent(MarketDataTool(InMemoryProvider(data))).run(state)
    assert update["market_data"] == data
    assert update["benchmark_data"] is None
    assert any(event.action == "BENCHMARK_UNAVAILABLE" for event in update["trace"])


def test_technical_response_discloses_vnindex_provenance_when_used() -> None:
    state = _state()
    state.benchmark_data = conflict_market_data("VNINDEX")
    state.technical_result = TechnicalAgent(TechnicalIndicatorToolkit()).run(state)["technical_result"]
    response = ResponseComposer.technical(state)
    assert any(item.source_id == "test:ohlcv:VNINDEX" for item in response.source_details)


def test_chart_and_analysis_use_the_same_rsi_implementation() -> None:
    data = conflict_market_data()
    service = ChartService(TechnicalIndicatorToolkit())
    from stock_multi_agent.models.chart import ChartRequest, ChartType

    artifact = service.build_artifact(ChartRequest(ticker="FPT", chart_type=ChartType.RSI, period="6M"), data)
    expected = TechnicalIndicatorToolkit().rsi_series([bar.close for bar in data.bars])[-1]
    assert artifact.records[-1]["rsi"] == pytest.approx(expected)


def test_risk_decomposes_unknown_event_and_portfolio_dimensions() -> None:
    state = _state()
    toolkit = TechnicalIndicatorToolkit()
    state.technical_result = TechnicalAgent(toolkit).run(state)["technical_result"]
    state.fundamental_result = FundamentalAgent(FinancialToolkit()).run(state)["fundamental_result"]
    result = RiskAgent(toolkit).run(state)["risk_result"]
    sections = {section.name: section for section in result.sections}
    assert sections["market"].status == "ASSESSED"
    assert sections["liquidity"].status == "ASSESSED"
    assert sections["event_news"].status == "UNKNOWN"
    assert sections["portfolio"].status == "UNKNOWN"


def test_bank_template_does_not_treat_debt_to_equity_as_generic_leverage() -> None:
    data = conflict_market_data().model_copy(update={"instrument": Instrument(ticker="FPT", sector="Banking")})
    state = AgentState(ticker="FPT", investment_horizon="3 months", user_goal="Analyse FPT", market_data=data)
    result = FundamentalAgent(FinancialToolkit()).run(state)["fundamental_result"]
    assert result.sector_template == "BANK"
    assert "Debt/equity" not in result.feature_contributions


@pytest.mark.parametrize(
    ("sector", "template", "expected_score"),
    [
        ("Technology", "NON_FINANCIAL_GENERIC", 5.7),
        ("Banking", "BANK", 4.9),
        ("Securities", "SECURITIES", 4.9),
        ("Insurance", "INSURANCE", 4.9),
        ("Real Estate", "REAL_ESTATE", 4.9),
    ],
)
def test_sector_templates_match_hand_calculated_golden_scores(
    sector: str,
    template: str,
    expected_score: float,
) -> None:
    data = conflict_market_data().model_copy(update={"instrument": Instrument(ticker="FPT", sector=sector)})
    state = AgentState(ticker="FPT", investment_horizon="3 months", user_goal="Analyse FPT", market_data=data)
    result = FundamentalAgent(FinancialToolkit()).run(state)["fundamental_result"]
    assert result.sector_template == template
    assert result.score == pytest.approx(expected_score)


def test_statement_normalizer_keeps_multiple_periods_and_known_lines() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        [
            ["revenue", "Revenue", 100.0, 120.0],
            ["profit_after_tax", "Profit after tax", 10.0, 15.0],
        ],
        columns=["item_id", "item_en", "2024", "2025"],
    )
    observations = VnstockMarketDataProvider._normalize_statement_observations(
        frame, "income_statement", "VCI:Finance.income_statement:FPT",
    )
    assert {(item.metric, item.period_end.year) for item in observations} == {
        ("revenue", 2024), ("revenue", 2025), ("net_profit", 2024), ("net_profit", 2025),
    }


def test_fundamental_uses_reported_growth_and_cash_conversion_when_ratios_are_missing() -> None:
    data = conflict_market_data().model_copy(update={
        "financials": conflict_market_data().financials.model_copy(update={"revenue_growth": None, "profit_growth": None}),
        "financial_statement_observations": [
            FinancialStatementObservation(statement="income_statement", metric="revenue", value=100, unit="currency", period_label="2024", period_end=date(2024, 12, 31), source_id="test:income"),
            FinancialStatementObservation(statement="income_statement", metric="revenue", value=120, unit="currency", period_label="2025", period_end=date(2025, 12, 31), source_id="test:income"),
            FinancialStatementObservation(statement="income_statement", metric="net_profit", value=10, unit="currency", period_label="2025", period_end=date(2025, 12, 31), source_id="test:income"),
            FinancialStatementObservation(statement="cash_flow", metric="operating_cash_flow", value=12, unit="currency", period_label="2025", period_end=date(2025, 12, 31), source_id="test:cash"),
        ],
    })
    result = FundamentalAgent(FinancialToolkit()).run(
        AgentState(ticker="FPT", investment_horizon="3 months", user_goal="Analyse FPT", market_data=data),
    )["fundamental_result"]
    assert result.feature_contributions["reported_revenue_growth"] == pytest.approx(0.7)
    assert result.feature_contributions["cash_conversion"] == pytest.approx(0.4)


def test_current_statements_keep_fundamental_analysis_available_when_ratios_are_stale() -> None:
    from stock_multi_agent.models.data import FinancialMetricObservation
    from stock_multi_agent.services.data_quality import DataQualityChecker

    statements = [
        FinancialStatementObservation(statement="income_statement", metric="revenue", value=100, unit="currency", period_label="2024", period_end=date(2024, 12, 31), source_id="test:income"),
        FinancialStatementObservation(statement="income_statement", metric="revenue", value=120, unit="currency", period_label="2025", period_end=date(2025, 12, 31), source_id="test:income"),
        FinancialStatementObservation(statement="income_statement", metric="net_profit", value=10, unit="currency", period_label="2024", period_end=date(2024, 12, 31), source_id="test:income"),
        FinancialStatementObservation(statement="income_statement", metric="net_profit", value=14, unit="currency", period_label="2025", period_end=date(2025, 12, 31), source_id="test:income"),
    ]
    data = conflict_market_data().model_copy(update={
        "financial_observations": [FinancialMetricObservation(metric="pe", value=18, unit="multiple", period_label="2018", period_end=date(2018, 12, 31), source_id="test:ratio")],
        "financial_statement_observations": statements,
    })
    quality = DataQualityChecker().evaluate(data, require_financials=True, today=date(2026, 10, 9))
    assert any(issue.code == "STALE_FINANCIAL_RATIOS" for issue in quality.issues)
    assert not any(issue.code == "STALE_FINANCIALS" for issue in quality.issues)
    result = FundamentalAgent(FinancialToolkit()).run(
        AgentState(ticker="FPT", investment_horizon="3 months", user_goal="Analyse FPT", market_data=data.model_copy(update={"quality_report": quality})),
    )["fundamental_result"]
    assert result.signal is not result.signal.UNKNOWN
    assert "reported_revenue_growth" in result.feature_contributions


def test_risk_assesses_verified_events_and_user_portfolio_when_supplied() -> None:
    data = conflict_market_data()
    event_ref = SourceReference(
        provider="test", source_id="test:events:FPT", data_kind=DataKind.ISSUER_EVENTS,
        fetched_at=data.fetched_at, as_of=data.as_of,
    )
    data = data.model_copy(update={
        "source_refs": [*data.source_refs, event_ref],
        "issuer_events": [IssuerEvent(event_code="SUSP", title="Trading suspension", event_date=data.as_of, source_id=event_ref.source_id)],
    })
    state = AgentState(
        ticker="FPT", investment_horizon="3 months", user_goal="Analyse FPT", market_data=data,
        portfolio_positions=[PortfolioPosition(ticker="FPT", weight=0.8), PortfolioPosition(ticker="VCB", weight=0.2)],
    )
    state.technical_result = TechnicalAgent(TechnicalIndicatorToolkit()).run(state)["technical_result"]
    state.fundamental_result = FundamentalAgent(FinancialToolkit()).run(state)["fundamental_result"]
    result = RiskAgent(TechnicalIndicatorToolkit()).run(state)["risk_result"]
    sections = {section.name: section for section in result.sections}
    assert sections["event_news"].status == "ASSESSED"
    assert sections["portfolio"].status == "ASSESSED"
    assert any("suspension" in item.casefold() for item in result.criticisms)
