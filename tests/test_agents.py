from stock_multi_agent.agents.fundamental_agent import FundamentalAgent
from stock_multi_agent.agents.manager_agent import ManagerAgent
from stock_multi_agent.agents.risk_agent import RiskAgent
from stock_multi_agent.agents.technical_agent import TechnicalAgent
from stock_multi_agent.models.analysis import FundamentalAnalysis, RiskAnalysis, RiskLevel, Signal, TechnicalAnalysis
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.tools.financial_data import FinancialToolkit
from stock_multi_agent.tools.technical_indicators import TechnicalIndicatorToolkit


def _state_with_results(market_data) -> AgentState:
    return AgentState(
        ticker="FPT", investment_horizon="3 months", user_goal="Analyze FPT",
        market_data=market_data,
        technical_result=TechnicalAnalysis(
            signal=Signal.BUY, score=8, confidence=0.85, evidence=[], risks=[],
            tools_used=["SMA20"], reasoning_summary="Uptrend", timeframe="3 months",
        ),
        fundamental_result=FundamentalAnalysis(
            signal=Signal.HOLD, score=6, confidence=0.72, strengths=[], weaknesses=[],
            evidence=[], reasoning_summary="Valuation is mixed",
        ),
    )


def test_risk_vetoes_volatile_conflicting_case(market_data) -> None:
    state = _state_with_results(market_data)
    update = RiskAgent(TechnicalIndicatorToolkit()).run(state)
    result = update["risk_result"]
    assert result.risk_level is RiskLevel.HIGH
    assert result.veto is True
    assert any("disagree" in item for item in result.criticisms)


def test_missing_fundamental_evidence_is_uncertainty_not_automatic_high_risk(market_data) -> None:
    stable_bars = [
        bar.model_copy(update={
            "open": 100.0 + index * 0.02,
            "high": 100.2 + index * 0.02,
            "low": 99.8 + index * 0.02,
            "close": 100.0 + index * 0.02,
            "volume": 20_000_000,
        })
        for index, bar in enumerate(market_data.bars)
    ]
    state = AgentState(
        ticker="FPT",
        investment_horizon="3 months",
        user_goal="Analyze FPT",
        market_data=market_data.model_copy(update={"bars": stable_bars}),
        technical_result=TechnicalAnalysis(
            signal=Signal.HOLD, score=5, confidence=0.75, evidence=[], risks=[],
            tools_used=[], reasoning_summary="Neutral", timeframe="3 months",
        ),
        fundamental_result=FundamentalAnalysis(
            signal=Signal.UNKNOWN, score=5, confidence=0.2, strengths=[], weaknesses=[],
            evidence=[], reasoning_summary="Insufficient financial evidence",
        ),
    )

    result = RiskAgent(TechnicalIndicatorToolkit()).run(state)["risk_result"]

    assert result.risk_level is RiskLevel.MEDIUM
    assert result.risk_score < 3.5
    assert result.veto is False
    assert result.policy_version == "risk-v3-balanced"


def test_manager_detects_conflict_and_routes_reanalysis(market_data) -> None:
    state = _state_with_results(market_data)
    state.risk_result = RiskAnalysis(
        risk_level=RiskLevel.HIGH, risk_score=9, criticisms=[], warnings=[], veto=True,
        reasoning_summary="High volatility and disagreement",
    )
    manager = ManagerAgent()
    conflicts = manager.conflicts(state)
    assert len(conflicts) >= 3
    update = manager.review(state)
    merged = state.model_copy(update=update)
    assert merged.iteration == 1
    assert manager.route_after_review(merged) == "reanalyze"


def test_max_iterations_protects_route(market_data) -> None:
    state = _state_with_results(market_data)
    state.max_iterations = 0
    state.risk_result = RiskAnalysis(
        risk_level=RiskLevel.HIGH, risk_score=9, criticisms=[], warnings=[], veto=True,
        reasoning_summary="High risk",
    )
    manager = ManagerAgent()
    update = manager.review(state)
    merged = state.model_copy(update=update)
    assert update["reanalysis_requested"] is False
    assert manager.route_after_review(merged) == "finalize"


def test_specialists_are_independent(market_data) -> None:
    state = AgentState(
        ticker="FPT", investment_horizon="3 months", user_goal="Analyze FPT", market_data=market_data
    )
    technical = TechnicalAgent(TechnicalIndicatorToolkit()).run(state)["technical_result"]
    fundamental = FundamentalAgent(FinancialToolkit()).run(state)["fundamental_result"]
    assert technical.signal is Signal.BUY
    assert fundamental.signal is Signal.HOLD

