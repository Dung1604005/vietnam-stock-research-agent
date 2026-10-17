from types import SimpleNamespace

from conftest import conflict_market_data
from stock_multi_agent.models.analysis import (
    FinalDecision,
    FundamentalAnalysis,
    Recommendation,
    RiskAnalysis,
    RiskLevel,
    Signal,
    TechnicalAnalysis,
)
from stock_multi_agent.models.screening import (
    ProgressiveScreenRound,
    RecommendationBucket,
    ScreenStatus,
)
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.recommendation_service import RecommendationService


def _state(ticker: str, bucket: RecommendationBucket) -> AgentState:
    high = bucket is RecommendationBucket.HIGH_REWARD_HIGH_RISK
    return AgentState(
        ticker=ticker,
        investment_horizon="3 months",
        user_goal="test",
        market_data=conflict_market_data(ticker),
        technical_result=TechnicalAnalysis(
            signal=Signal.BUY,
            score=8.0 if high else 6.5,
            confidence=0.8,
            evidence=[], risks=[], tools_used=["RSI"], reasoning_summary="technical",
            timeframe="3 months",
            diagnostics={"annualized_volatility": 0.40 if high else 0.18},
        ),
        fundamental_result=FundamentalAnalysis(
            signal=Signal.BUY,
            score=7.0,
            confidence=0.75,
            strengths=[], weaknesses=[], evidence=[], reasoning_summary="fundamental",
        ),
        risk_result=RiskAnalysis(
            risk_level=RiskLevel.HIGH if high else RiskLevel.LOW,
            risk_score=6.0 if high else 3.0,
            criticisms=[], warnings=[], veto=False, reasoning_summary="risk",
            data_uncertainty_level=RiskLevel.MEDIUM if high else RiskLevel.LOW,
        ),
        final_decision=FinalDecision(
            ticker=ticker,
            recommendation=Recommendation.BUY,
            confidence=0.72,
            technical_summary="technical", fundamental_summary="fundamental", risk_summary="risk",
            agreements=[], conflicts=[], final_reasoning="decision", evidence=[],
        ),
    )


class _FakeScreening:
    def __init__(self, tickers: list[str]) -> None:
        self.tickers = tickers

    def run_progressive(self, spec, horizon, target_count):
        del spec, horizon
        return SimpleNamespace(
            universe=SimpleNamespace(key="default", tickers=[f"T{i:02d}" for i in range(100)]),
            pass_count=len(self.tickers),
            warnings=[],
            progressive_rounds=[ProgressiveScreenRound(
                round_number=1,
                input_count=len(self.tickers),
                survivor_count=min(target_count, len(self.tickers)),
                eliminated_tickers=self.tickers[target_count:],
            )],
            candidates=[SimpleNamespace(ticker=ticker, status=ScreenStatus.PASS) for ticker in self.tickers],
            selected_tickers=self.tickers[:target_count],
            refreshed_tickers=[],
            price_as_of_comparable=True,
        )


class _AllFailedScreening(_FakeScreening):
    def run_progressive(self, spec, horizon, target_count):
        result = super().run_progressive(spec, horizon, target_count)
        result.pass_count = 0
        result.candidates = [
            SimpleNamespace(ticker=ticker, status=ScreenStatus.FAIL, technical_score=6.0 - index * 0.1)
            for index, ticker in enumerate(self.tickers)
        ]
        return result


class _FakeAnalysis:
    def __init__(self, states: dict[str, AgentState]) -> None:
        self.states = states
        self.calls: list[str] = []
        self.options: list[dict[str, object]] = []

    def analyze(
        self,
        ticker: str,
        horizon: str,
        max_iterations: int = 2,
        allow_llm_planning: bool = True,
        force_refresh: bool = False,
    ) -> AgentState:
        del horizon
        self.calls.append(ticker)
        self.options.append({
            "max_iterations": max_iterations,
            "allow_llm_planning": allow_llm_planning,
            "force_refresh": force_refresh,
        })
        return self.states[ticker]


def test_dual_recommendation_returns_exact_balanced_count_when_policy_allows() -> None:
    tickers = ["FPT", "VNM", "HPG", "VCB", "SSI", "MBB"]
    states = {
        ticker: _state(
            ticker,
            RecommendationBucket.HIGH_REWARD_HIGH_RISK if index % 2 == 0
            else RecommendationBucket.SAFE_MODERATE_RETURN,
        )
        for index, ticker in enumerate(tickers)
    }
    analysis = _FakeAnalysis(states)
    result = RecommendationService(_FakeScreening(tickers), analysis).recommend(requested_count=4)

    assert result.universe_size == 100
    assert result.returned_count == 4
    assert len(result.high_reward) == 2
    assert len(result.safe_picks) == 2
    assert result.analyzed_count == 4
    assert analysis.calls == tickers[:4]
    assert result.deep_analysis_limits == [4]
    assert all(option["max_iterations"] == 0 for option in analysis.options)
    assert all(option["allow_llm_planning"] is False for option in analysis.options)


def test_dual_recommendation_reports_honest_shortfall() -> None:
    tickers = ["FPT", "VNM"]
    states = {
        ticker: _state(ticker, RecommendationBucket.SAFE_MODERATE_RETURN)
        for ticker in tickers
    }
    result = RecommendationService(_FakeScreening(tickers), _FakeAnalysis(states)).recommend(
        requested_count=4,
        max_deep_analysis=4,
    )
    assert result.returned_count == 2
    assert any("Requested 4" in warning for warning in result.warnings)


def test_dual_recommendation_expands_deep_analysis_until_count_is_met() -> None:
    tickers = ["FPT", "VNM", "HPG", "VCB", "SSI", "MBB", "ACB", "TCB"]
    states: dict[str, AgentState] = {}
    for index, ticker in enumerate(tickers):
        bucket = (
            RecommendationBucket.HIGH_REWARD_HIGH_RISK if index % 2 == 0
            else RecommendationBucket.SAFE_MODERATE_RETURN
        )
        state = _state(ticker, bucket)
        if index < 4:
            state.technical_result = state.technical_result.model_copy(update={"score": 6.5})
            state.risk_result = state.risk_result.model_copy(update={
                "risk_score": 5.0,
                "data_uncertainty_level": RiskLevel.HIGH,
            })
        states[ticker] = state

    analysis = _FakeAnalysis(states)
    result = RecommendationService(_FakeScreening(tickers), analysis).recommend(
        requested_count=4,
        max_deep_analysis=4,
    )
    assert result.returned_count == 4
    assert result.deep_analysis_limits == [4, 8]
    assert analysis.calls == tickers


def test_dual_recommendation_returns_labelled_reference_candidates_when_none_qualify() -> None:
    tickers = ["FPT", "VNM", "HPG", "VCB"]
    states = {}
    for ticker in tickers:
        state = _state(ticker, RecommendationBucket.SAFE_MODERATE_RETURN)
        state.risk_result = state.risk_result.model_copy(update={
            "risk_score": 5.5,
            "data_uncertainty_level": RiskLevel.HIGH,
        })
        state.final_decision = state.final_decision.model_copy(update={
            "recommendation": Recommendation.WATCH,
        })
        states[ticker] = state

    result = RecommendationService(_FakeScreening(tickers), _FakeAnalysis(states)).recommend(
        requested_count=4,
    )

    assert result.qualified_count == 0
    assert result.reference_count == 4
    assert result.returned_count == 4
    assert len(result.reference_only) == 4
    assert all(item.selection_tier.value == "REFERENCE_ONLY" for item in result.reference_only)
    assert all(item.policy_gaps for item in result.reference_only)
    assert any("reference-only" in warning for warning in result.warnings)


def test_dual_recommendation_deeply_analyzes_best_failed_screen_candidates_as_references() -> None:
    tickers = ["FPT", "VNM", "HPG"]
    states = {}
    for ticker in tickers:
        state = _state(ticker, RecommendationBucket.SAFE_MODERATE_RETURN)
        state.risk_result = state.risk_result.model_copy(update={
            "risk_score": 6.0,
            "data_uncertainty_level": RiskLevel.HIGH,
        })
        state.final_decision = state.final_decision.model_copy(update={
            "recommendation": Recommendation.WATCH,
        })
        states[ticker] = state

    result = RecommendationService(_AllFailedScreening(tickers), _FakeAnalysis(states)).recommend(
        requested_count=2,
    )

    assert result.screened_count == 0
    assert result.returned_count == 2
    assert result.reference_count == 2
    assert [item.ticker for item in result.reference_only] == ["FPT", "VNM"]
    assert any("near-screen" in warning for warning in result.warnings)
