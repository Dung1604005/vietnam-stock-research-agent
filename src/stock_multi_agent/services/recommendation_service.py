from __future__ import annotations

from stock_multi_agent.models.analysis import Recommendation, RiskLevel, Signal
from stock_multi_agent.models.screening import (
    DualRecommendation,
    RecommendationBucket,
    RecommendationCandidate,
    ScreenSpec,
    ScreenStatus,
)
from stock_multi_agent.providers.resilience import is_rate_limit_error
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.screening_service import ScreeningService


class RecommendationService:
    """Two-phase, deterministic stock research shortlist.

    Phase 1 evaluates the bounded universe with cheap technical analysis and
    successive halving. Phase 2 runs the full graph only for survivors and
    classifies them into two disclosed risk/return buckets.
    """

    def __init__(self, screening: ScreeningService, analysis: AnalysisService) -> None:
        self.screening = screening
        self.analysis = analysis

    def recommend(
        self,
        *,
        horizon: str = "3 months",
        universe_key: str = "default",
        requested_count: int = 6,
        max_quick_screen: int = 100,
        max_deep_analysis: int | None = None,
    ) -> DualRecommendation:
        if not 1 <= requested_count <= 100:
            raise ValueError("requested_count must be between 1 and 100")
        quick_limit = min(100, max(requested_count, max_quick_screen))
        deep_limit = max_deep_analysis or max(10, requested_count * 2)
        deep_limit = min(quick_limit, max(requested_count, deep_limit))
        spec = ScreenSpec(
            universe_key=universe_key,
            maximum_candidates=quick_limit,
            minimum_technical_score=5.0,
            eligible_signals=[Signal.BUY, Signal.HOLD],
            candidate_selection_policy="progressive_successive_halving_v1",
            factor_policy_version="screen-v3-progressive",
        )
        screen = self.screening.run_progressive(spec, horizon, target_count=deep_limit)
        warnings = list(screen.warnings)
        high_pool: list[RecommendationCandidate] = []
        safe_pool: list[RecommendationCandidate] = []
        analyzed_count = 0
        candidate_queue = [
            item.ticker for item in screen.candidates if item.status is ScreenStatus.PASS
        ]
        current_limit = min(deep_limit, len(candidate_queue))
        deep_analysis_limits = [current_limit] if current_limit else []
        cursor = 0
        rate_limited = False
        while cursor < len(candidate_queue) and not self._enough_candidates(
            high_pool, safe_pool, requested_count,
        ):
            while cursor < current_limit:
                ticker = candidate_queue[cursor]
                cursor += 1
                try:
                    state = self.analysis.analyze(ticker, horizon, max_iterations=2)
                    analyzed_count += 1
                    candidate = self._classify(ticker, state)
                    if candidate is None:
                        continue
                    if candidate.bucket is RecommendationBucket.HIGH_REWARD_HIGH_RISK:
                        high_pool.append(candidate)
                    else:
                        safe_pool.append(candidate)
                    if self._enough_candidates(high_pool, safe_pool, requested_count):
                        break
                except Exception as exc:
                    warnings.append(f"{ticker}: deep analysis unavailable ({type(exc).__name__}: {exc})")
                    if is_rate_limit_error(exc):
                        warnings.append("Deep analysis stopped because the data provider rate limit was reached.")
                        rate_limited = True
                        break
            if rate_limited or self._enough_candidates(high_pool, safe_pool, requested_count):
                break
            if current_limit >= len(candidate_queue):
                break
            current_limit = min(len(candidate_queue), max(current_limit + 1, current_limit * 2))
            deep_analysis_limits.append(current_limit)

        high_reward, safe_picks = self._select_exact(high_pool, safe_pool, requested_count)
        returned_count = len(high_reward) + len(safe_picks)
        if returned_count < requested_count:
            warnings.append(
                f"Requested {requested_count} recommendations, but only {returned_count} passed the full "
                f"technical/fundamental/risk policy after {analyzed_count} deep analyses."
            )
        return DualRecommendation(
            high_reward=high_reward,
            safe_picks=safe_picks,
            universe_key=screen.universe.key,
            universe_size=len(screen.universe.tickers),
            screened_count=screen.pass_count,
            analyzed_count=analyzed_count,
            requested_count=requested_count,
            returned_count=returned_count,
            horizon=horizon,
            progressive_rounds=screen.progressive_rounds,
            deep_analysis_limits=deep_analysis_limits,
            warnings=list(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _classify(ticker: str, state: AgentState) -> RecommendationCandidate | None:
        technical = state.technical_result
        fundamental = state.fundamental_result
        risk = state.risk_result
        final = state.final_decision
        data = state.market_data
        if not all((technical, fundamental, risk, final, data)):
            return None
        if risk.veto or final.recommendation is Recommendation.AVOID:
            return None
        volatility_value = technical.diagnostics.get("annualized_volatility", 0.0)
        volatility = float(volatility_value) if isinstance(volatility_value, (int, float)) else 0.0

        if technical.score >= 7.0 and risk.risk_score >= 5.0 and volatility >= 0.30:
            bucket = RecommendationBucket.HIGH_REWARD_HIGH_RISK
            reasoning = (
                f"Động lượng kỹ thuật mạnh ({technical.score:.1f}/10) đi kèm rủi ro "
                f"{risk.risk_score:.1f}/10 và biến động {volatility:.1%}."
            )
        elif (
            risk.risk_score < 5.0
            and risk.data_uncertainty_level is not RiskLevel.HIGH
            and fundamental.signal in {Signal.BUY, Signal.HOLD}
        ):
            bucket = RecommendationBucket.SAFE_MODERATE_RETURN
            reasoning = (
                f"Rủi ro đo lường thấp hơn ({risk.risk_score:.1f}/10), cơ bản "
                f"{fundamental.signal.value} và không có veto."
            )
        else:
            return None

        evidence = [
            *(f"{item.label}: {item.value}" for item in technical.evidence[:1]),
            *(f"{item.label}: {item.value}" for item in fundamental.evidence[:1]),
            risk.reasoning_summary,
        ][:3]
        return RecommendationCandidate(
            ticker=ticker,
            bucket=bucket,
            rank_in_bucket=1,
            technical_score=technical.score,
            fundamental_score=fundamental.score,
            risk_score=risk.risk_score,
            risk_level=risk.risk_level,
            recommendation=final.recommendation,
            confidence=final.confidence,
            volatility=volatility,
            reasoning=reasoning,
            evidence_highlights=evidence,
            as_of=data.as_of,
            source_ids=list(dict.fromkeys(source.source_id for source in data.source_refs)),
        )

    @staticmethod
    def _enough_candidates(
        high_pool: list[RecommendationCandidate],
        safe_pool: list[RecommendationCandidate],
        requested_count: int,
    ) -> bool:
        if len(high_pool) + len(safe_pool) < requested_count:
            return False
        return requested_count == 1 or bool(high_pool and safe_pool)

    @staticmethod
    def _select_exact(
        high_pool: list[RecommendationCandidate],
        safe_pool: list[RecommendationCandidate],
        requested_count: int,
    ) -> tuple[list[RecommendationCandidate], list[RecommendationCandidate]]:
        high_sorted = sorted(
            high_pool,
            key=lambda item: (-(item.technical_score + item.fundamental_score), -item.confidence, item.ticker),
        )
        safe_sorted = sorted(
            safe_pool,
            key=lambda item: (item.risk_score, -item.fundamental_score, -item.technical_score, item.ticker),
        )
        high_quota = requested_count // 2
        safe_quota = requested_count - high_quota
        chosen_high = high_sorted[:high_quota]
        chosen_safe = safe_sorted[:safe_quota]
        remaining = requested_count - len(chosen_high) - len(chosen_safe)
        if remaining:
            high_left = high_sorted[len(chosen_high):]
            safe_left = safe_sorted[len(chosen_safe):]
            fill = sorted(
                [*high_left, *safe_left],
                key=lambda item: (-item.confidence, -item.technical_score, item.risk_score, item.ticker),
            )[:remaining]
            chosen_high.extend(item for item in fill if item.bucket is RecommendationBucket.HIGH_REWARD_HIGH_RISK)
            chosen_safe.extend(item for item in fill if item.bucket is RecommendationBucket.SAFE_MODERATE_RETURN)
        chosen_high = [item.model_copy(update={"rank_in_bucket": rank}) for rank, item in enumerate(chosen_high, 1)]
        chosen_safe = [item.model_copy(update={"rank_in_bucket": rank}) for rank, item in enumerate(chosen_safe, 1)]
        return chosen_high, chosen_safe
