from __future__ import annotations

from datetime import date
from time import perf_counter

from stock_multi_agent.models.analysis import Recommendation, RiskLevel, Signal
from stock_multi_agent.models.screening import (
    CandidateSelectionTier,
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
        started_at = perf_counter()
        if not 1 <= requested_count <= 100:
            raise ValueError("requested_count must be between 1 and 100")
        quick_limit = min(100, max(requested_count, max_quick_screen))
        # Fast bounded mode: one deep-analysis slot per requested result. The
        # reference-only tier guarantees useful output without doubling the
        # expensive full-graph workload merely to satisfy strict buckets.
        deep_limit = max_deep_analysis or requested_count
        deep_limit = min(quick_limit, max(requested_count, deep_limit))
        spec = ScreenSpec(
            universe_key=universe_key,
            maximum_candidates=quick_limit,
            minimum_technical_score=5.0,
            eligible_signals=[Signal.BUY, Signal.HOLD],
            refresh_stale_data=False,
            candidate_selection_policy="progressive_successive_halving_v1",
            factor_policy_version="screen-v4-fast-recommendation",
        )
        quick_started_at = perf_counter()
        screen = self.screening.run_progressive(spec, horizon, target_count=deep_limit)
        quick_screen_seconds = perf_counter() - quick_started_at
        warnings = list(screen.warnings)
        high_pool: list[RecommendationCandidate] = []
        safe_pool: list[RecommendationCandidate] = []
        reference_pool: list[RecommendationCandidate] = []
        analyzed_count = 0
        candidate_queue = [
            item.ticker for item in screen.candidates if item.status is ScreenStatus.PASS
        ]
        if len(candidate_queue) < deep_limit:
            near_screen = sorted(
                (
                    item for item in screen.candidates
                    if item.status is ScreenStatus.FAIL and item.technical_score is not None
                ),
                key=lambda item: (-(item.technical_score or 0.0), item.ticker),
            )
            fallback_tickers = [
                item.ticker for item in near_screen
                if item.ticker not in candidate_queue
            ][:deep_limit - len(candidate_queue)]
            if fallback_tickers:
                candidate_queue.extend(fallback_tickers)
                warnings.append(
                    f"Quick screen supplied only {screen.pass_count} strict survivors; "
                    f"{len(fallback_tickers)} highest-scoring near-screen symbols were sent to deep "
                    "analysis for REFERENCE_ONLY fallback consideration."
                )
        current_limit = min(deep_limit, len(candidate_queue))
        deep_analysis_limits = [current_limit] if current_limit else []
        cursor = 0
        rate_limited = False
        deep_refreshed_tickers: list[str] = []
        screen_candidate_by_ticker = {item.ticker: item for item in screen.candidates}
        deep_started_at = perf_counter()
        while cursor < len(candidate_queue) and not self._enough_candidates(
            high_pool, safe_pool, requested_count,
        ):
            while cursor < current_limit:
                ticker = candidate_queue[cursor]
                cursor += 1
                try:
                    screen_candidate = screen_candidate_by_ticker[ticker]
                    screen_as_of = getattr(screen_candidate, "as_of", None)
                    force_refresh = bool(
                        screen_as_of is None
                        or (date.today() - screen_as_of).days > 3
                    )
                    if force_refresh:
                        deep_refreshed_tickers.append(ticker)
                    state = self.analysis.analyze(
                        ticker,
                        horizon,
                        max_iterations=0,
                        allow_llm_planning=False,
                        force_refresh=force_refresh,
                    )
                    analyzed_count += 1
                    candidate = self._classify(ticker, state)
                    if candidate is None:
                        reference = self._reference_candidate(ticker, state)
                        if reference is not None:
                            reference_pool.append(reference)
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
        deep_analysis_seconds = perf_counter() - deep_started_at
        qualified_count = len(high_reward) + len(safe_picks)
        reference_only = self._select_references(reference_pool, requested_count - qualified_count)
        returned_count = qualified_count + len(reference_only)
        if reference_only:
            warnings.append(
                f"Only {qualified_count} symbols passed the full policy; added {len(reference_only)} "
                "clearly labelled reference-only symbols ranked from completed deep analyses."
            )
        if deep_refreshed_tickers:
            warnings.append(
                f"Refreshed {len(deep_refreshed_tickers)} shortlisted symbols before deep analysis; "
                "the remaining universe was not refreshed during this prompt."
            )
        if returned_count < requested_count:
            warnings.append(
                f"Requested {requested_count} symbols, but only {returned_count} had enough completed "
                f"technical/fundamental/risk evidence after {analyzed_count} deep analyses."
            )
        return DualRecommendation(
            high_reward=high_reward,
            safe_picks=safe_picks,
            reference_only=reference_only,
            universe_key=screen.universe.key,
            universe_size=len(screen.universe.tickers),
            screened_count=screen.pass_count,
            analyzed_count=analyzed_count,
            requested_count=requested_count,
            returned_count=returned_count,
            qualified_count=qualified_count,
            reference_count=len(reference_only),
            horizon=horizon,
            progressive_rounds=screen.progressive_rounds,
            deep_analysis_limits=deep_analysis_limits,
            refreshed_tickers=list(dict.fromkeys([
                *screen.refreshed_tickers,
                *deep_refreshed_tickers,
            ])),
            price_as_of_comparable=screen.price_as_of_comparable,
            quick_screen_seconds=round(quick_screen_seconds, 3),
            deep_analysis_seconds=round(deep_analysis_seconds, 3),
            total_seconds=round(perf_counter() - started_at, 3),
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
            selection_tier=CandidateSelectionTier.QUALIFIED,
            veto=risk.veto,
        )

    @staticmethod
    def _reference_candidate(ticker: str, state: AgentState) -> RecommendationCandidate | None:
        technical = state.technical_result
        fundamental = state.fundamental_result
        risk = state.risk_result
        final = state.final_decision
        data = state.market_data
        if not all((technical, fundamental, risk, final, data)):
            return None
        volatility_value = technical.diagnostics.get("annualized_volatility", 0.0)
        volatility = float(volatility_value) if isinstance(volatility_value, (int, float)) else 0.0
        gaps: list[str] = []
        if risk.veto:
            gaps.append("Risk Agent applied a veto")
        if final.recommendation is Recommendation.AVOID:
            gaps.append("Manager conclusion is AVOID")
        if risk.data_uncertainty_level is RiskLevel.HIGH:
            gaps.append("Data uncertainty is HIGH")
        if technical.score < 7.0:
            gaps.append("Technical score is below the high-reward threshold of 7.0")
        if risk.risk_score >= 5.0 and volatility < 0.30:
            gaps.append("Does not meet the high-risk/high-volatility bucket combination")
        if risk.risk_score >= 5.0:
            gaps.append("Risk score is not below the safe-bucket threshold of 5.0")
        if fundamental.signal not in {Signal.BUY, Signal.HOLD}:
            gaps.append("Fundamental signal is not BUY/HOLD")
        if not gaps:
            gaps.append("Does not match either strict dual-recommendation bucket")

        evidence = [
            *(f"{item.label}: {item.value}" for item in technical.evidence[:1]),
            *(f"{item.label}: {item.value}" for item in fundamental.evidence[:1]),
            risk.reasoning_summary,
        ][:3]
        return RecommendationCandidate(
            ticker=ticker,
            bucket=RecommendationBucket.UNCLASSIFIED,
            rank_in_bucket=1,
            technical_score=technical.score,
            fundamental_score=fundamental.score,
            risk_score=risk.risk_score,
            risk_level=risk.risk_level,
            recommendation=final.recommendation,
            confidence=final.confidence,
            volatility=volatility,
            reasoning=(
                "Reference-only candidate from the ranked technical shortlist; it did not pass every "
                "full-analysis rule. Review the disclosed policy gaps before further research."
            ),
            evidence_highlights=evidence,
            as_of=data.as_of,
            source_ids=list(dict.fromkeys(source.source_id for source in data.source_refs)),
            selection_tier=CandidateSelectionTier.REFERENCE_ONLY,
            policy_gaps=gaps,
            veto=risk.veto,
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

    @staticmethod
    def _select_references(
        candidates: list[RecommendationCandidate],
        count: int,
    ) -> list[RecommendationCandidate]:
        if count <= 0:
            return []
        recommendation_priority = {
            Recommendation.BUY: 0,
            Recommendation.HOLD: 1,
            Recommendation.WATCH: 2,
            Recommendation.AVOID: 3,
        }
        ranked = sorted(
            candidates,
            key=lambda item: (
                item.veto,
                recommendation_priority[item.recommendation],
                item.risk_score,
                -(item.technical_score + item.fundamental_score),
                -item.confidence,
                item.ticker,
            ),
        )[:count]
        return [item.model_copy(update={"rank_in_bucket": rank}) for rank, item in enumerate(ranked, 1)]
