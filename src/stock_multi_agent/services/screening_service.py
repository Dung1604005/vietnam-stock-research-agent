from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import ceil

from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.models.screening import (
    ProgressiveScreenRound, RankingDirection, ScreenCandidate, ScreenSpec, ScreenStatus, UniverseDefinition,
)
from stock_multi_agent.models.state import TraceEvent
from stock_multi_agent.providers.market_provider import MarketDataError
from stock_multi_agent.providers.resilience import is_rate_limit_error
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.universe_service import UniverseService


@dataclass
class ScreenRun:
    universe: UniverseDefinition
    spec: ScreenSpec
    candidates: list[ScreenCandidate] = field(default_factory=list)
    traces: list[TraceEvent] = field(default_factory=list)
    source_details: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    newest_as_of: object | None = None
    newest_fetched_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    price_as_of_aligned: bool = False
    progressive_rounds: list[ProgressiveScreenRound] = field(default_factory=list)
    selected_tickers: list[str] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(item.status is ScreenStatus.PASS for item in self.candidates)


class ScreeningService:
    """Bounded batch orchestration over the configured universe and cached provider."""

    def __init__(self, analysis: AnalysisService, universes: UniverseService) -> None:
        self.analysis = analysis
        self.universes = universes

    def run(self, spec: ScreenSpec, horizon: str) -> ScreenRun:
        universe = self.universes.select(spec.universe_key)
        if spec.as_of is not None:
            raise ValueError("Historical as-of screening is unavailable until a point-in-time universe/data source is attached.")
        tickers = universe.tickers[:spec.maximum_candidates]
        configured_order = {ticker: index for index, ticker in enumerate(tickers)}
        result = ScreenRun(universe=universe, spec=spec, started_at=datetime.now().astimezone())
        for index, ticker in enumerate(tickers):
            try:
                state = self.analysis.analyze_technical(
                    ticker,
                    horizon,
                    allow_llm_planning=False,
                    include_forecast=False,
                )
                if state.market_data is None or state.technical_result is None:
                    raise ValueError("Technical analysis returned no structured result")
                data = state.market_data
                technical = state.technical_result
                reasons = []
                if data.quality_report.status.value == "FAIL":
                    status = ScreenStatus.UNAVAILABLE
                    reasons.append("Excluded: market-data quality failed.")
                elif technical.signal not in spec.eligible_signals:
                    status = ScreenStatus.FAIL
                    reasons.append(f"Signal {technical.signal.value} is not eligible.")
                elif technical.score < spec.minimum_technical_score:
                    status = ScreenStatus.FAIL
                    reasons.append(f"Technical score {technical.score:.1f} is below {spec.minimum_technical_score:.1f}.")
                else:
                    status = ScreenStatus.PASS
                    reasons.append(f"Signal {technical.signal.value} and score {technical.score:.1f} meet screen policy.")
                if data.quality_report.status.value == "WARN":
                    reasons.append("Data-quality warnings are attached; this is not a recommendation.")
                result.candidates.append(ScreenCandidate(
                    ticker=ticker,
                    status=status,
                    technical_signal=technical.signal,
                    technical_score=technical.score,
                    as_of=data.as_of,
                    source_ids=[source.source_id for source in data.source_refs],
                    reasons=reasons,
                ))
                result.traces.extend(state.trace)
                result.source_details.extend(data.source_refs)
                if data.as_of and (result.newest_as_of is None or data.as_of > result.newest_as_of):
                    result.newest_as_of = data.as_of
                if result.newest_fetched_at is None or data.fetched_at > result.newest_fetched_at:
                    result.newest_fetched_at = data.fetched_at
            except (MarketDataError, ValueError) as exc:
                result.candidates.append(ScreenCandidate(
                    ticker=ticker, status=ScreenStatus.UNAVAILABLE,
                    reasons=[f"Excluded: {type(exc).__name__}: {exc}"],
                ))
                if is_rate_limit_error(exc):
                    result.warnings.append(
                        f"{ticker}: provider rate limit reached; stopped the screen to avoid additional requests."
                    )
                    for skipped_ticker in tickers[index + 1:]:
                        result.candidates.append(ScreenCandidate(
                            ticker=skipped_ticker, status=ScreenStatus.UNAVAILABLE,
                            reasons=["Not evaluated: provider rate limit was reached before this symbol."],
                        ))
                    break
                result.warnings.append(f"{ticker}: {exc}")
        comparable_dates = {item.as_of for item in result.candidates if item.status is not ScreenStatus.UNAVAILABLE and item.as_of}
        result.price_as_of_aligned = len(comparable_dates) == 1
        if spec.require_same_price_as_of and not result.price_as_of_aligned:
            result.warnings.append("Price as-of dates differ or are missing: technical ranking is withheld.")
            for item in result.candidates:
                if item.status is not ScreenStatus.UNAVAILABLE:
                    item.status = ScreenStatus.UNAVAILABLE
                    item.reasons.append("Excluded from ranking: price as-of is not aligned across the screen.")
        ranked = sorted(
            (item for item in result.candidates if item.status is ScreenStatus.PASS),
            key=lambda item: (
                item.technical_score if spec.ranking_direction is RankingDirection.LOWER_SCORE_FIRST else -(item.technical_score or 0),
                configured_order[item.ticker],
            ),
        )
        for rank, item in enumerate(ranked, start=1):
            item.rank = rank
        priority = {ScreenStatus.PASS: 0, ScreenStatus.FAIL: 1, ScreenStatus.UNAVAILABLE: 2}
        result.candidates.sort(key=lambda item: (
            priority[item.status], item.rank or 10_000, configured_order[item.ticker],
        ))
        result.completed_at = datetime.now().astimezone()
        return result

    def run_progressive(self, spec: ScreenSpec, horizon: str, target_count: int) -> ScreenRun:
        """Rank once, then halve the eligible pool until the requested top-K remains.

        Binary search is not valid for an unsorted stock universe. This method
        evaluates every bounded symbol exactly once, sorts by the disclosed
        technical policy, and applies successive halving without repeating data
        calls. Therefore no unevaluated symbol is silently assumed to be weak.
        """
        if not 1 <= target_count <= spec.maximum_candidates:
            raise ValueError("target_count must be within the bounded screening universe")
        result = self.run(spec, horizon)
        survivors = [item for item in result.candidates if item.status is ScreenStatus.PASS]
        round_number = 1
        while len(survivors) > target_count:
            input_count = len(survivors)
            keep_count = max(target_count, ceil(input_count / 2))
            eliminated = survivors[keep_count:]
            survivors = survivors[:keep_count]
            result.progressive_rounds.append(ProgressiveScreenRound(
                round_number=round_number,
                input_count=input_count,
                survivor_count=len(survivors),
                eliminated_tickers=[item.ticker for item in eliminated],
            ))
            result.traces.append(TraceEvent(
                agent=AgentName.MANAGER,
                phase="PLAN",
                action="SUCCESSIVE_HALVING",
                detail=(
                    f"Round {round_number}: retained {len(survivors)} of {input_count} "
                    f"technically ranked candidates; target={target_count}."
                ),
            ))
            round_number += 1
        if not result.progressive_rounds:
            result.progressive_rounds.append(ProgressiveScreenRound(
                round_number=1,
                input_count=len(survivors),
                survivor_count=len(survivors),
                eliminated_tickers=[],
            ))
        result.selected_tickers = [item.ticker for item in survivors[:target_count]]
        return result
