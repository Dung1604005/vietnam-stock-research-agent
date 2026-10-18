from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from datetime import date, datetime
from math import ceil
from collections import OrderedDict
from copy import deepcopy
from time import monotonic
from stock_multi_agent.services.request_telemetry import BudgetExceeded, current_metrics

from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.models.screening import (
    ProgressiveScreenRound, RankingDirection, ScreenCandidate, ScreenSpec, ScreenStatus, UniverseDefinition,
)
from stock_multi_agent.models.state import TraceEvent
from stock_multi_agent.providers.market_provider import MarketDataError
from stock_multi_agent.providers.resilience import is_rate_limit_error
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.universe_service import UniverseService
from stock_multi_agent.services.screen_filters import validate_filter_keys, matches_filter


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
    price_as_of_comparable: bool = False
    progressive_rounds: list[ProgressiveScreenRound] = field(default_factory=list)
    selected_tickers: list[str] = field(default_factory=list)
    refreshed_tickers: list[str] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(item.status is ScreenStatus.PASS for item in self.candidates)


class ScreeningService:
    """Bounded batch orchestration over the configured universe and cached provider."""

    def __init__(
        self,
        analysis: AnalysisService,
        universes: UniverseService,
        today_provider: Callable[[], date] | None = None,
    ) -> None:
        self.analysis = analysis
        self.universes = universes
        self.today_provider = today_provider or date.today
        self._screen_cache = OrderedDict()

    def run(self, spec: ScreenSpec, horizon: str) -> ScreenRun:
        validate_filter_keys(spec.filters)
        universe = self.universes.select(spec.universe_key)
        if spec.as_of is not None:
            raise ValueError("Historical as-of screening is unavailable until a point-in-time universe/data source is attached.")
        tickers = universe.tickers[:spec.maximum_candidates]
        configured_order = {ticker: index for index, ticker in enumerate(tickers)}
        result = ScreenRun(universe=universe, spec=spec, started_at=datetime.now().astimezone())
        for index, ticker in enumerate(tickers):
            try:
                if current_metrics():
                    current_metrics().consume('deadline')
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
                candidate = ScreenCandidate(
                    ticker=ticker,
                    status=status,
                    technical_signal=technical.signal,
                    technical_score=technical.score,
                    as_of=data.as_of,
                    source_ids=[source.source_id for source in data.source_refs],
                    reasons=reasons,
                )
                self._apply_rsi_filters(candidate, data, spec)
                result.candidates.append(candidate)
                result.traces.extend(state.trace)
                result.source_details.extend(data.source_refs)
                if data.as_of and (result.newest_as_of is None or data.as_of > result.newest_as_of):
                    result.newest_as_of = data.as_of
                if result.newest_fetched_at is None or data.fetched_at > result.newest_fetched_at:
                    result.newest_fetched_at = data.fetched_at
            except (MarketDataError, ValueError, BudgetExceeded) as exc:
                result.candidates.append(ScreenCandidate(
                    ticker=ticker, status=ScreenStatus.UNAVAILABLE,
                    reasons=[f"Excluded: {type(exc).__name__}: {exc}"],
                ))
                if is_rate_limit_error(exc) or isinstance(exc, BudgetExceeded):
                    reason = 'request budget reached' if isinstance(exc, BudgetExceeded) else 'provider rate limit reached'
                    result.warnings.append(
                        f"{ticker}: {reason}; stopped the screen to avoid additional requests."
                    )
                    for skipped_ticker in tickers[index + 1:]:
                        result.candidates.append(ScreenCandidate(
                            ticker=skipped_ticker, status=ScreenStatus.UNAVAILABLE,
                            reasons=["Not evaluated: provider rate limit was reached before this symbol."],
                        ))
                    break
                result.warnings.append(f"{ticker}: {exc}")
        available = [item for item in result.candidates if item.status is not ScreenStatus.UNAVAILABLE]
        dated = [item for item in available if item.as_of is not None]
        if spec.require_same_price_as_of and available and spec.refresh_stale_data:
            today = self.today_provider()
            newest_before_refresh = max(
                (item.as_of for item in dated if item.as_of is not None),
                default=None,
            )
            refresh_targets = [
                item for item in available
                if item.as_of is None
                or (
                    newest_before_refresh is not None
                    and (newest_before_refresh - item.as_of).days > spec.max_price_as_of_lag_days
                )
                or (
                    item.as_of is not None
                    and spec.max_price_age_days is not None
                    and (today - item.as_of).days > spec.max_price_age_days
                )
            ]
            for item in refresh_targets:
                try:
                    result.refreshed_tickers.append(item.ticker)
                    result.traces.append(TraceEvent(
                        agent=AgentName.DATA,
                        phase="ACT",
                        action="REFRESH_STALE_DATA",
                        detail=(
                            f"Bypass cache once for stale/missing as-of data: {item.ticker}; "
                            f"tolerance={spec.max_price_as_of_lag_days} days."
                        ),
                        tool="get_stock_data",
                    ))
                    refreshed = self.analysis.analyze_technical(
                        item.ticker,
                        horizon,
                        allow_llm_planning=False,
                        include_forecast=False,
                        force_refresh=True,
                    )
                    if refreshed.market_data is None or refreshed.technical_result is None:
                        raise ValueError("Refreshed technical analysis returned no structured result")
                    data = refreshed.market_data
                    technical = refreshed.technical_result
                    item.technical_signal = technical.signal
                    item.technical_score = technical.score
                    item.as_of = data.as_of
                    item.source_ids = [source.source_id for source in data.source_refs]
                    item.reasons = []
                    if data.quality_report.status.value == "FAIL":
                        item.status = ScreenStatus.UNAVAILABLE
                        item.reasons.append("Excluded after refresh: market-data quality failed.")
                    elif technical.signal not in spec.eligible_signals:
                        item.status = ScreenStatus.FAIL
                        item.reasons.append(f"Signal {technical.signal.value} is not eligible after refresh.")
                    elif technical.score < spec.minimum_technical_score:
                        item.status = ScreenStatus.FAIL
                        item.reasons.append(
                            f"Technical score {technical.score:.1f} is below "
                            f"{spec.minimum_technical_score:.1f} after refresh."
                        )
                    else:
                        item.status = ScreenStatus.PASS
                        item.reasons.append(
                            f"Refreshed data: signal {technical.signal.value} and score "
                            f"{technical.score:.1f} meet screen policy."
                        )
                    self._apply_rsi_filters(item, data, spec)
                    result.traces.extend(refreshed.trace)
                    result.source_details.extend(data.source_refs)
                    if result.newest_fetched_at is None or data.fetched_at > result.newest_fetched_at:
                        result.newest_fetched_at = data.fetched_at
                except (MarketDataError, ValueError) as exc:
                    item.status = ScreenStatus.UNAVAILABLE
                    item.reasons.append(f"Refresh failed: {type(exc).__name__}: {exc}")
                    result.warnings.append(f"{item.ticker}: stale-data refresh failed ({exc})")
                    if is_rate_limit_error(exc):
                        result.warnings.append(
                            "Stopped stale-data refresh because the provider rate limit was reached."
                        )
                        break

        available = [item for item in result.candidates if item.status is not ScreenStatus.UNAVAILABLE]
        dated = [item for item in available if item.as_of is not None]
        comparable_dates = {item.as_of for item in dated}
        result.price_as_of_aligned = bool(dated) and len(comparable_dates) == 1 and len(dated) == len(available)
        if spec.require_same_price_as_of:
            if not dated:
                result.warnings.append("Price as-of dates are missing: no candidate can be ranked.")
                for item in available:
                    item.status = ScreenStatus.UNAVAILABLE
                    item.reasons.append("Excluded from ranking: price as-of is missing after refresh.")
            else:
                newest = max(item.as_of for item in dated if item.as_of is not None)
                relative_stale = [
                    item for item in dated
                    if item.as_of is not None
                    and (newest - item.as_of).days > spec.max_price_as_of_lag_days
                ]
                absolute_stale = [
                    item for item in dated
                    if item.as_of is not None
                    and spec.max_price_age_days is not None
                    and (self.today_provider() - item.as_of).days > spec.max_price_age_days
                ]
                stale = list({item.ticker: item for item in [*relative_stale, *absolute_stale]}.values())
                missing_dates = [item for item in available if item.as_of is None]
                for item in missing_dates:
                    item.status = ScreenStatus.UNAVAILABLE
                    item.reasons.append("Excluded from ranking: price as-of is missing after refresh.")
                for item in stale:
                    relative_lag = (newest - item.as_of).days if item.as_of is not None else -1
                    age = (self.today_provider() - item.as_of).days if item.as_of is not None else -1
                    item.status = ScreenStatus.UNAVAILABLE
                    item.reasons.append(
                        f"Excluded from ranking: refreshed price data remains stale "
                        f"(relative lag={relative_lag} days, age={age} days)."
                    )
                remaining = [
                    item for item in result.candidates
                    if item.status is not ScreenStatus.UNAVAILABLE and item.as_of is not None
                ]
                result.price_as_of_comparable = bool(remaining)
                if stale or missing_dates:
                    result.warnings.append(
                        f"Excluded {len(stale) + len(missing_dates)} stale/missing-date symbols after refresh; "
                        f"ranking continues for data within {spec.max_price_as_of_lag_days} days of {newest}."
                    )
                elif not result.price_as_of_aligned:
                    result.warnings.append(
                        f"Price as-of dates differ but remain within the "
                        f"{spec.max_price_as_of_lag_days}-day comparison tolerance."
                    )
        else:
            result.price_as_of_comparable = bool(available)
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

    @staticmethod
    def _apply_rsi_filters(candidate, data, spec):
        constraints = {field: value for field, value in spec.filters.items() if field.startswith('rsi_')}
        if not constraints or candidate.status is not ScreenStatus.PASS:
            return
        from stock_multi_agent.tools.technical_indicators import TechnicalIndicatorToolkit
        try:
            value = TechnicalIndicatorToolkit().rsi([bar.close for bar in data.bars], 14)
            candidate.filter_values['rsi'] = value
            for field, threshold in constraints.items():
                op = field.rsplit('_', 1)[1]
                if not matches_filter(value, op, threshold):
                    candidate.status = ScreenStatus.FAIL
                    candidate.reasons.append(f'RSI14={value:.6g} does not meet {field}={threshold}.')
                else:
                    candidate.reasons.append(f'RSI14={value:.6g} meets {field}={threshold}.')
        except ValueError:
            candidate.status = ScreenStatus.UNAVAILABLE
            candidate.reasons.append('RSI14 unavailable: at least 15 finite closing prices are required.')

    def run_progressive(self, spec: ScreenSpec, horizon: str, target_count: int) -> ScreenRun:
        """Rank once, then halve the eligible pool until the requested top-K remains.

        Binary search is not valid for an unsorted stock universe. This method
        evaluates every bounded symbol exactly once, sorts by the disclosed
        technical policy, and applies successive halving without repeating data
        calls. Therefore no unevaluated symbol is silently assumed to be weak.
        """
        if not 1 <= target_count <= spec.maximum_candidates:
            raise ValueError("target_count must be within the bounded screening universe")
        validate_filter_keys(spec.filters)
        key = (spec.model_dump_json(), horizon, str(self.today_provider()), tuple(self.universes.select(spec.universe_key).tickers))
        cached = self._screen_cache.get(key)
        if cached and monotonic() - cached[0] < 120:
            result = deepcopy(cached[1])
            if current_metrics():
                current_metrics().cache_hits += 1
        else:
            result = self.run(spec, horizon)
            if not any('budget reached' in warning or 'rate limit' in warning for warning in result.warnings):
                self._screen_cache[key] = (monotonic(), deepcopy(result))
                while len(self._screen_cache) > 8:
                    self._screen_cache.popitem(last=False)
        enrichment_filters = {field: value for field, value in spec.filters.items() if not field.startswith('rsi_')}
        if enrichment_filters or spec.exchange:
            # Enrich only a ranked shortlist, not all 100 symbols. Explicitly
            # disclose which survivors were not evaluated under custom filters.
            candidates = [item for item in result.candidates if item.status is ScreenStatus.PASS]
            limit = min(100, max(20, target_count * 3))
            for index, item in enumerate(candidates):
                if index >= limit:
                    item.status = ScreenStatus.UNAVAILABLE
                    item.reasons.append('Custom-filter enrichment budget exhausted before this ranked candidate.')
                    continue
                try:
                    data = self.analysis.fetch_data(item.ticker, horizon, include_financials=bool(enrichment_filters), include_forecast=False).market_data
                    if data is None or data.as_of != item.as_of:
                        raise ValueError('Filter data and technical snapshot are not aligned')
                    if spec.exchange and (not data.instrument or data.instrument.exchange != spec.exchange):
                        raise ValueError('Exchange missing or does not match requested exchange')
                    for field, threshold in enrichment_filters.items():
                        metric, operator = field.rsplit('_', 1)
                        value = getattr(data.financials, metric, None)
                        item.filter_values[metric] = value
                        boundary = threshold / 100 if metric == 'roe' else threshold
                        if not matches_filter(value, operator, boundary):
                            item.status = ScreenStatus.FAIL
                            item.reasons.append(f'Filter {field}={threshold} failed or metric unavailable.')
                    result.source_details.extend(data.source_refs)
                except (MarketDataError, ValueError, BudgetExceeded) as exc:
                    item.status = ScreenStatus.UNAVAILABLE
                    item.reasons.append(f'Custom filters unavailable: {type(exc).__name__}')
            result.warnings.append(f'Custom filters checked at most {limit} technically ranked survivors; ROE thresholds use percentage points. Not an exhaustive market-wide financial screen.')
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
