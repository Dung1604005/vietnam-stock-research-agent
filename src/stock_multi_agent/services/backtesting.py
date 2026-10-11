from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date
from statistics import fmean

import numpy as np

from stock_multi_agent.models.evaluation import (
    BacktestConfig,
    BacktestDataset,
    BacktestPeriod,
    BacktestReport,
    WalkForwardConfig,
    WalkForwardWindow,
)
from stock_multi_agent.models.data import DataKind


class BacktestInputError(ValueError):
    """The input cannot support an honest point-in-time backtest."""


class PointInTimeBacktestEngine:
    """Deterministic rank backtest; it never fetches live data or invokes an LLM."""

    def run(
        self,
        dataset: BacktestDataset,
        config: BacktestConfig,
        walk_forward_config: WalkForwardConfig | None = None,
    ) -> BacktestReport:
        self._validate(dataset, config)
        prices = self._price_index(dataset.prices)
        benchmark = self._session_prices(dataset.benchmark_prices, dataset.benchmark_ticker)
        signals_by_date = self._signal_index(dataset)
        universes = {snapshot.as_of: snapshot for snapshot in dataset.universe_snapshots}
        signal_dates = sorted(universes)
        periods: list[BacktestPeriod] = []
        failures: list[str] = []
        previous_weights: dict[str, float] = {}

        # The final rebalance has no known exit boundary, so it is excluded.
        for index, signal_date in enumerate(signal_dates[:-1]):
            next_signal_date = signal_dates[index + 1]
            snapshot = universes[signal_date]
            candidates = [signal for signal in signals_by_date.get(signal_date, []) if signal.ticker in snapshot.tickers]
            if not candidates:
                failures.append(f"{signal_date}: no eligible signals in the point-in-time universe")
                continue
            candidates.sort(key=lambda item: (-item.score, item.ticker))
            selected = [item.ticker for item in candidates[:config.top_n]]
            entry_session = self._session_after(prices, selected, signal_date, config.execution_lag_sessions)
            exit_session = self._session_after(prices, selected, next_signal_date, config.execution_lag_sessions)
            if entry_session is None or exit_session is None or exit_session <= entry_session:
                failures.append(f"{signal_date}: missing aligned executable prices for the holding period")
                continue
            if entry_session not in benchmark or exit_session not in benchmark:
                failures.append(f"{signal_date}: benchmark lacks an aligned executable price")
                continue
            returns = [prices[ticker][exit_session] / prices[ticker][entry_session] - 1 for ticker in selected]
            weights = {ticker: 1 / len(selected) for ticker in selected}
            turnover = sum(abs(weights.get(ticker, 0) - previous_weights.get(ticker, 0)) for ticker in set(weights) | set(previous_weights))
            transaction_cost = turnover * config.transaction_cost_bps / 10_000
            benchmark_return = benchmark[exit_session] / benchmark[entry_session] - 1
            periods.append(BacktestPeriod(
                signal_as_of=signal_date,
                entry_session=entry_session,
                exit_session=exit_session,
                selected_tickers=selected,
                gross_return=fmean(returns),
                transaction_cost=transaction_cost,
                net_return=fmean(returns) - transaction_cost,
                benchmark_return=benchmark_return,
                turnover=turnover,
            ))
            previous_weights = weights

        if not periods:
            raise BacktestInputError("No executable backtest periods remain after point-in-time validation.")
        net_returns = np.array([period.net_return for period in periods], dtype=float)
        benchmark_returns = np.array([period.benchmark_return for period in periods], dtype=float)
        excess_returns = net_returns - benchmark_returns
        metrics = self._metrics(net_returns, benchmark_returns, periods)
        confidence = self._confidence_intervals(excess_returns, config)
        walk_forward = self.walk_forward_windows(signal_dates, walk_forward_config) if walk_forward_config else []
        return BacktestReport(
            dataset_version=dataset.dataset_version,
            strategy_version=config.strategy_version,
            input_sha256=self._digest(dataset),
            config=config,
            periods=periods,
            metrics=metrics,
            confidence_intervals=confidence,
            failures=failures,
            walk_forward=walk_forward,
        )

    @staticmethod
    def walk_forward_windows(dates: list[date], config: WalkForwardConfig | None) -> list[WalkForwardWindow]:
        if config is None:
            return []
        ordered = sorted(set(dates))
        windows: list[WalkForwardWindow] = []
        start = 0
        fold = 1
        while start + config.training_windows + config.test_windows <= len(ordered):
            train_end = start + config.training_windows - 1
            test_start = train_end + 1
            test_end = test_start + config.test_windows - 1
            windows.append(WalkForwardWindow(
                fold=fold,
                train_start=ordered[start], train_end=ordered[train_end],
                test_start=ordered[test_start], test_end=ordered[test_end],
            ))
            start += config.step_windows
            fold += 1
        if not windows:
            raise BacktestInputError("Not enough universe snapshots for the requested walk-forward configuration.")
        return windows

    @staticmethod
    def _validate(dataset: BacktestDataset, config: BacktestConfig) -> None:
        if config.top_n < 1:
            raise BacktestInputError("top_n must be positive")
        for snapshot in dataset.universe_snapshots:
            if not snapshot.point_in_time_supported:
                raise BacktestInputError(
                    f"{snapshot.as_of}: universe is not point-in-time supported; refusing survivorship-biased backtest."
                )
            if snapshot.source_reference.as_of and snapshot.source_reference.as_of > snapshot.as_of:
                raise BacktestInputError(f"{snapshot.as_of}: universe source as_of is in the future")
            if snapshot.source_reference.data_kind is not DataKind.INSTRUMENT_UNIVERSE:
                raise BacktestInputError(f"{snapshot.as_of}: universe snapshot lacks instrument-universe provenance")
        snapshot_dates = {snapshot.as_of for snapshot in dataset.universe_snapshots}
        for signal in dataset.signals:
            if signal.as_of not in snapshot_dates:
                raise BacktestInputError(f"{signal.ticker}: signal date {signal.as_of} has no universe snapshot")
            if signal.available_at.date() > signal.as_of:
                raise BacktestInputError(f"{signal.ticker}: signal became available after its as_of date")
        all_prices = [*dataset.prices, *dataset.benchmark_prices]
        if config.require_adjusted_prices and any(not item.adjusted for item in all_prices):
            raise BacktestInputError("Adjusted-price policy is required but an unadjusted price was supplied.")
        if any(item.source_reference.as_of and item.source_reference.as_of > item.session for item in all_prices):
            raise BacktestInputError("A price source reference has an as_of date after its price session.")
        if any(item.source_reference.data_kind is not DataKind.OHLCV for item in all_prices):
            raise BacktestInputError("A price observation lacks OHLCV provenance.")
        policies = {signal.policy_version for signal in dataset.signals}
        if len(policies) != 1:
            raise BacktestInputError("A backtest dataset must contain exactly one signal policy version.")

    @staticmethod
    def _price_index(items):
        output: dict[str, dict[date, float]] = defaultdict(dict)
        for item in items:
            if item.session in output[item.ticker]:
                raise BacktestInputError(f"Duplicate price: {item.ticker} {item.session}")
            output[item.ticker][item.session] = item.close
        return output

    @staticmethod
    def _session_prices(items, expected_ticker: str) -> dict[date, float]:
        matching = [item for item in items if item.ticker == expected_ticker]
        if not matching:
            raise BacktestInputError(f"No benchmark prices found for {expected_ticker}")
        return {item.session: item.close for item in matching}

    @staticmethod
    def _signal_index(dataset: BacktestDataset):
        output = defaultdict(list)
        seen: set[tuple[date, str]] = set()
        for signal in dataset.signals:
            key = (signal.as_of, signal.ticker)
            if key in seen:
                raise BacktestInputError(f"Duplicate signal: {signal.ticker} {signal.as_of}")
            seen.add(key)
            output[signal.as_of].append(signal)
        return output

    @staticmethod
    def _session_after(prices, tickers: list[str], after: date, lag: int) -> date | None:
        common: set[date] | None = None
        for ticker in tickers:
            if ticker not in prices:
                return None
            available = {session for session in prices[ticker] if session > after}
            common = available if common is None else common.intersection(available)
        if not common or len(common) < lag:
            return None
        return sorted(common)[lag - 1]

    @staticmethod
    def _metrics(net_returns: np.ndarray, benchmark_returns: np.ndarray, periods: list[BacktestPeriod]):
        equity = np.cumprod(1 + net_returns)
        benchmark_equity = np.cumprod(1 + benchmark_returns)
        peaks = np.maximum.accumulate(equity)
        drawdown = equity / peaks - 1
        periods_per_year = 12
        annualized_return = float(equity[-1] ** (periods_per_year / len(net_returns)) - 1)
        annualized_volatility = float(np.std(net_returns, ddof=1) * np.sqrt(periods_per_year)) if len(net_returns) > 1 else 0.0
        sharpe = annualized_return / annualized_volatility if annualized_volatility else None
        return {
            "periods": len(periods),
            "gross_total_return": round(float(np.prod(1 + np.array([period.gross_return for period in periods])) - 1), 6),
            "net_total_return": round(float(equity[-1] - 1), 6),
            "benchmark_total_return": round(float(benchmark_equity[-1] - 1), 6),
            "active_total_return": round(float(equity[-1] - benchmark_equity[-1]), 6),
            "annualized_return": round(annualized_return, 6),
            "annualized_volatility": round(annualized_volatility, 6),
            "sharpe_zero_rate": round(sharpe, 6) if sharpe is not None else None,
            "max_drawdown": round(float(drawdown.min()), 6),
            "hit_rate": round(float(np.mean(net_returns > 0)), 6),
            "mean_turnover": round(float(np.mean([period.turnover for period in periods])), 6),
            "total_transaction_cost": round(float(sum(period.transaction_cost for period in periods)), 6),
        }

    @staticmethod
    def _confidence_intervals(excess_returns: np.ndarray, config: BacktestConfig):
        if len(excess_returns) < 2:
            return {"mean_period_excess_return_95": None}
        rng = np.random.default_rng(config.random_seed)
        sampled_means = np.array([
            rng.choice(excess_returns, size=len(excess_returns), replace=True).mean()
            for _ in range(config.bootstrap_samples)
        ])
        lower, upper = np.quantile(sampled_means, [0.025, 0.975])
        return {"mean_period_excess_return_95": (round(float(lower), 6), round(float(upper), 6))}

    @staticmethod
    def _digest(dataset: BacktestDataset) -> str:
        payload = dataset.model_dump(mode="json")
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
