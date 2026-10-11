from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from stock_multi_agent.models.data import DataKind, Instrument, PriceAdjustment, SourceReference
from stock_multi_agent.models.evaluation import (
    BacktestConfig,
    BacktestDataset,
    ForwardEvaluationCapture,
    HistoricalPrice,
    PointInTimeUniverseSnapshot,
    SignalSnapshot,
    WalkForwardConfig,
)
from stock_multi_agent.services.backtesting import BacktestInputError, PointInTimeBacktestEngine
from stock_multi_agent.services.forward_evaluation import ForwardDatasetBuilder, ForwardEvaluationRecorder
from stock_multi_agent.providers.universe_provider import InstrumentUniverseSnapshot
from conftest import conflict_market_data


def _reference(kind: DataKind, as_of: date) -> SourceReference:
    return SourceReference(
        provider="fixture", source_id=f"fixture:{kind.value}:{as_of.isoformat()}", data_kind=kind,
        fetched_at=datetime.combine(as_of, datetime.min.time(), tzinfo=timezone.utc), as_of=as_of,
    )


def point_in_time_dataset() -> BacktestDataset:
    rebalance_dates = [date(2024, 1, 2) + timedelta(days=7 * index) for index in range(5)]
    tickers = ["AAA", "BBB"]
    snapshots = [PointInTimeUniverseSnapshot(
        as_of=value, tickers=tickers, source_reference=_reference(DataKind.INSTRUMENT_UNIVERSE, value),
    ) for value in rebalance_dates]
    signals = [SignalSnapshot(
        ticker=ticker, as_of=value, available_at=datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc),
        score=9.0 if ticker == "AAA" else 5.0, policy_version="fixture-v1", source_ids=[f"fixture:signal:{ticker}"],
    ) for value in rebalance_dates for ticker in tickers]
    sessions = [date(2024, 1, 3) + timedelta(days=7 * index) for index in range(5)]
    prices: list[HistoricalPrice] = []
    benchmark: list[HistoricalPrice] = []
    for index, session in enumerate(sessions):
        prices.extend([
            HistoricalPrice(ticker="AAA", session=session, close=100 + 10 * index, adjusted=True, source_reference=_reference(DataKind.OHLCV, session)),
            HistoricalPrice(ticker="BBB", session=session, close=100 - 2 * index, adjusted=True, source_reference=_reference(DataKind.OHLCV, session)),
        ])
        benchmark.append(HistoricalPrice(
            ticker="VNINDEX", session=session, close=100 + index, adjusted=True, source_reference=_reference(DataKind.OHLCV, session),
        ))
    return BacktestDataset(
        dataset_version="fixture-pit-v1", universe_snapshots=snapshots, signals=signals,
        prices=prices, benchmark_prices=benchmark,
    )


def test_backtest_reports_cost_turnover_benchmark_and_walk_forward() -> None:
    report = PointInTimeBacktestEngine().run(
        point_in_time_dataset(), BacktestConfig(top_n=1, transaction_cost_bps=100, bootstrap_samples=100),
        WalkForwardConfig(training_windows=2, test_windows=1, step_windows=1),
    )
    assert len(report.periods) == 4
    assert report.periods[0].selected_tickers == ["AAA"]
    assert report.periods[0].net_return < report.periods[0].gross_return
    assert report.metrics["total_transaction_cost"] > 0
    assert report.metrics["benchmark_total_return"] is not None
    assert report.confidence_intervals["mean_period_excess_return_95"] is not None
    assert len(report.walk_forward) == 3


def test_backtest_refuses_current_or_non_point_in_time_universe() -> None:
    dataset = point_in_time_dataset()
    dataset.universe_snapshots[0].point_in_time_supported = False
    with pytest.raises(BacktestInputError, match="survivorship"):
        PointInTimeBacktestEngine().run(dataset, BacktestConfig())


def test_backtest_refuses_signal_that_was_not_available_at_decision_time() -> None:
    dataset = point_in_time_dataset()
    first = dataset.signals[0]
    dataset.signals[0] = first.model_copy(update={"available_at": first.available_at + timedelta(days=1)})
    with pytest.raises(BacktestInputError, match="became available"):
        PointInTimeBacktestEngine().run(dataset, BacktestConfig())


def test_backtest_refuses_unadjusted_prices_when_policy_requires_adjusted() -> None:
    dataset = point_in_time_dataset()
    dataset.prices[0].adjusted = False
    with pytest.raises(BacktestInputError, match="Adjusted-price"):
        PointInTimeBacktestEngine().run(dataset, BacktestConfig(require_adjusted_prices=True))


class _ForwardAnalysisFixture:
    def analyze_technical(self, ticker: str, horizon: str):
        del horizon
        return SimpleNamespace(
            market_data=conflict_market_data(ticker),
            technical_result=SimpleNamespace(score=7.5, policy_version="technical-demo-v1"),
        )

    def fetch_data(self, ticker: str, horizon: str, include_financials: bool = False):
        del horizon, include_financials
        return SimpleNamespace(market_data=conflict_market_data(ticker))


class _ForwardUniverseFixture:
    def fetch_current(self) -> InstrumentUniverseSnapshot:
        return InstrumentUniverseSnapshot(
            instruments=[Instrument(ticker="FPT"), Instrument(ticker="VNM")],
            source_reference=_reference(DataKind.INSTRUMENT_UNIVERSE, date.today()),
        )


def _workspace_test_root() -> Path:
    root = Path(__file__).resolve().parents[1] / ".runtime" / "pytest-forward-evaluation" / uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    return root


def test_forward_recorder_persists_real_time_contract_without_network() -> None:
    test_root = _workspace_test_root()
    capture, destination = ForwardEvaluationRecorder(
        _ForwardAnalysisFixture(), _ForwardUniverseFixture(), root=test_root,
    ).capture(["FPT", "INVALID"])

    assert destination.exists()
    assert capture.universe.point_in_time_supported is True
    assert [signal.ticker for signal in capture.signals] == ["FPT"]
    assert capture.prices[0].adjusted is True
    assert any("INVALID: not present" in failure for failure in capture.failures)


def test_forward_builder_uses_only_previously_captured_records() -> None:
    test_root = _workspace_test_root()
    decision_dates = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    for index, decision_date in enumerate(decision_dates):
        captured_at = datetime.combine(decision_date, datetime.min.time(), tzinfo=timezone.utc)
        capture = ForwardEvaluationCapture(
            captured_at=captured_at,
            universe=PointInTimeUniverseSnapshot(
                as_of=decision_date, tickers=["FPT"],
                source_reference=_reference(DataKind.INSTRUMENT_UNIVERSE, decision_date),
            ),
            signals=[SignalSnapshot(
                ticker="FPT", as_of=decision_date, available_at=captured_at, score=7.0,
                policy_version="technical-demo-v1", source_ids=["fixture:signal:FPT"],
            )],
            prices=[HistoricalPrice(
                ticker="FPT", session=decision_date, close=100 + index,
                adjusted=True, source_reference=_reference(DataKind.OHLCV, decision_date),
            )],
            benchmark_prices=[HistoricalPrice(
                ticker="VNINDEX", session=decision_date, close=100 + index,
                adjusted=True, source_reference=_reference(DataKind.OHLCV, decision_date),
            )],
        )
        (test_root / f"{index}.json").write_text(capture.model_dump_json(), encoding="utf-8")

    dataset = ForwardDatasetBuilder().build(test_root, "forward-fixture-v1")
    report = PointInTimeBacktestEngine().run(dataset, BacktestConfig(bootstrap_samples=100))

    assert dataset.metadata["capture_mode"] == "forward-only-v1"
    assert len(dataset.universe_snapshots) == 3
    assert len(report.periods) == 1
