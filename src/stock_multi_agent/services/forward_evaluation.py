from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from stock_multi_agent.models.data import DataKind, PriceAdjustment
from stock_multi_agent.models.evaluation import (
    BacktestDataset,
    ForwardEvaluationCapture,
    HistoricalPrice,
    PointInTimeUniverseSnapshot,
    SignalSnapshot,
)
from stock_multi_agent.providers.universe_provider import InstrumentUniverseProvider
from stock_multi_agent.services.analysis_service import AnalysisService


class ForwardEvaluationRecorder:
    """Capture real observations today for a future-only, bounded demo evaluation."""

    def __init__(self, analysis: AnalysisService, universe_provider: InstrumentUniverseProvider, root: Path | None = None) -> None:
        self.analysis = analysis
        self.universe_provider = universe_provider
        self.root = (root or Path(__file__).resolve().parents[3] / ".runtime" / "forward_evaluation" / "captures").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def capture(self, tickers: list[str], horizon: str = "3 months") -> tuple[ForwardEvaluationCapture, Path]:
        selected = list(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker.strip()))
        if not selected:
            raise ValueError("At least one ticker is required for a bounded forward evaluation capture.")
        snapshot = self.universe_provider.fetch_current()
        captured_at = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
        current_tickers = {item.ticker for item in snapshot.instruments}
        universe = PointInTimeUniverseSnapshot(
            as_of=captured_at.date(), tickers=[item.ticker for item in snapshot.instruments],
            source_reference=snapshot.source_reference,
            # This is valid only from this observed capture forward. It is not
            # a reconstructed historical membership dataset.
            point_in_time_supported=True,
        )
        capture = ForwardEvaluationCapture(captured_at=captured_at, universe=universe)
        for ticker in selected:
            if ticker not in current_tickers:
                capture.failures.append(f"{ticker}: not present in the captured active listing universe")
                continue
            try:
                state = self.analysis.analyze_technical(ticker, horizon)
                if state.market_data is None or state.technical_result is None:
                    raise ValueError("technical analysis returned no structured result")
                data = state.market_data
                ohlcv_ref = next((ref for ref in data.source_refs if ref.data_kind is DataKind.OHLCV), None)
                if ohlcv_ref is None:
                    raise ValueError("OHLCV provenance is missing")
                latest = data.bars[-1]
                capture.signals.append(SignalSnapshot(
                    ticker=ticker, as_of=captured_at.date(), available_at=captured_at,
                    score=state.technical_result.score, policy_version=state.technical_result.policy_version,
                    source_ids=[ref.source_id for ref in data.source_refs],
                ))
                capture.prices.append(HistoricalPrice(
                    ticker=ticker, session=latest.date, close=latest.close,
                    adjusted=data.price_adjustment is PriceAdjustment.ADJUSTED, source_reference=ohlcv_ref,
                ))
                if data.price_adjustment is not PriceAdjustment.ADJUSTED:
                    capture.failures.append(f"{ticker}: recorded as unadjusted/unknown; strict adjusted-price backtest will refuse it")
            except Exception as exc:
                capture.failures.append(f"{ticker}: {type(exc).__name__}: {exc}")
        try:
            benchmark_state = self.analysis.fetch_data("VNINDEX", horizon, include_financials=False)
            data = benchmark_state.market_data
            if data is None:
                raise ValueError("VNINDEX price is unavailable")
            ohlcv_ref = next((ref for ref in data.source_refs if ref.data_kind is DataKind.OHLCV), None)
            if ohlcv_ref is None:
                raise ValueError("VNINDEX OHLCV provenance is missing")
            latest = data.bars[-1]
            capture.benchmark_prices.append(HistoricalPrice(
                ticker="VNINDEX", session=latest.date, close=latest.close,
                adjusted=data.price_adjustment is PriceAdjustment.ADJUSTED, source_reference=ohlcv_ref,
            ))
            if data.price_adjustment is not PriceAdjustment.ADJUSTED:
                capture.failures.append("VNINDEX: recorded as unadjusted/unknown; strict adjusted-price backtest will refuse it")
        except Exception as exc:
            capture.failures.append(f"VNINDEX: {type(exc).__name__}: {exc}")
        destination = self.root / f"{captured_at.strftime('%Y%m%dT%H%M%S%z')}.json"
        destination.write_text(capture.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return capture, destination


class ForwardDatasetBuilder:
    """Compile only separately captured observations; never refetch/rewrite past sessions."""

    def build(self, capture_directory: Path, dataset_version: str) -> BacktestDataset:
        captures = self._load(capture_directory)
        if len(captures) < 2:
            raise ValueError("At least two captures on different dates are required for a forward dataset.")
        by_date: dict = {}
        for capture in captures:
            existing = by_date.get(capture.universe.as_of)
            if existing is None or capture.captured_at < existing.captured_at:
                by_date[capture.universe.as_of] = capture
        selected = [by_date[key] for key in sorted(by_date)]
        if len(selected) < 2:
            raise ValueError("Captures must span at least two distinct decision dates.")
        prices = self._first_prices(selected, "prices")
        benchmark_prices = self._first_prices(selected, "benchmark_prices")
        signals = [signal for capture in selected for signal in capture.signals]
        if not signals:
            raise ValueError("No verified technical signals were captured.")
        return BacktestDataset(
            dataset_version=dataset_version,
            universe_snapshots=[capture.universe for capture in selected],
            signals=signals, prices=prices, benchmark_prices=benchmark_prices,
            metadata={
                "capture_mode": "forward-only-v1",
                "capture_count": len(selected),
                "failures": [failure for capture in selected for failure in capture.failures],
            },
        )

    @staticmethod
    def _load(capture_directory: Path) -> list[ForwardEvaluationCapture]:
        captures = [ForwardEvaluationCapture.model_validate_json(path.read_text(encoding="utf-8")) for path in sorted(capture_directory.glob("*.json"))]
        return sorted(captures, key=lambda item: item.captured_at)

    @staticmethod
    def _first_prices(captures: list[ForwardEvaluationCapture], attribute: str) -> list[HistoricalPrice]:
        first: dict[tuple[str, object], HistoricalPrice] = {}
        for capture in captures:
            for price in getattr(capture, attribute):
                first.setdefault((price.ticker, price.session), price)
        return list(first.values())
