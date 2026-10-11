from __future__ import annotations

import importlib.util
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from stock_multi_agent.models.analysis import MarketData, PriceBar
from stock_multi_agent.models.data import DataKind, SourceReference


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_lstm_ohlcv_dataset.py"
SPEC = importlib.util.spec_from_file_location("collect_lstm_ohlcv_dataset", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_real_bar_rows_have_next_observed_session_label() -> None:
    data = MarketData(
        ticker="FPT",
        source="vnstock:VCI",
        bars=[
            PriceBar(date=date(2026, 1, 2), open=100, high=102, low=99, close=100, volume=1000),
            PriceBar(date=date(2026, 1, 5), open=101, high=104, low=100, close=102, volume=1200),
        ],
        fetched_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
        source_refs=[SourceReference(
            provider="vnstock", source_id="VCI:Quote.history:FPT", data_kind=DataKind.OHLCV,
            fetched_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
        )],
    )

    rows = MODULE.rows_for_market_data(data)

    assert len(rows) == 1
    assert rows[0]["date"] == "2026-01-02"
    assert rows[0]["target_date"] == "2026-01-05"
    assert rows[0]["next_session_close"] == 102
    assert rows[0]["next_session_return"] == pytest.approx(0.02)
    assert rows[0]["source_id"] == "VCI:Quote.history:FPT"
