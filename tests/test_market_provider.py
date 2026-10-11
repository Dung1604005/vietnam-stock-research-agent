from __future__ import annotations

import pandas as pd
import pytest

from stock_multi_agent.providers.market_provider import MarketDataError, VnstockMarketDataProvider


def test_normalizes_vnstock_4_wide_ratio_schema() -> None:
    frame = pd.DataFrame(
        [
            ["P/E", "P/E", "pe_ratio", 20.0, 18.0],
            ["P/B", "P/B", "pb_ratio", 4.0, 3.5],
            ["ROE", "ROE (%)", "roe", 16.0, 20.0],
            ["ROA", "ROA (%)", "roa", 8.0, 10.0],
            ["Nợ/VCSH", "Debt to Equity", "debt_to_equity", 0.8, 0.7],
            ["Biên LN", "After-tax Profit Margin (%)", "net_margin", 12.0, 15.0],
        ],
        columns=["item", "item_en", "item_id", "2024", "2025"],
    )
    metrics = VnstockMarketDataProvider._normalize_ratios(frame)
    assert metrics.pe == pytest.approx(18.0)
    assert metrics.pb == pytest.approx(3.5)
    assert metrics.roe == pytest.approx(0.20)
    assert metrics.roa == pytest.approx(0.10)
    assert metrics.debt_to_equity == pytest.approx(0.7)
    assert metrics.profit_margin == pytest.approx(0.15)


def test_normalizes_quote_time_column() -> None:
    frame = pd.DataFrame([
        {"time": "2026-10-08", "open": 10, "high": 12, "low": 9, "close": 11, "volume": 1000},
        {"time": "2026-10-09", "open": 11, "high": 13, "low": 10, "close": 12, "volume": 1200},
    ])
    bars = VnstockMarketDataProvider._normalize_bars(frame)
    assert len(bars) == 2
    assert bars[-1].close == 12


def test_empty_quote_is_rejected() -> None:
    with pytest.raises(MarketDataError):
        VnstockMarketDataProvider._normalize_bars(pd.DataFrame())
