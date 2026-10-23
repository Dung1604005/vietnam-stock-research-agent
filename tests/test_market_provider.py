from __future__ import annotations

import pandas as pd
import pytest
from datetime import date
from types import SimpleNamespace

from stock_multi_agent.providers.market_provider import MarketDataError, VnstockMarketDataProvider


def test_vci_uses_kbs_as_default_price_fallback(monkeypatch) -> None:
    monkeypatch.delenv("VNSTOCK_PRICE_FALLBACK_SOURCES", raising=False)
    provider = VnstockMarketDataProvider(source="VCI", enable_cache=False)

    assert provider._price_sources() == ("VCI", "KBS")


def test_price_fallback_setting_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("VNSTOCK_PRICE_FALLBACK_SOURCES", "")
    provider = VnstockMarketDataProvider(source="VCI", enable_cache=False)

    assert provider._price_sources() == ("VCI",)


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


def test_vci_history_keeps_actual_quarters_and_selects_latest_instead_of_head_four():
    calls = []
    class Explorer:
        def _get_report(self, report_type, mode, period, limit, **kwargs):
            calls.append((report_type, mode, period, limit))
            # VCI returns ascending history; the public SDK used head(4) then
            # collapsed all four 2018 quarters to the same annual label.
            return pd.DataFrame([
                {'year': 2018, 'quarter': q, 'pe': 10 + q, 'roe': .15 + .01 * q}
                for q in range(1, 5)] + [
                {'year': 2025, 'quarter': 4, 'pe': 16, 'roe': .22},
                {'year': 2026, 'quarter': 2, 'pe': 17, 'roe': .23}])
    provider = VnstockMarketDataProvider(enable_cache=False)
    frame = provider._financial_report_frame(SimpleNamespace(provider=Explorer()), 'ratio', 'quarter')
    metrics, observations = provider._normalize_ratios_with_observations(frame, 'live:ratio')
    assert calls == [('ratio', 'raw', 'quarter', 512)]
    assert metrics.pe == 17 and metrics.roe == pytest.approx(.23)  # No erroneous /100 from display labels.
    assert max(item.period_end for item in observations) == date(2026, 6, 30)
    assert all(item.frequency == 'quarter' for item in observations)
    assert len({item.period_end for item in observations}) == 6


def test_financial_history_compatibility_fallback_preserves_missing_values():
    calls = []
    def ratio(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame([{'year': 2025, 'pe_ratio': 12, 'roe': None}])
    provider = VnstockMarketDataProvider(enable_cache=False)
    frame = provider._financial_report_frame(SimpleNamespace(ratio=ratio), 'ratio', 'quarter')
    assert calls[0]['dropna'] is False
    assert provider._normalize_ratios(frame).roe is None
