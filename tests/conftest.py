from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from stock_multi_agent.models.analysis import FinancialMetrics, MarketData, PriceBar
from stock_multi_agent.models.data import DataKind, PriceAdjustment, SourceReference


def conflict_market_data(ticker: str = "FPT") -> MarketData:
    bars: list[PriceBar] = []
    value_date = date.today()
    while value_date.weekday() >= 5:
        value_date -= timedelta(days=1)
    business_dates: list[date] = []
    while len(business_dates) < 260:
        if value_date.weekday() < 5:
            business_dates.append(value_date)
        value_date -= timedelta(days=1)
    for index, value_date in enumerate(reversed(business_dates)):
        if index < 100:
            baseline = 100 + 0.5 * index
        elif index < 150:
            baseline = 150 - 1.2 * (index - 100)
        else:
            baseline = 90 + 0.38 * (index - 150)
        close = baseline + (5 if index % 2 else -5)
        open_price = close - (1.0 if index % 2 else -1.0)
        bars.append(PriceBar(
            date=value_date,
            open=open_price,
            high=max(open_price, close) + 1,
            low=min(open_price, close) - 1,
            close=close,
            volume=1_000_000 + index * 1_000 + (300_000 if index >= 255 else 0),
        ))
    fetched_at = datetime.combine(bars[-1].date, datetime.min.time(), tzinfo=timezone.utc)
    return MarketData(
        ticker=ticker,
        source="test-fixture",
        bars=bars,
        fetched_at=fetched_at,
        as_of=bars[-1].date,
        price_adjustment=PriceAdjustment.ADJUSTED,
        source_refs=[SourceReference(
            provider="test",
            source_id=f"test:ohlcv:{ticker}",
            data_kind=DataKind.OHLCV,
            fetched_at=fetched_at,
            as_of=bars[-1].date,
            schema_version="test-v1",
        )],
        financials=FinancialMetrics(
            revenue_growth=0.12,
            profit_growth=-0.02,
            eps=5.2,
            pe=28.0,
            pb=5.5,
            roe=0.20,
            debt_to_equity=1.0,
            operating_cash_flow=1_000_000,
            profit_margin=0.10,
        ),
    )


class InMemoryProvider:
    def __init__(self, data: MarketData) -> None:
        self.data = data

    def get_market_data(self, ticker: str, include_financials: bool = True) -> MarketData:
        del include_financials
        assert ticker == self.data.ticker
        return self.data


@pytest.fixture
def market_data() -> MarketData:
    return conflict_market_data()
