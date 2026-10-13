from __future__ import annotations

from stock_multi_agent.models.analysis import MarketData
from stock_multi_agent.providers.market_provider import MarketDataProvider


class MarketDataTool:
    """A capability wrapper: acquisition only, never recommendation policy."""

    def __init__(self, provider: MarketDataProvider) -> None:
        self.provider = provider

    def get_stock_data(
        self,
        ticker: str,
        include_financials: bool = True,
        force_refresh: bool = False,
    ) -> MarketData:
        refresh = getattr(self.provider, "refresh_market_data", None)
        if force_refresh and callable(refresh):
            data = refresh(ticker, include_financials=include_financials)
        else:
            data = self.provider.get_market_data(ticker, include_financials=include_financials)
        if data.ticker != ticker.strip().upper():
            raise ValueError("Provider returned data for a different ticker")
        return data
