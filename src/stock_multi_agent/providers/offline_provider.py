"""Verified, exact-history replay provider. Missing tickers never trigger network."""
from stock_multi_agent.models.demo_bundle import DemoBundle
from stock_multi_agent.providers.market_provider import MarketDataError
from stock_multi_agent.services.demo_bundle import DemoBundleService


class OfflineMarketDataProvider:
    def __init__(self, bundle: DemoBundle | str):
        self.bundle = DemoBundle.model_validate(bundle.model_dump(mode='json')) if isinstance(bundle, DemoBundle) else DemoBundleService.load(bundle)
        self.source = 'SNAPSHOT_REPLAY'
        self.origin = 'SNAPSHOT_REPLAY'
        self.cache_hits = 0
        self.cache_misses = 0

    def get_market_data(self, ticker: str, include_financials: bool = True, **kwargs):
        symbol = ticker.strip().upper()
        if symbol not in self.bundle.datasets:
            raise MarketDataError(f'Offline replay has no verified data for {symbol}; no network fallback is allowed')
        data = self.bundle.datasets[symbol].model_copy(deep=True)
        data.access_mode = 'SNAPSHOT_REPLAY'
        note = f'SNAPSHOT_REPLAY: original data as of {data.as_of}; bundle captured {self.bundle.captured_at.isoformat()}'
        data.warnings = list(dict.fromkeys([*data.warnings, note]))
        return data

    def refresh_market_data(self, ticker: str, include_financials: bool = True):
        return self.get_market_data(ticker, include_financials)
