"""Explicit real-data capture for a private local demo; no implicit public upload."""
from argparse import ArgumentParser
from pathlib import Path
import os

from stock_multi_agent.providers.market_provider import VnstockMarketDataProvider
from stock_multi_agent.config import load_workspace_env
from stock_multi_agent.services.demo_bundle import DemoBundleService


def main():
    parser = ArgumentParser(description='Capture full real OHLCV/facts/provenance into a checksum-verified local replay bundle.')
    parser.add_argument('tickers', nargs='+')
    parser.add_argument('--output', required=True, help='New .json filename, preferably .runtime/demo/demo-YYYY-MM-DD.json')
    parser.add_argument('--rights-note', required=True, help='Your permitted local usage and redistribution restrictions')
    parser.add_argument('--lookback-days', type=int, default=550)
    args = parser.parse_args()
    if not 1 <= len(args.tickers) <= 20 or not 90 <= args.lookback_days <= 4000:
        parser.error('Capture 1–20 symbols and 90–4000 calendar lookback days per explicit run')
    target = Path(args.output)
    if target.exists():
        parser.error('Output exists; select a new filename to preserve the earlier capture')
    if os.name == 'nt' and target.resolve().drive.upper() != 'D:':
        parser.error('Keep replay files on drive D')
    load_workspace_env()
    provider = VnstockMarketDataProvider(lookback_days=args.lookback_days)
    datasets = []
    failures = []
    for ticker in dict.fromkeys(symbol.strip().upper() for symbol in args.tickers):
        try:
            datasets.append(provider.get_market_data(ticker, include_financials=True))
        except Exception as exc:
            # No raw exception/URL/token printed by this capture tool.
            failures.append(f'{ticker}: {type(exc).__name__}')
    if not datasets:
        raise SystemExit('No source data captured. ' + '; '.join(failures))
    destination = DemoBundleService.capture(datasets, target, rights_note=args.rights_note)
    print(f'Private replay saved: {destination}')
    print(f'Tickers: {len(datasets)}; bars: {sum(len(item.bars) for item in datasets)}; failures: {len(failures)}')
    for failure in failures:
        print(f'- {failure}')


if __name__ == '__main__':
    main()
