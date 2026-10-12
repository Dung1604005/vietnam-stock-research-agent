from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_multi_agent.config import load_workspace_env
from stock_multi_agent.providers.market_provider import MarketDataError, VnstockMarketDataProvider


ROOT = Path(__file__).resolve().parents[1]


def configured_universe() -> list[str]:
    with (ROOT / "config" / "watchlist.json").open(encoding="utf-8") as handle:
        groups = json.load(handle)
    return list(dict.fromkeys(ticker for tickers in groups.values() for ticker in tickers))


def main() -> None:
    load_workspace_env(ROOT)
    parser = argparse.ArgumentParser(description="Fetch real data and report provenance/freshness/quality.")
    parser.add_argument("tickers", nargs="*", help="Ticker list; defaults to the configured watchlist universe")
    parser.add_argument("--prices-only", action="store_true")
    parser.add_argument("--show-health", action="store_true", help="Print in-process cache/resilience metrics after the report")
    args = parser.parse_args()

    provider = VnstockMarketDataProvider()
    tickers = [ticker.upper() for ticker in args.tickers] or configured_universe()
    failures = 0
    print("ticker\tstatus\tas_of\tfetched_at\tsources\tissues")
    for ticker in tickers:
        try:
            data = provider.get_market_data(ticker, include_financials=not args.prices_only)
            issues = "; ".join(f"{item.code}:{item.severity.value}" for item in data.quality_report.issues) or "none"
            sources = ",".join(item.source_id for item in data.source_refs)
            print(
                f"{ticker}\t{data.quality_report.status.value}\t{data.as_of}\t"
                f"{data.fetched_at.isoformat()}\t{sources}\t{issues}"
            )
            failures += int(data.quality_report.status.value == "FAIL")
        except MarketDataError as exc:
            failures += 1
            print(f"{ticker}\tFAIL\t-\t-\t-\t{exc}")
    if args.show_health:
        print("health\t" + json.dumps(provider.health(), ensure_ascii=False, sort_keys=True))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
