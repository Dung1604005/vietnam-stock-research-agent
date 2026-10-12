from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from stock_multi_agent.config import load_workspace_env
from stock_multi_agent.models.analysis import MarketData
from stock_multi_agent.providers.market_provider import MarketDataError, VnstockMarketDataProvider
from stock_multi_agent.providers.universe_provider import VnstockInstrumentUniverseProvider


ROOT = Path(__file__).resolve().parents[1]
FIELDNAMES = [
    "ticker", "date", "open", "high", "low", "close", "volume",
    "target_date", "next_session_close", "next_session_return",
    "source", "source_id", "fetched_at", "price_adjustment",
]


def rows_for_market_data(data: MarketData) -> list[dict[str, str | float]]:
    """Make real, next-observed-session labels without inventing a calendar."""

    reference = next((item for item in data.source_refs if item.data_kind.value == "OHLCV"), None)
    source_id = reference.source_id if reference else "unknown"
    rows: list[dict[str, str | float]] = []
    for current, following in zip(data.bars, data.bars[1:]):
        rows.append({
            "ticker": data.ticker,
            "date": current.date.isoformat(),
            "open": current.open,
            "high": current.high,
            "low": current.low,
            "close": current.close,
            "volume": current.volume,
            "target_date": following.date.isoformat(),
            "next_session_close": following.close,
            "next_session_return": following.close / current.close - 1,
            "source": data.source,
            "source_id": source_id,
            "fetched_at": data.fetched_at.isoformat(),
            "price_adjustment": data.price_adjustment.value,
        })
    return rows


def completed_tickers(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["ticker"] for row in csv.DictReader(handle) if row.get("ticker")}


def normalize_existing_output(path: Path) -> int:
    """Repair an interrupted/concurrent append without synthesising any rows."""

    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"Existing dataset has an incompatible schema: {reader.fieldnames}")
        unique = {(row.get("ticker"), row.get("date")): row for row in reader if row.get("ticker") and row.get("date")}
    temporary = path.with_suffix(".deduplicated.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(unique.values())
    temporary.replace(path)
    return len(unique)


def current_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str | float]]) -> None:
    if not rows:
        return
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def write_metadata(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def read_previous_failures(path: Path) -> list[dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        failures = payload.get("failures", [])
        return [item for item in failures if isinstance(item, dict) and item.get("ticker") and item.get("error")]
    except (OSError, ValueError, TypeError):
        return []


def main() -> None:
    load_workspace_env(ROOT)
    parser = argparse.ArgumentParser(
        description="Collect a real, resumable OHLCV + next-session-label dataset for offline LSTM experiments."
    )
    parser.add_argument("--target-rows", type=int, default=100_000, help="Minimum number of labelled rows to collect.")
    parser.add_argument("--lookback-days", type=int, default=1460, help="Calendar days of history requested for each ticker.")
    parser.add_argument("--max-tickers", type=int, help="Optional cap for this run; use it for short resumable collection batches.")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime" / "ml" / "vci_ohlcv_lstm_100k.csv")
    parser.add_argument("--tickers", nargs="*", help="Optional explicit ticker list. Defaults to the current VCI listing catalogue.")
    args = parser.parse_args()
    if args.target_rows < 1 or args.lookback_days < 60:
        parser.error("--target-rows must be positive and --lookback-days must be at least 60")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(".metadata.json")
    deduplicated_rows = normalize_existing_output(output)
    existing = completed_tickers(output)
    rows_written = current_row_count(output)
    failures = read_previous_failures(metadata_path)
    failed_tickers = {item["ticker"] for item in failures}

    universe_source: dict[str, object]
    if args.tickers:
        tickers = list(dict.fromkeys(ticker.strip().upper() for ticker in args.tickers if ticker.strip()))
        universe_source = {"kind": "explicit", "tickers": tickers}
    else:
        snapshot = VnstockInstrumentUniverseProvider().fetch_current()
        tickers = [instrument.ticker for instrument in snapshot.instruments]
        universe_source = {
            "kind": "current_vci_listing_catalogue",
            "source_id": snapshot.source_reference.source_id,
            "source_url": str(snapshot.source_reference.source_url),
            "captured_at": snapshot.source_reference.fetched_at.isoformat(),
            "point_in_time_supported": False,
            "selection": "ticker_ascending",
        }

    provider = VnstockMarketDataProvider(lookback_days=args.lookback_days)
    processed_this_run = 0
    print(
        f"Resuming at {rows_written:,}/{args.target_rows:,} rows; {len(existing):,} completed tickers "
        f"({deduplicated_rows:,} unique rows after integrity check)."
    )
    for ticker in tickers:
        if rows_written >= args.target_rows:
            break
        if ticker in existing or ticker in failed_tickers:
            continue
        if args.max_tickers is not None and processed_this_run >= args.max_tickers:
            break
        try:
            data = provider.get_market_data(ticker, include_financials=False)
            rows = rows_for_market_data(data)
            remaining = args.target_rows - rows_written
            rows = rows[:remaining]
            write_rows(output, rows)
            rows_written += len(rows)
            existing.add(ticker)
            processed_this_run += 1
            print(f"{ticker}: +{len(rows):,} rows ({rows_written:,}/{args.target_rows:,})")
        except MarketDataError as exc:
            processed_this_run += 1
            failures.append({"ticker": ticker, "error": str(exc)})
            failed_tickers.add(ticker)
            print(f"{ticker}: skipped ({exc})")

    checksum = hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None
    metadata = {
        "schema_version": "lstm-ohlcv-next-session-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": rows_written,
        "target_rows": args.target_rows,
        "ticker_count": len(completed_tickers(output)),
        "columns": FIELDNAMES,
        "lookback_days_requested": args.lookback_days,
        "universe": universe_source,
        "source": "vnstock:VCI; one OHLCV request per ticker",
        "adjustment_policy": "UNKNOWN; do not use for a performance/backtest claim without an adjusted-price/corporate-action source.",
        "labelling": "next observed price-bar session, not an assumed calendar day",
        "resume": "rerun the same command; already completed tickers are skipped",
        "failures": failures,
        "sha256": checksum,
    }
    write_metadata(metadata_path, metadata)
    print(f"Dataset: {output}")
    print(f"Metadata: {metadata_path}")
    print(f"Finished with {rows_written:,} real labelled rows; {len(failures):,} ticker failures.")
    if rows_written < args.target_rows:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
