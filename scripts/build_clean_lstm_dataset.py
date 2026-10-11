from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / ".runtime" / "ml" / "vci_ohlcv_lstm_1m.csv"
DEFAULT_OUTPUT = ROOT / ".runtime" / "ml" / "vci_ohlcv_lstm_clean_v1.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cleaner, immutable LSTM dataset without overwriting the 1M source.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-cutoff", default="2024-11-27")
    parser.add_argument("--minimum-history-rows", type=int, default=400)
    parser.add_argument("--minimum-active-ratio", type=float, default=0.90)
    parser.add_argument("--minimum-liquidity-percentile", type=float, default=0.50)
    parser.add_argument("--maximum-calendar-gap-days", type=int, default=10)
    parser.add_argument("--maximum-absolute-return", type=float, default=0.20)
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    if not source.exists():
        parser.error(f"Input dataset does not exist: {source}")
    if source == output:
        parser.error("Output must be a new file; the source dataset is immutable.")
    if not 0 <= args.minimum_active_ratio <= 1 or not 0 <= args.minimum_liquidity_percentile <= 1:
        parser.error("Ratio and percentile thresholds must be between 0 and 1.")

    data = pd.read_csv(source, parse_dates=["date", "target_date"])
    required = {
        "ticker", "date", "open", "high", "low", "close", "volume",
        "target_date", "next_session_close", "next_session_return",
        "source", "source_id", "fetched_at", "price_adjustment",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Source dataset is missing columns: {sorted(missing)}")
    if data.duplicated(["ticker", "date"]).any():
        raise ValueError("Source dataset contains duplicate ticker/date pairs.")

    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)
    data["current_return"] = data.groupby("ticker")["close"].pct_change(fill_method=None)
    data["calendar_gap_days"] = (data["target_date"] - data["date"]).dt.days
    data["traded_value"] = data["close"] * data["volume"]

    cutoff = pd.Timestamp(args.selection_cutoff)
    selection_data = data[data["target_date"] <= cutoff]
    ticker_metrics = selection_data.groupby("ticker").agg(
        selection_history_rows=("date", "size"),
        selection_active_ratio=("volume", lambda values: float(values.gt(0).mean())),
        selection_median_traded_value=("traded_value", "median"),
    )
    ticker_metrics["selection_liquidity_percentile"] = ticker_metrics[
        "selection_median_traded_value"
    ].rank(pct=True)
    eligible = ticker_metrics[
        ticker_metrics["selection_history_rows"].ge(args.minimum_history_rows)
        & ticker_metrics["selection_active_ratio"].ge(args.minimum_active_ratio)
        & ticker_metrics["selection_liquidity_percentile"].ge(args.minimum_liquidity_percentile)
    ].copy()

    counts = {"source_rows": int(len(data)), "source_tickers": int(data["ticker"].nunique())}
    clean = data[data["ticker"].isin(eligible.index)].copy()
    counts["after_point_in_time_universe"] = int(len(clean))
    clean = clean[
        clean["calendar_gap_days"].between(1, args.maximum_calendar_gap_days)
    ].copy()
    counts["after_calendar_gap_rule"] = int(len(clean))
    clean = clean[
        clean["current_return"].abs().le(args.maximum_absolute_return)
        & clean["next_session_return"].abs().le(args.maximum_absolute_return)
    ].copy()
    counts["after_unadjusted_outlier_rule"] = int(len(clean))
    clean = clean[
        clean["open"].gt(0)
        & clean["high"].ge(clean[["open", "close", "low"]].max(axis=1))
        & clean["low"].le(clean[["open", "close", "high"]].min(axis=1))
        & clean["volume"].ge(0)
        & np.isfinite(clean[["open", "high", "low", "close", "volume", "next_session_return"]]).all(axis=1)
    ].copy()
    counts["output_rows"] = int(len(clean))
    counts["output_tickers"] = int(clean["ticker"].nunique())

    clean = clean.merge(
        eligible.reset_index(), on="ticker", how="left", validate="many_to_one",
    )
    clean["cleaning_version"] = "lstm-clean-v1"
    clean = clean.sort_values(["ticker", "date"]).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    clean.to_csv(temporary, index=False, date_format="%Y-%m-%d")
    temporary.replace(output)

    metadata = {
        "schema_version": "lstm-clean-dataset-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source),
        "source_sha256": sha256(source),
        "output_path": str(output),
        "output_sha256": sha256(output),
        "counts": counts,
        "rules": {
            "selection_cutoff": args.selection_cutoff,
            "minimum_history_rows_at_cutoff": args.minimum_history_rows,
            "minimum_active_ratio_at_cutoff": args.minimum_active_ratio,
            "minimum_liquidity_percentile_at_cutoff": args.minimum_liquidity_percentile,
            "calendar_gap_days": [1, args.maximum_calendar_gap_days],
            "maximum_absolute_unadjusted_return": args.maximum_absolute_return,
            "valid_ohlcv": True,
        },
        "diagnostics": {
            "zero_volume_ratio": float(clean["volume"].eq(0).mean()),
            "unchanged_next_return_ratio": float(clean["next_session_return"].eq(0).mean()),
            "date_min": clean["date"].min().date().isoformat(),
            "date_max": clean["date"].max().date().isoformat(),
        },
        "warnings": [
            "Price-adjustment status remains UNKNOWN; filtering cannot replace an adjusted-price source.",
            "The source universe is a current-listing catalogue and retains survivorship bias.",
            "The calendar-gap rule is a heuristic because an official exchange calendar is unavailable.",
            "This dataset is for an educational research demo, not a performance claim.",
        ],
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
