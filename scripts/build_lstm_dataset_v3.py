"""Build NEW V3 data locally, without network access or touching V2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stock_multi_agent.services.ml_dataset_v3 import build_dataset, write_dataset, refresh_runtime_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(".runtime/ml/vci_ohlcv_lstm_1m.csv"))
    parser.add_argument("--output", type=Path, default=Path(".runtime/ml/v3/dataset-1"))
    parser.add_argument("--limit-rows", type=int, help="Pilot only; report is labeled as truncated source")
    parser.add_argument("--refresh-runtime", action="store_true", help="Preserve existing CSVs/old ZIP; explicitly package the latest shared code in a new ZIP")
    args = parser.parse_args()
    if args.refresh_runtime:
        manifest = refresh_runtime_bundle(args.output)
        print(json.dumps({"output": str(args.output.resolve()), "runtime_bundle": manifest["runtime_bundle"], "csv_unchanged": True}, indent=2))
        return
    raw = pd.read_csv(args.input, nrows=args.limit_rows)
    result = build_dataset(raw)
    result.manifest["pilot_row_limit"] = args.limit_rows
    manifest = write_dataset(result, args.output, source_path=args.input)
    protocol = Path(__file__).resolve().parents[1] / "evaluation/configs/lstm-v3-protocol.json"
    (args.output / "protocol.json").write_text(protocol.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "retained_rows": manifest["retained_rows"], "endpoints": manifest["training_endpoints"], "quarantine": manifest["quarantined_rows"], "reconciled": manifest["reconciled_counts"], "validation_status": manifest["validation_status"]}, indent=2))


if __name__ == "__main__":
    main()
