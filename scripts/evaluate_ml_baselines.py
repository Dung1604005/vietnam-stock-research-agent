from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stock_multi_agent.services.ml_baselines import evaluate_baselines
from stock_multi_agent.services.ml_dataset_v3 import sha256_file
from stock_multi_agent.services.ml_features_v3 import window_endpoints
from stock_multi_agent.services.ml_validation import expanding_purged_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate zero/persistence/mean/median/Ridge/optional tree on identical V3 endpoints")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--no-tree", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new report path to retain the previous experiment")
    frame = pd.read_csv(args.data, parse_dates=["date", "target_end_1", "target_end_5"])
    folds = expanding_purged_folds(frame, window_endpoints(frame), folds=args.folds)
    report = evaluate_baselines(frame, folds, include_tree=not args.no_tree)
    report["dataset_sha256"] = sha256_file(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(report['reports'])} fold/horizon results: {args.output}")


if __name__ == "__main__":
    main()
