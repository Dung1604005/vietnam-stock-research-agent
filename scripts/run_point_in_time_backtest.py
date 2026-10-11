from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_multi_agent.models.evaluation import BacktestConfig, BacktestDataset, WalkForwardConfig
from stock_multi_agent.services.backtesting import BacktestInputError, PointInTimeBacktestEngine
from stock_multi_agent.services.evaluation_registry import EvaluationRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline, point-in-time validated deterministic backtest.")
    parser.add_argument("dataset", type=Path, help="Versioned JSON BacktestDataset manifest; never a live provider response")
    parser.add_argument("--config", type=Path, help="Optional JSON BacktestConfig")
    parser.add_argument("--walk-forward", nargs=3, type=int, metavar=("TRAIN", "TEST", "STEP"))
    parser.add_argument("--write-report", action="store_true", help="Write an immutable report under .runtime/evaluations")
    args = parser.parse_args()
    dataset = BacktestDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
    config = BacktestConfig.model_validate_json(args.config.read_text(encoding="utf-8")) if args.config else BacktestConfig()
    walk_forward = WalkForwardConfig(
        training_windows=args.walk_forward[0], test_windows=args.walk_forward[1], step_windows=args.walk_forward[2],
    ) if args.walk_forward else None
    try:
        report = PointInTimeBacktestEngine().run(dataset, config, walk_forward)
    except BacktestInputError as exc:
        raise SystemExit(f"Backtest refused: {exc}") from exc
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if args.write_report:
        destination = EvaluationRegistry().write(
            "point-in-time-backtest", report.model_dump(mode="json"),
            {"dataset": dataset.dataset_version}, config.model_dump(mode="json"),
        )
        print(f"Evaluation report written: {destination}")


if __name__ == "__main__":
    main()
