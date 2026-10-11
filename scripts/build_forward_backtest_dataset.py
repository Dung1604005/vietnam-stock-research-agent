from __future__ import annotations

import argparse
from pathlib import Path

from stock_multi_agent.services.forward_evaluation import ForwardDatasetBuilder


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a versioned point-in-time dataset from real forward captures.")
    parser.add_argument("--captures", type=Path, default=ROOT / ".runtime" / "forward_evaluation" / "captures")
    parser.add_argument("--version", required=True, help="Immutable label, e.g. demo-forward-2026q4-v1")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime" / "forward_evaluation" / "datasets")
    args = parser.parse_args()
    dataset = ForwardDatasetBuilder().build(args.captures, args.version)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"{args.version}.json"
    destination.write_text(dataset.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"Dataset written: {destination}")


if __name__ == "__main__":
    main()
