from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from stock_multi_agent.nlu.evaluation import (
    assert_no_query_overlap,
    evaluate_dataset,
    load_dataset,
    write_report,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "tests" / "data" / "semantic_queries.json"
DEFAULT_REPORT_DIR = ROOT / ".runtime" / "evaluations"


def _print_report(report: dict, show_errors: bool) -> None:
    print(f"Semantic Evaluation · {report['dataset']} · {report['version']}")
    labels = {
        "intent": "Intent Accuracy",
        "ticker": "Ticker Accuracy",
        "period": "Period Accuracy",
        "follow_up": "Follow-up Accuracy",
        "clarification": "Clarification Accuracy",
        "multi_intent": "Multi-intent Accuracy",
    }
    for name, label in labels.items():
        if name in report["metrics"]:
            count = report["counts"][name]
            print(f"{label:<28} {report['metrics'][name]:>6.1%}  ({count['passed']}/{count['total']})")
    calibration = report["calibration"]
    print(f"Intent calibration ECE       {calibration['expected_calibration_error']}")
    if show_errors:
        print(f"\nErrors ({len(report['errors'])}):")
        for item in report["errors"]:
            print(
                f"- {item['query']}\n  expected={item['expected']} actual={item['actual']} "
                f"confidence={item['confidence']} failed={','.join(item['failed'])}"
            )


def main() -> None:
    argument_parser = argparse.ArgumentParser(description="Evaluate semantic parsing without market-data calls.")
    argument_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Regression dataset")
    argument_parser.add_argument("--holdout", type=Path, help="Separately labelled holdout dataset; overlap is rejected")
    argument_parser.add_argument("--show-errors", action="store_true")
    argument_parser.add_argument("--write-report", action="store_true", help="Write JSON report below .runtime/evaluations")
    args = argument_parser.parse_args()

    regression_cases, regression_version = load_dataset(args.dataset)
    reports = [evaluate_dataset(regression_cases, args.dataset.name, regression_version)]
    if args.holdout:
        holdout_cases, holdout_version = load_dataset(args.holdout)
        if not holdout_cases:
            raise SystemExit("Holdout dataset is empty; it must contain independently labelled cases.")
        assert_no_query_overlap(regression_cases, holdout_cases)
        reports.append(evaluate_dataset(holdout_cases, args.holdout.name, holdout_version))

    for report in reports:
        _print_report(report, args.show_errors)
    if args.write_report:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = write_report({"generated_at": datetime.now(timezone.utc).isoformat(), "reports": reports}, DEFAULT_REPORT_DIR / f"semantic-{timestamp}.json")
        print(f"Report written: {destination}")


if __name__ == "__main__":
    main()
