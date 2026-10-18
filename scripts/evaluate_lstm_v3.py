"""Attach independently captured, now-mature forward predictions to V3 reports.

CSV must contain ticker,horizon,forecast_origin,target_end,captured_at,
label_available_at,actual,prediction,majority_direction. Predictions must have
been captured before the target date, using the frozen artifact identity.
This checks timestamps/identity but is not a cryptographic attestation of capture.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_multi_agent.services.ml_dataset_v3 import sha256_file
from stock_multi_agent.services.ml_model_registry import evaluate_model_gate, verify_artifact_directory
from stock_multi_agent.services.ml_validation import paired_date_block_bootstrap, regression_metrics


def evaluate_forward_rows(frame: pd.DataFrame, protocol: dict, *, artifact_identity: str) -> dict:
    required = {"ticker", "horizon", "forecast_origin", "target_end", "captured_at", "label_available_at", "actual", "prediction", "majority_direction", "artifact_identity"}
    if required.difference(frame.columns):
        raise ValueError(f"Missing forward columns: {sorted(required.difference(frame.columns))}")
    rows = frame.copy()
    if rows.empty or rows.duplicated(["ticker", "horizon", "forecast_origin"]).any():
        raise ValueError("Forward data is empty or contains duplicate predictions")
    for column in ("forecast_origin", "target_end", "captured_at", "label_available_at"):
        rows[column] = pd.to_datetime(rows[column], utc=True, errors="raise")
    frozen = pd.Timestamp(protocol["frozen_at"])
    if frozen.tzinfo is None:
        raise ValueError("Protocol freeze must be timezone-aware")
    if not rows.artifact_identity.eq(artifact_identity).all():
        raise ValueError("Forward artifact identity mismatch")
    invalid = rows.captured_at.lt(frozen) | rows.captured_at.ge(rows.target_end)
    invalid |= rows.captured_at.lt(rows.forecast_origin) | rows.label_available_at.lt(rows.target_end)
    invalid |= rows.target_end.le(rows.forecast_origin)
    # At least h weekdays must separate origin and target. Holidays can extend
    # this; they must never let a 1-day label masquerade as a 5-session label.
    weekday_distance = np.busday_count(rows.forecast_origin.dt.tz_localize(None).to_numpy(dtype="datetime64[D]"), rows.target_end.dt.tz_localize(None).to_numpy(dtype="datetime64[D]"))
    invalid |= weekday_distance < pd.to_numeric(rows.horizon, errors="coerce").to_numpy()
    invalid |= rows.label_available_at.gt(pd.Timestamp(datetime.now(timezone.utc)))
    if invalid.any() or not rows.horizon.isin(protocol["horizons"]).all():
        raise ValueError("Forward capture/label timestamps or horizons violate the frozen protocol")
    if not rows.majority_direction.isin([-1, 1]).all():
        raise ValueError("Majority direction must be frozen using training labels, not test labels")
    if not np.isfinite(rows[["actual", "prediction"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite forward values")
    reports = {}
    for horizon, group in rows.groupby("horizon"):
        if group.majority_direction.nunique() != 1:
            raise ValueError("Majority direction must remain fixed for the frozen artifact and horizon")
        actual, predicted = group.actual.to_numpy(float), group.prediction.to_numpy(float)
        dates = group.forecast_origin.to_numpy()
        candidate_correct = np.sign(predicted) == np.sign(actual)
        majority_correct = group.majority_direction.to_numpy() == np.sign(actual)
        nonflat = actual != 0
        direction_interval = {"status": "INSUFFICIENT_DATES", "lower": None, "upper": None}
        if nonflat.any():
            # Absolute error of 0/1 correctness against 1 is misclassification.
            direction_interval = paired_date_block_bootstrap(np.ones(nonflat.sum()), candidate_correct[nonflat].astype(float), majority_correct[nonflat].astype(float), dates[nonflat], block_sessions=max(5, int(horizon)))
        reports[str(int(horizon))] = {"independent": True, "protocol_id": protocol["protocol_id"], "predictor_identity": artifact_identity, "capture_started_at": group.captured_at.min().isoformat(), "sessions": int(group.forecast_origin.dt.normalize().nunique()), "capture_audit": "TIMESTAMPS_AND_PREDICTOR_ID_CHECKED_NOT_EXTERNAL_ATTESTATION", **regression_metrics(actual, predicted, train_direction=int(group.majority_direction.iloc[0])), "mae_improvement_ci": paired_date_block_bootstrap(actual, predicted, np.zeros_like(actual), dates, block_sessions=max(5, int(horizon))), "direction_improvement_ci": direction_interval}
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--forward-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New review report; never rewrites frozen artifact")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new evaluation output path")
    manifest, _, report, protocol = verify_artifact_directory(args.artifact)
    report["forward"] = evaluate_forward_rows(pd.read_csv(args.forward_ledger), protocol, artifact_identity=manifest["predictor_identity"])
    report["source_release_identity"] = manifest["artifact_identity"]
    report["forward_ledger_sha256"] = sha256_file(args.forward_ledger)
    report["gate"] = evaluate_model_gate(report, protocol).as_dict()
    report["promotion_note"] = "Review this NEW report; original artifact remains frozen. A separate reviewed artifact release is required to change its policy."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["gate"], indent=2))


if __name__ == "__main__":
    main()
