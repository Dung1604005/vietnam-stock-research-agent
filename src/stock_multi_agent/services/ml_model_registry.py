"""Fail-closed artifact validation and separate research/policy gates for V3."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from stock_multi_agent.services.ml_dataset_v3 import sha256_file
from stock_multi_agent.services.ml_features_v3 import FEATURE_VERSION, FEATURES


# Numerical equivalence only, NOT a tunable economic improvement threshold.
# Covers float32 aggregation versus float64 metrics in legacy V3 reports.
GATE_IMPLEMENTATION_VERSION = "v3-gate-numeric-fix-1"
MAE_ABSOLUTE_TOLERANCE = 1e-10
MAE_RELATIVE_TOLERANCE = 1e-6


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mae_strictly_better(candidate: float, reference: float) -> bool:
    """A tie at floating-point precision is not evidence of improvement."""
    if not _finite_number(candidate) or not _finite_number(reference) or candidate < 0 or reference <= 0:
        return False
    tolerance = max(MAE_ABSOLUTE_TOLERANCE, MAE_RELATIVE_TOLERANCE * max(abs(candidate), abs(reference)))
    return reference - candidate > tolerance


def research_gate_diagnostics(report: dict, protocol: dict, *, horizon: int = 1) -> dict:
    limits = protocol["gate"]
    folds = [item for item in report.get("folds", []) if item.get("horizon") == horizon]
    fold_ids = {item.get("fold_id") for item in folds}
    registered_seeds = set(protocol.get("seeds", []))
    baseline_ok = True
    summaries = []
    for fold_id in sorted(fold_ids, key=str):
        members = [item for item in folds if item.get("fold_id") == fold_id]
        seeds_ok = {item.get("seed") for item in members} == registered_seeds and len(members) == len(registered_seeds)
        baseline_ok &= seeds_ok
        member_summaries = []
        for item in members:
            valid = _finite_number(item.get("mae")) and item["mae"] >= 0
            valid &= _finite_number(item.get("zero_mae")) and item["zero_mae"] > 0
            signal = _finite_number(item.get("calibration_alpha")) and 0 < item["calibration_alpha"] <= 1 and item.get("uses_lstm_signal") is not False
            better = valid and mae_strictly_better(item["mae"], item["zero_mae"])
            member_summaries.append({"seed": item.get("seed"), "strict_mae_win": bool(better), "uses_lstm_signal": bool(signal)})
            baseline_ok &= valid
            for baseline in ("ridge_mae", "tree_mae"):
                value = item.get(baseline)
                if not _finite_number(value) or value <= 0 or not valid or item["mae"] > value * (1 + limits["max_relative_loss_vs_ridge_tree"]):
                    baseline_ok = False
        # A seed is a repeated trial, not a new fold. A constant calibration
        # baseline cannot be counted as a winning LSTM even if its bias helps.
        wins = seeds_ok and all(row["strict_mae_win"] and row["uses_lstm_signal"] for row in member_summaries)
        summaries.append({"fold_id": fold_id, "winning_fold": bool(wins), "members": member_summaries})
    winning_folds = sum(row["winning_fold"] for row in summaries)
    passed = (fold_ids == set(range(protocol.get("outer_folds", 3)))
        and len(fold_ids) >= limits["minimum_outer_folds"]
        and winning_folds >= limits["minimum_winning_folds"] and baseline_ok)
    return {"gate_implementation_version": GATE_IMPLEMENTATION_VERSION, "horizon": horizon,
        "winning_folds": winning_folds, "required_winning_folds": limits["minimum_winning_folds"],
        "ridge_tree_and_seed_checks_passed": bool(baseline_ok), "passed": bool(passed), "folds": summaries,
        "numerical_tolerance": {"absolute": MAE_ABSOLUTE_TOLERANCE, "relative": MAE_RELATIVE_TOLERANCE}}


@dataclass(frozen=True)
class ModelGate:
    artifact_status: str
    validation_status: str
    policy_eligibility: bool
    technical_blend_weight: float
    reasons: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"artifact_status": self.artifact_status, "validation_status": self.validation_status, "policy_eligibility": self.policy_eligibility, "technical_blend_weight": self.technical_blend_weight, "reasons": list(self.reasons)}


def evaluate_model_gate(report: dict, protocol: dict, *, horizon: int = 1) -> ModelGate:
    reasons: list[str] = []
    limits = protocol["gate"]
    if report.get("pilot") is True:
        return ModelGate("VERIFIED_FILES", "PILOT_ONLY", False, 0.0, ("Pilot runs cannot be promoted",))
    finite = _finite_number

    if report.get("protocol_id") != protocol.get("protocol_id") or horizon not in protocol.get("horizons", []):
        return ModelGate("VERIFIED_FILES", "RESEARCH_GATE_PENDING_OR_REJECTED", False, 0.0, ("Protocol identity/horizon mismatch",))
    folds = [item for item in report.get("folds", []) if item.get("horizon") == horizon]
    fold_ids = {item.get("fold_id") for item in folds}
    diagnostics = research_gate_diagnostics(report, protocol, horizon=horizon)
    if not diagnostics["passed"]:
        reasons.append("Development walk-forward has not cleared all pre-registered fold and Ridge/tree gates "
            f"({diagnostics['winning_folds']}/{diagnostics['required_winning_folds']} required winning folds; numerical ties and alpha=0 are not wins)")
    # Report all known blockers, not merely the first one encountered.
    if report.get("engineering_checks") != {"feature_parity": True, "causality": True, "purged_splits": True, "train_only_scaling": True}:
        reasons.append("Engineering checks are missing or not all verified")
    if limits.get("require_adjusted_prices_for_blend") and report.get("price_adjustment") != "ADJUSTED":
        reasons.append("Verified adjusted prices are required for Technical blend")
    latest = [item for item in folds if item.get("fold_id") == protocol.get("outer_folds", 3) - 1]
    if not any(finite(item.get("calibration_alpha")) and item["calibration_alpha"] > 0 for item in latest):
        reasons.append("Latest-fold model has no calibrated LSTM signal; constant baselines cannot receive LSTM blend weight")
    forward = report.get("forward", {}).get(str(horizon), {})
    independent = forward.get("independent") is True and forward.get("protocol_id") == protocol.get("protocol_id")
    try:
        frozen = datetime.fromisoformat(protocol["frozen_at"].replace("Z", "+00:00"))
        captured = datetime.fromisoformat(forward.get("capture_started_at", "").replace("Z", "+00:00"))
        independent &= frozen.utcoffset() is not None and captured.utcoffset() is not None and captured >= frozen
    except (ValueError, TypeError, KeyError):
        independent = False
    sessions = forward.get("sessions", 0)
    forward_pending = not independent or not isinstance(sessions, int) or isinstance(sessions, bool) or sessions < limits["minimum_forward_sessions"]
    if forward_pending:
        reasons.append("A new, protocol-locked independent forward holdout with enough observed sessions is required")
    else:
        for key in ("mae_improvement_ci", "direction_improvement_ci"):
            interval = forward.get(key, {})
            if interval.get("status") != "ESTIMATED" or not finite(interval.get("lower")) or interval["lower"] <= 0:
                reasons.append(f"{key} is inconclusive or does not improve the registered baseline")
    if not diagnostics["passed"]:
        return ModelGate("VERIFIED_FILES", "RESEARCH_GATE_PENDING_OR_REJECTED", False, 0.0, tuple(reasons))
    if forward_pending:
        return ModelGate("VERIFIED_FILES", "PENDING_FORWARD_VALIDATION", False, 0.0, tuple(reasons))
    if reasons:
        return ModelGate("VERIFIED_FILES", "INCONCLUSIVE", False, 0.0, tuple(reasons))
    return ModelGate("VERIFIED_FILES", "FORWARD_VALIDATED", True, min(0.2, limits["maximum_technical_weight"]), ())


def verify_artifact_directory(directory: Path) -> tuple[dict, dict, dict, dict]:
    """No arbitrary manifest paths or custom Keras objects; all files are pinned."""
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "lstm-artifact-v3":
        raise ValueError("Unsupported V3 manifest schema")
    required = {"model.keras", "spec.json", "evaluation.json", "protocol.json"}
    files = manifest.get("files", {})
    if set(files) != required:
        raise ValueError("Manifest must pin exactly the V3 model/spec/evaluation/protocol files")
    for filename in required:
        path = directory / filename
        if not path.is_file() or path.is_symlink() or path.resolve().parent != directory or sha256_file(path) != files[filename]:
            raise ValueError(f"V3 artifact hash/path mismatch: {filename}")
    spec = json.loads((directory / "spec.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if spec.get("feature_version") != FEATURE_VERSION or spec.get("features") != FEATURES:
        raise ValueError("V3 shared feature version/order mismatch")
    if spec.get("horizons") != [1, 5] or spec.get("window") != protocol.get("window") or spec.get("window") != 30:
        raise ValueError("V3 horizon/window mismatch")
    if spec.get("protocol_id") != protocol.get("protocol_id") or report.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("Protocol identity mismatch")
    expected_identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if manifest.get("artifact_identity") != expected_identity:
        raise ValueError("Artifact identity does not match its pinned files")
    predictor_identity = hashlib.sha256(json.dumps({name: files[name] for name in ("model.keras", "spec.json", "protocol.json")}, sort_keys=True).encode()).hexdigest()
    if manifest.get("predictor_identity") != predictor_identity:
        raise ValueError("Predictor identity does not match model/spec/protocol")
    for forward in report.get("forward", {}).values():
        if forward.get("predictor_identity") != predictor_identity:
            raise ValueError("Forward evidence belongs to a different frozen predictor")
    return manifest, spec, report, protocol


def write_artifact_manifest(directory: Path, *, code_revision: str = "UNRECORDED") -> dict:
    files = {name: sha256_file(directory / name) for name in ("model.keras", "spec.json", "evaluation.json", "protocol.json")}
    predictor_identity = hashlib.sha256(json.dumps({name: files[name] for name in ("model.keras", "spec.json", "protocol.json")}, sort_keys=True).encode()).hexdigest()
    manifest = {"schema_version": "lstm-artifact-v3", "files": files, "code_revision": code_revision, "artifact_identity": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), "predictor_identity": predictor_identity, "identity_policy": "predictor_identity excludes evaluation; artifact_identity pins the complete reviewed release", "default_policy": "SHADOW_UNTIL_FORWARD_GATE"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
