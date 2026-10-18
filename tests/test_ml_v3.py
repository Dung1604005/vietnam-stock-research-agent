from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_multi_agent.services.lstm_forecast_v3 import LstmV3ArtifactError, LstmV3ForecastModel
from stock_multi_agent.services.ml_baselines import baseline_predictions, evaluate_baselines
from stock_multi_agent.services.ml_dataset_v3 import build_dataset, write_dataset, refresh_runtime_bundle
from stock_multi_agent.services.ml_features_v3 import FEATURES, FEATURE_VERSION, add_targets, build_feature_frame, window_batch, window_endpoints
from stock_multi_agent.services.ml_model_registry import evaluate_model_gate, verify_artifact_directory, write_artifact_manifest
from stock_multi_agent.services.ml_validation import TrainOnlyScaler, expanding_purged_folds, paired_date_block_bootstrap, regression_metrics

ROOT = Path(__file__).resolve().parents[1]


def raw_rows(count=1200, ticker="FPT"):
    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, count)))
    return pd.DataFrame({"ticker": ticker, "date": pd.bdate_range("2020-01-01", periods=count), "open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": rng.integers(10000, 20000, count), "source": "synthetic-test-fixture", "source_id": f"fixture:{ticker}", "fetched_at": "2026-10-17T00:00:00+00:00", "price_adjustment": "UNKNOWN"})


def protocol():
    return json.loads((ROOT / "evaluation/configs/lstm-v3-protocol.json").read_text())


def research_report():
    return {"protocol_id": protocol()["protocol_id"], "folds": [{"fold_id": fold, "seed": seed, "horizon": horizon, "mae": 0.009, "zero_mae": 0.01, "ridge_mae": 0.009, "tree_mae": 0.009, "calibration_alpha": .5} for fold in range(3) for horizon in (1, 5) for seed in protocol()["seeds"]], "engineering_checks": {"feature_parity": True, "causality": True, "purged_splits": True, "train_only_scaling": True}, "price_adjustment": "ADJUSTED", "forward": {}}


def test_v3_features_causal_and_train_runtime_parity(market_data):
    original = raw_rows(180)
    future = original.copy()
    future.loc[130:, ["open", "high", "low", "close", "volume"]] *= 2
    one, two = build_feature_frame(original), build_feature_frame(future)
    np.testing.assert_allclose(one.loc[:129, FEATURES], two.loc[:129, FEATURES], equal_nan=True)
    columns = ["date", "open", "high", "low", "close", "volume"]
    runtime_raw = pd.DataFrame([bar.model_dump() for bar in market_data.bars])[columns]
    runtime_raw["ticker"] = market_data.ticker
    offline = build_feature_frame(runtime_raw)
    last_prefix = build_feature_frame(runtime_raw.iloc[:-5])
    np.testing.assert_allclose(offline.iloc[-6][FEATURES].to_numpy(float), last_prefix.iloc[-1][FEATURES].to_numpy(float), rtol=1e-7)
    assert not any("cross_section" in value or "liquidity_percentile" in value for value in FEATURES)


def test_builder_counts_provenance_duplicates_and_no_label_filter():
    raw = raw_rows(300)
    raw.loc[150, ["open", "high", "low", "close"]] *= 3
    duplicate = raw.iloc[[20]].copy()
    raw = pd.concat([raw, duplicate], ignore_index=True)
    result = build_dataset(raw)
    assert result.manifest["reconciled_counts"]
    assert result.manifest["input_rows"] == result.manifest["retained_rows"] + result.manifest["quarantined_rows"]
    assert result.manifest["quarantine_reasons"] == {"EXACT_DUPLICATE": 1}
    row = result.rows.iloc[149]
    assert row.target_return_1 > 1
    assert row.training_endpoint  # hard future labels are not dropped
    assert result.rows.price_adjustment.eq("UNKNOWN").all()
    assert result.rows.price_unit.eq("UNKNOWN").all()
    assert result.rows.available_at.notna().all()


def test_builder_invalid_bar_breaks_sequences_and_labels():
    raw = raw_rows(300)
    raw.loc[150, "high"] = 1
    built = build_dataset(raw)
    assert built.quarantine.quarantine_reason.tolist() == ["INVALID_OHLCV"]
    left = built.rows[built.rows.date < raw.iloc[150].date]
    right = built.rows[built.rows.date > raw.iloc[150].date]
    assert left.iloc[-1].segment_id != right.iloc[0].segment_id
    assert pd.isna(left.iloc[-1].target_return_1)
    assert not right.iloc[:88].training_endpoint.any()


def test_conflicting_duplicate_is_not_arbitrarily_picked():
    raw = raw_rows(300)
    duplicate = raw.iloc[[100]].copy()
    duplicate[["open", "high", "low", "close"]] *= 2
    result = build_dataset(pd.concat([raw, duplicate], ignore_index=True))
    assert result.manifest["quarantine_reasons"]["CONFLICTING_DUPLICATE"] == 2
    assert not result.rows.date.eq(raw.iloc[100].date).any()


def test_short_gap_and_scale_changes_reset_window():
    raw = raw_rows(300)
    raw.loc[150:, "price_adjustment"] = "ADJUSTED"
    result = build_dataset(raw)
    assert result.rows.iloc[149].segment_id != result.rows.iloc[150].segment_id
    assert not result.rows.iloc[150:238].training_endpoint.any()


def test_builder_preserves_old_output_and_portable_code(isolated_runtime_root):
    path = isolated_runtime_root / "ml-v3-test-data"
    result = build_dataset(raw_rows(200))
    manifest = write_dataset(result, path)
    assert (path / "ml_v3_runtime.zip").exists()
    assert manifest["runtime_bundle"]["sha256"]
    with pytest.raises(FileExistsError):
        write_dataset(result, path)


def test_purged_folds_no_label_leak_and_disjoint_outer_tests():
    frame = build_dataset(raw_rows()).rows
    endpoints = window_endpoints(frame)
    folds = expanding_purged_folds(frame, endpoints)
    used = set()
    for fold in folds:
        train, validation, calibration, test = [frame.iloc[getattr(fold, name)] for name in ("train", "validation", "calibration", "test")]
        assert train.target_end_5.max() < validation.date.min()
        assert validation.target_end_5.max() < calibration.date.min()
        assert calibration.target_end_5.max() < test.date.min()
        assert not used.intersection(fold.test)
        used.update(fold.test)
        assert len(fold.train) and len(fold.calibration)


def test_scaler_clips_using_train_only():
    train = np.array([[1., 2.], [2., 3.], [3., 4.]])
    scaler = TrainOnlyScaler().fit(train)
    before = scaler.specification()
    scaler.transform(np.array([[1e9, -1e9]]))
    assert before == scaler.specification()
    assert np.isfinite(scaler.transform(train)).all()


def test_window_batch_allocates_only_requested_endpoints():
    frame = build_dataset(raw_rows(200)).rows
    endpoints = window_endpoints(frame)
    values = frame[FEATURES].to_numpy(dtype=np.float32)
    batch = window_batch(values, endpoints[:3], 30)
    assert batch.shape == (3, 30, len(FEATURES))
    np.testing.assert_array_equal(batch[0, -1], values[endpoints[0]])


def test_baselines_same_rows_and_tree_optional():
    frame = build_dataset(raw_rows()).rows
    folds = expanding_purged_folds(frame, window_endpoints(frame))
    predictions, details = baseline_predictions(frame, folds[0], 1, include_tree=False)
    assert set(predictions) == {"zero", "persistence", "train_mean", "train_median", "ridge"}
    assert all(len(value) == len(folds[0].test) for value in predictions.values())
    assert details["selected_using"] == "VALIDATION_ONLY"
    report = evaluate_baselines(frame, folds[:1], include_tree=False)
    assert report["same_endpoints"] and len(report["reports"]) == 2


def test_date_block_bootstrap_and_flat_direction_metrics():
    dates = np.repeat(pd.bdate_range("2025-01-01", periods=50).to_numpy(), 2)
    actual = np.full(100, 0.01)
    ci = paired_date_block_bootstrap(actual, actual, np.zeros(100), dates, repetitions=50)
    assert ci["date_count"] == 50 and ci["lower"] > 0
    metrics = regression_metrics(np.zeros(5), np.zeros(5))
    assert metrics["flat_fraction"] == 1 and metrics["nonflat_directional_accuracy"] is None


def test_registry_fail_closed_pending_forward_and_missing_tree():
    report = research_report()
    gate = evaluate_model_gate(report, protocol())
    assert gate.validation_status == "PENDING_FORWARD_VALIDATION"
    assert not gate.policy_eligibility and gate.technical_blend_weight == 0
    report["folds"][0]["tree_mae"] = None
    assert evaluate_model_gate(report, protocol()).validation_status == "RESEARCH_GATE_PENDING_OR_REJECTED"
    report["pilot"] = True
    assert evaluate_model_gate(report, protocol()).validation_status == "PILOT_ONLY"


def test_registry_counts_seeds_not_independent_folds_and_direction_gate():
    report = research_report()
    report["folds"] = [dict(report["folds"][0], seed=seed) for seed in range(5)]
    assert not evaluate_model_gate(report, protocol()).policy_eligibility
    report = research_report()
    report["forward"]["1"] = {"independent": True, "protocol_id": protocol()["protocol_id"], "capture_started_at": "2026-10-18T00:00:00+00:00", "sessions": 50, "mae_improvement_ci": {"status": "ESTIMATED", "lower": .001}, "direction_improvement_ci": {"status": "ESTIMATED", "lower": -.001}}
    assert evaluate_model_gate(report, protocol()).validation_status == "INCONCLUSIVE"
    report["forward"]["1"]["direction_improvement_ci"]["lower"] = .01
    assert evaluate_model_gate(report, protocol()).technical_blend_weight == .2
    report["price_adjustment"] = "UNKNOWN"
    assert not evaluate_model_gate(report, protocol()).policy_eligibility


def make_artifact(path: Path, market_data):
    path.mkdir()
    rows = pd.DataFrame([dict(ticker=market_data.ticker, **bar.model_dump()) for bar in market_data.bars])
    frame = build_feature_frame(rows)
    scaler = TrainOnlyScaler().fit(frame.dropna(subset=FEATURES)[FEATURES].to_numpy(float))
    spec = {"model_version": "lstm-single-ticker-v3a", "protocol_id": protocol()["protocol_id"], "window": 30, "features": FEATURES, "feature_version": FEATURE_VERSION, "horizons": [1, 5], "scaler": scaler.specification(), "target_scale": [.02, .04], "calibration": {"1": {"alpha": .5, "bias": 0, "absolute_residual_quantile_90": .03}}, "price_unit": "UNKNOWN", "price_adjustment": "ADJUSTED"}
    (path / "model.keras").write_bytes(b"synthetic model bytes; fake model injected only in tests")
    for filename, value in (("spec.json", spec), ("evaluation.json", research_report()), ("protocol.json", protocol())):
        (path / filename).write_text(json.dumps(value))
    write_artifact_manifest(path, code_revision="test-fixture")


def test_adapter_feature_parity_and_shadow_weight(market_data, isolated_runtime_root):
    path = isolated_runtime_root / "ml-v3-artifact-good"
    make_artifact(path, market_data)
    class FakeModel:
        captured = None
        def predict(self, values, verbose=0):
            self.captured = values
            return np.array([[.1, .2]])
    model = FakeModel()
    adapter = LstmV3ForecastModel(path, keras_model=model)
    result = adapter.predict_next_session(market_data)
    assert result.forecast_price == pytest.approx(result.current_price * 1.001)
    assert model.captured.shape == (1, 30, len(FEATURES))
    if "technical_blend_weight" in type(result).model_fields:
        assert result.technical_blend_weight == 0
    assert adapter.gate.validation_status == "PENDING_FORWARD_VALIDATION"


def test_artifact_hash_schema_corruption_rejected(market_data, isolated_runtime_root):
    path = isolated_runtime_root / "ml-v3-artifact-corrupt"
    make_artifact(path, market_data)
    (path / "model.keras").write_bytes(b"tampered")
    with pytest.raises(LstmV3ArtifactError, match="hash/path"):
        LstmV3ForecastModel(path, keras_model=object())


def test_artifact_feature_order_rejected_even_with_new_hash(market_data, isolated_runtime_root):
    path = isolated_runtime_root / "ml-v3-artifact-order"
    make_artifact(path, market_data)
    spec = json.loads((path / "spec.json").read_text())
    spec["features"] = list(reversed(FEATURES))
    (path / "spec.json").write_text(json.dumps(spec))
    write_artifact_manifest(path)
    with pytest.raises(ValueError, match="feature version/order"):
        verify_artifact_directory(path)


def test_forward_evaluator_rejects_backdated_predictions():
    location = ROOT / "scripts/evaluate_lstm_v3.py"
    spec = importlib.util.spec_from_file_location("forward_v3_test", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = protocol()
    config["frozen_at"] = "2024-01-01T00:00:00+00:00"
    config["protocol_id"] = "synthetic-forward-test"
    rows = pd.DataFrame({"ticker": ["FPT"], "horizon": [1], "forecast_origin": ["2025-01-01"], "target_end": ["2025-01-02"], "captured_at": ["2025-01-03T00:00:00Z"], "label_available_at": ["2025-01-03T00:00:00Z"], "actual": [.01], "prediction": [.005], "majority_direction": [1], "artifact_identity": ["test"]})
    with pytest.raises(ValueError, match="timestamps"):
        module.evaluate_forward_rows(rows, config, artifact_identity="test")


def test_new_notebook_json_and_python_import_are_safe():
    notebook = json.loads((ROOT / "notebooks/train_lstm_kaggle_v3.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    script = ROOT / "notebooks/train_lstm_kaggle_v3.py"
    spec = importlib.util.spec_from_file_location("safe_v3_training_import", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)  # no TensorFlow import, GPU download or training


def test_writer_refuses_partial_directory_before_writing_csv(isolated_runtime_root):
    path = isolated_runtime_root / "ml-v3-partial-output"
    path.mkdir()
    (path / "ml_v3_runtime.zip").write_bytes(b"old interrupted experiment")
    with pytest.raises(FileExistsError, match="not empty"):
        write_dataset(build_dataset(raw_rows(200)), path)
    assert not (path / "vci_ohlcv_lstm_v3.csv").exists()


@pytest.mark.parametrize("column", ["source", "source_id"])
def test_builder_quarantines_empty_provenance(column):
    rows = raw_rows(200)
    rows.loc[100, column] = " "
    result = build_dataset(rows)
    assert result.quarantine.quarantine_reason.tolist() == ["INVALID_DATE_OR_PROVENANCE"]


def test_csv_roundtrip_preserves_endpoints_and_portable_runtime(isolated_runtime_root):
    import subprocess
    import sys
    result = build_dataset(raw_rows(300))
    path = isolated_runtime_root / "ml-v3-roundtrip"
    write_dataset(result, path)
    reloaded = pd.read_csv(path / "vci_ohlcv_lstm_v3.csv", parse_dates=["date", "target_end_1", "target_end_5"])
    np.testing.assert_array_equal(window_endpoints(result.rows), window_endpoints(reloaded))
    # A clean Python process can import the exact packaged runtime without the
    # application/Streamlit/LangGraph/TensorFlow dependency graph.
    script = "import sys; sys.path.insert(0, sys.argv[1]); from stock_multi_agent.services.ml_features_v3 import FEATURES; from stock_multi_agent.services.ml_model_registry import evaluate_model_gate; print(len(FEATURES))"
    completed = subprocess.run([sys.executable, "-I", "-c", script, str(path / "ml_v3_runtime.zip")], check=True, capture_output=True, text=True)
    assert completed.stdout.strip() == "17"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0.001"])
def test_registry_rejects_nonfinite_or_mistyped_metrics(bad):
    report = research_report()
    report["folds"][0]["ridge_mae"] = bad
    assert not evaluate_model_gate(report, protocol()).policy_eligibility
    assert evaluate_model_gate(report, protocol()).validation_status == "RESEARCH_GATE_PENDING_OR_REJECTED"


def test_registry_requires_registered_seeds_and_stable_protocol():
    report = research_report()
    report["folds"] = [item for item in report["folds"] if item["seed"] == 42]
    assert evaluate_model_gate(report, protocol()).validation_status == "RESEARCH_GATE_PENDING_OR_REJECTED"
    report = research_report()
    report["protocol_id"] = "different experiment"
    assert not evaluate_model_gate(report, protocol()).policy_eligibility


def test_registry_bootstrap_dates_count_days_not_intraday_rows():
    dates = pd.date_range("2025-01-01", periods=100, freq="min").to_numpy()
    result = paired_date_block_bootstrap(np.ones(100), np.ones(100), np.zeros(100), dates)
    assert result["date_count"] == 1 and result["status"] == "INSUFFICIENT_DATES"
    with pytest.raises(ValueError, match="aligned"):
        paired_date_block_bootstrap(np.ones(5), np.ones(4), np.ones(5), dates[:5])


def test_adapter_zero_signal_cannot_receive_weight(market_data, isolated_runtime_root):
    path = isolated_runtime_root / "ml-v3-zero-alpha"
    make_artifact(path, market_data)
    spec = json.loads((path / "spec.json").read_text())
    spec["calibration"]["1"]["alpha"] = 0
    (path / "spec.json").write_text(json.dumps(spec))
    write_artifact_manifest(path)
    class FakeModel:
        def predict(self, values, verbose=0):
            return np.zeros((1, 2))
    result = LstmV3ForecastModel(path, keras_model=FakeModel()).predict_next_session(market_data)
    assert result.technical_blend_weight == 0
    assert any("alpha=0" in warning for warning in result.warnings)


def test_predictor_identity_stable_across_evaluation_release(market_data, isolated_runtime_root):
    path = isolated_runtime_root / "ml-v3-predictor-identity"
    make_artifact(path, market_data)
    original, _, report, _ = verify_artifact_directory(path)
    report["forward"] = {"1": {"predictor_identity": original["predictor_identity"]}}
    (path / "evaluation.json").write_text(json.dumps(report))
    updated = write_artifact_manifest(path)
    assert updated["predictor_identity"] == original["predictor_identity"]
    assert updated["artifact_identity"] != original["artifact_identity"]
    verify_artifact_directory(path)
    report["forward"]["1"]["predictor_identity"] = "wrong-model"
    (path / "evaluation.json").write_text(json.dumps(report))
    write_artifact_manifest(path)
    with pytest.raises(ValueError, match="different frozen predictor"):
        verify_artifact_directory(path)


def test_runtime_refresh_preserves_data_and_original_zip(isolated_runtime_root):
    path = isolated_runtime_root / "ml-v3-runtime-refresh"
    result = build_dataset(raw_rows(200))
    original = write_dataset(result, path)
    refreshed = refresh_runtime_bundle(path)
    assert original["files"]["dataset"] == refreshed["files"]["dataset"]
    assert (path / original["runtime_bundle"]["filename"]).exists()
    assert refreshed["runtime_bundle_history"] == [original["runtime_bundle"]]
    assert refresh_runtime_bundle(path) == refreshed


def test_optional_tree_resource_failure_keeps_other_baselines(monkeypatch):
    import sys
    import types
    class DeniedTree:
        def __init__(self, **kwargs):
            pass
        def fit(self, *args):
            raise PermissionError("synthetic sandbox worker denial")
    fake = types.ModuleType("sklearn.ensemble")
    fake.HistGradientBoostingRegressor = DeniedTree
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", fake)
    frame = build_dataset(raw_rows()).rows
    fold = expanding_purged_folds(frame, window_endpoints(frame))[0]
    predictions, details = baseline_predictions(frame, fold, 1)
    assert "ridge" in predictions and "tree" not in predictions
    assert details["tree_status"] == "EXECUTION_UNAVAILABLE"
