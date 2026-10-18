"""V3 Kaggle training entry point. Upload the NEW builder directory as a private
Dataset (CSV + manifest + ml_v3_runtime.zip + protocol.json), then import this
script or paste its Python code into a cell. Do not paste .ipynb JSON as Python.

No dependencies are installed and no training starts when this module is imported.
Run main([]) in a notebook or python train_lstm_kaggle_v3.py --help in a terminal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def _bootstrap_runtime(data: Path) -> None:
    metadata = json.loads((data.parent / "dataset.manifest.json").read_text())
    filename = metadata["runtime_bundle"]["filename"]
    if Path(filename).name != filename or not filename.endswith(".zip"):
        raise ValueError("Runtime bundle must be a local ZIP filename")
    bundle = data.parent / filename
    if not bundle.is_file():
        raise FileNotFoundError(f"Attach the runtime ZIP named by dataset.manifest.json: {filename}")
    if bundle.exists():
        actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
        if actual != metadata["runtime_bundle"]["sha256"]:
            raise ValueError("Shared training runtime bundle checksum mismatch")
        sys.path.insert(0, str(bundle))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/lstm-v3-run-1"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--pilot-endpoints", type=int, default=0, help="Training-only subsample per fold; pilot cannot qualify for promotion")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.folds < 1 or args.pilot_endpoints < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("epochs/batch/folds must be positive; pilot endpoints nonnegative; seeds unique")
    if args.data is None:
        matches = sorted(Path("/kaggle/input").rglob("vci_ohlcv_lstm_v3.csv"))
        if len(matches) != 1:
            raise FileNotFoundError("Attach exactly one NEW V3 builder dataset, or specify --data explicitly")
        args.data = matches[0]
    _bootstrap_runtime(args.data)
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from stock_multi_agent.services.ml_baselines import baseline_predictions
    from stock_multi_agent.services.ml_dataset_v3 import sha256_file
    from stock_multi_agent.services.ml_features_v3 import FEATURES, FEATURE_VERSION, window_batch, window_endpoints
    from stock_multi_agent.services.ml_model_registry import GATE_IMPLEMENTATION_VERSION, evaluate_model_gate, write_artifact_manifest
    from stock_multi_agent.services.ml_validation import TrainOnlyScaler, expanding_purged_folds, paired_date_block_bootstrap, regression_metrics

    protocol_path = args.data.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("feature_version") != FEATURE_VERSION or protocol.get("window") != 30 or protocol.get("horizons") != [1, 5]:
        raise ValueError("Shared feature/horizon/window contract differs from frozen protocol")
    if not args.pilot_endpoints and (args.folds != protocol["outer_folds"] or args.seeds != protocol["seeds"]):
        raise ValueError("Full runs must use registered folds/seeds; use --pilot-endpoints for non-promotable smoke runs")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Use a new --output directory or explicitly --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_manifest = json.loads((args.data.parent / "dataset.manifest.json").read_text())
    dataset_hash = sha256_file(args.data)
    if dataset_hash != dataset_manifest["files"]["dataset"]["sha256"]:
        raise ValueError("Dataset checksum mismatch")
    run_config = {"dataset_sha256": dataset_hash, "protocol_sha256": sha256_file(protocol_path), "epochs": args.epochs, "batch_size": args.batch_size, "folds": args.folds, "seeds": args.seeds, "pilot_endpoints": args.pilot_endpoints, "feature_version": FEATURE_VERSION, "evaluation_precision": "float64-v1", "gate_implementation_version": GATE_IMPLEMENTATION_VERSION}
    config_path = args.output / "run-config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise ValueError("Resume config/dataset/protocol differs; create a NEW experiment")
    config_path.write_text(json.dumps(run_config, indent=2))
    frame = pd.read_csv(args.data, parse_dates=["date", "target_end_1", "target_end_5"])
    endpoints = window_endpoints(frame, protocol["window"])
    folds = expanding_purged_folds(frame, endpoints, folds=args.folds, gap_sessions=protocol["gap_sessions"])
    targets = frame[["target_return_1", "target_return_5"]].to_numpy(dtype=np.float32)
    evaluation_targets = frame[["target_return_1", "target_return_5"]].to_numpy(dtype=np.float64)
    started = time.perf_counter()
    report = {"schema_version": "lstm-evaluation-v3", "protocol_id": protocol["protocol_id"], "dataset_sha256": dataset_hash, "scope": "DEVELOPMENT_WALK_FORWARD", "pilot": bool(args.pilot_endpoints or dataset_manifest.get("pilot_row_limit")), "folds": [], "forward": {}, "engineering_checks": {}, "price_adjustment": "ADJUSTED" if frame.price_adjustment.eq("ADJUSTED").all() else "UNKNOWN", "limitations": ["Historical data already seen in prior experiments is development, not independent holdout", "Current-listing universe is not point-in-time", "Engineering checks require a separately reviewed test run before promotion"]}

    class Batches(tf.keras.utils.Sequence):
        def __init__(self, values, indices, target_scale, *, shuffle=False, seed=42, **kwargs):
            super().__init__(**kwargs)
            self.values, self.indices, self.target_scale = values, np.array(indices), target_scale
            self.shuffle = shuffle
            self.rng = np.random.default_rng(seed)
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.indices) / args.batch_size))

        def __getitem__(self, index):
            batch = self.indices[index * args.batch_size:(index + 1) * args.batch_size]
            return window_batch(self.values, batch, protocol["window"]), targets[batch] / self.target_scale

        def on_epoch_end(self):
            if self.shuffle:
                self.rng.shuffle(self.indices)

    def predict(model, values, indices):
        result = []
        for start in range(0, len(indices), args.batch_size):
            result.append(model.predict_on_batch(window_batch(values, indices[start:start + args.batch_size], protocol["window"])))
        return np.concatenate(result, axis=0)

    selected_spec = None
    final_fold_validation_loss = float("inf")
    for fold in folds:
        scaler = TrainOnlyScaler().fit(frame.iloc[fold.train][FEATURES].to_numpy(float))
        values = scaler.transform(frame[FEATURES].to_numpy(float))  # O(rows x features), never O(rows x window x features)
        scale = np.maximum(targets[fold.train].std(axis=0), 1e-5)
        baseline_by_h = {h: baseline_predictions(frame, fold, h) for h in (1, 5)}
        for seed in args.seeds:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            trial = args.output / f"fold-{fold.fold_id}-seed-{seed}"
            trial.mkdir(exist_ok=True)
            checkpoint = trial / "best.keras"
            trial_result_path = trial / "results.json"
            if args.resume and trial_result_path.exists():
                saved = json.loads(trial_result_path.read_text())
                if saved.get("checkpoint_sha256") != sha256_file(checkpoint):
                    raise ValueError(f"Completed trial checkpoint changed: {trial.name}; choose a NEW run")
                report["folds"].extend(saved["results"])
                if fold.fold_id == folds[-1].fold_id and saved["validation_loss"] < final_fold_validation_loss:
                    final_fold_validation_loss = saved["validation_loss"]
                    selected_spec = saved["spec"]
                    import shutil
                    shutil.copy2(checkpoint, args.output / "model.keras")
                continue
            train_indices = fold.train
            if args.pilot_endpoints and len(train_indices) > args.pilot_endpoints:
                train_indices = np.random.default_rng(seed).choice(train_indices, args.pilot_endpoints, replace=False)
            if args.resume and checkpoint.exists():
                model = tf.keras.models.load_model(checkpoint, safe_mode=True)
                print("Resuming checkpoint weights/optimizer; callback patience restarts:", trial.name)
            else:
                model = tf.keras.Sequential([
                    tf.keras.layers.Input((protocol["window"], len(FEATURES))),
                    tf.keras.layers.LSTM(24, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
                    tf.keras.layers.Dropout(0.15),
                    tf.keras.layers.Dense(12, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
                    tf.keras.layers.Dense(2, kernel_initializer="zeros", bias_initializer="zeros"),
                ], name="single_ticker_lstm_v3a")
                model.compile(optimizer=tf.keras.optimizers.Adam(3e-4, clipnorm=1), loss="mae")
            history = model.fit(Batches(values, train_indices, scale, shuffle=True, seed=seed), validation_data=Batches(values, fold.validation, scale), epochs=args.epochs, callbacks=[tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True), tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5)], verbose=2)
            pd.DataFrame(history.history).to_csv(trial / "history.csv", index=False)
            model = tf.keras.models.load_model(checkpoint, safe_mode=True)
            raw_calibration = predict(model, values, fold.calibration).astype(np.float64) * scale
            raw_test = predict(model, values, fold.test).astype(np.float64) * scale
            calibration = {}
            trial_results = []
            prediction_rows = []
            for column, horizon in enumerate((1, 5)):
                actual_cal = evaluation_targets[fold.calibration, column]
                # Calibration alone picks shrinkage and bias. Outer test is never consulted.
                options = []
                for alpha in np.linspace(0, 1, 11):
                    bias = float(np.median(actual_cal - alpha * raw_calibration[:, column]))
                    residual = actual_cal - (alpha * raw_calibration[:, column] + bias)
                    options.append((float(np.mean(np.abs(residual))), float(alpha), bias, residual))
                _, alpha, bias, residual = min(options, key=lambda item: item[0])
                radius = float(np.quantile(np.abs(residual), 0.9, method="higher"))
                calibration[str(horizon)] = {"alpha": alpha, "bias": bias, "absolute_residual_quantile_90": radius, "fit_scope": "DISJOINT_CALIBRATION_BLOCK", "calibration_samples": len(actual_cal)}
                predicted = alpha * raw_test[:, column] + bias
                actual = evaluation_targets[fold.test, column]
                baselines, details = baseline_by_h[horizon]
                dates = frame.iloc[fold.test].date.to_numpy()
                train_signs = np.sign(targets[fold.train, column])
                majority = 1 if np.count_nonzero(train_signs > 0) >= np.count_nonzero(train_signs < 0) else -1
                metrics = regression_metrics(actual, predicted, train_direction=majority)
                baseline_mae = {name: regression_metrics(actual, prediction)["mae"] for name, prediction in baselines.items()}
                result = {"fold_id": fold.fold_id, "seed": seed, "horizon": horizon, **metrics, "metric_precision": "float64", "zero_mae": baseline_mae["zero"], "ridge_mae": baseline_mae["ridge"], "tree_mae": baseline_mae.get("tree"), "tree_status": details["tree_status"], "mae_improvement_ci": paired_date_block_bootstrap(actual, predicted, baselines["zero"], dates, **{key: protocol["bootstrap"][key] for key in ("block_sessions", "repetitions")}), "interval_coverage": float(np.mean(np.abs(actual - predicted) <= radius)), "interval_average_width_return": 2 * radius, "split": fold.manifest(), "calibration_alpha": alpha, "uses_lstm_signal": alpha > 0}
                trial_results.append(result)
                part = frame.iloc[fold.test][["ticker", "date", "target_end_1", "target_end_5", "active_ratio_20"]].copy()
                part["actual"], part["prediction"], part["horizon"], part["fold_id"], part["seed"] = actual, predicted, horizon, fold.fold_id, seed
                part["ridge"], part["zero"] = baselines["ridge"], 0.0
                if "tree" in baselines:
                    part["tree"] = baselines["tree"]
                prediction_rows.append(part)
                result["subgroups"] = {"activity_100pct": regression_metrics(actual[part.active_ratio_20.eq(1)], predicted[part.active_ratio_20.eq(1)]) if part.active_ratio_20.eq(1).any() else None, "activity_below_100pct": regression_metrics(actual[part.active_ratio_20.lt(1)], predicted[part.active_ratio_20.lt(1)]) if part.active_ratio_20.lt(1).any() else None}
            pd.concat(prediction_rows).to_csv(trial / "predictions.csv", index=False)
            validation_loss = float(min(history.history["val_loss"]))
            spec = {"schema_version": "lstm-spec-v3", "model_version": "lstm-single-ticker-v3a", "protocol_id": protocol["protocol_id"], "feature_version": FEATURE_VERSION, "features": FEATURES, "window": protocol["window"], "horizons": [1, 5], "scaler": scaler.specification(), "target_scale": scale.tolist(), "calibration": calibration, "price_unit": str(frame.price_unit.iloc[0]) if frame.price_unit.nunique() == 1 else "MIXED_UNAVAILABLE", "price_adjustment": report["price_adjustment"], "max_gap_business_days": dataset_manifest["max_gap_business_days"], "seed": seed, "selected_fold": fold.fold_id, "selected_using": "LATEST_FOLD_VALIDATION_LOSS_ONLY", "dataset_sha256": dataset_hash}
            trial_result_path.write_text(json.dumps({"results": trial_results, "validation_loss": validation_loss, "spec": spec, "checkpoint_sha256": sha256_file(checkpoint)}, indent=2))
            report["folds"].extend(trial_results)
            if fold.fold_id == folds[-1].fold_id and validation_loss < final_fold_validation_loss:
                final_fold_validation_loss = validation_loss
                selected_spec = spec
                import shutil
                shutil.copy2(checkpoint, args.output / "model.keras")
    report["elapsed_seconds"] = time.perf_counter() - started
    report["metric_precision"] = "float64"
    report["gate_implementation_version"] = GATE_IMPLEMENTATION_VERSION
    report["tensorflow_version"] = tf.__version__
    report["gate"] = evaluate_model_gate(report, protocol).as_dict()
    (args.output / "spec.json").write_text(json.dumps(selected_spec, indent=2))
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2))
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    write_artifact_manifest(args.output, code_revision=os.environ.get("STOCK_AGENT_CODE_REVISION", "UNRECORDED"))
    print(json.dumps({"status": report["gate"], "artifact_directory": str(args.output), "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    # Notebook kernels pass unrelated argv; run main([]) when importing in a cell.
    main()
