"""Free baselines evaluated on the exact same endpoints as the LSTM."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stock_multi_agent.services.ml_features_v3 import FEATURES
from stock_multi_agent.services.ml_validation import TrainOnlyScaler, WalkForwardFold, regression_metrics


def baseline_predictions(frame: pd.DataFrame, fold: WalkForwardFold, horizon: int, *, include_tree: bool = True) -> tuple[dict[str, np.ndarray], dict]:
    target = f"target_return_{horizon}"
    train = frame.iloc[fold.train]
    y_train = train[target].to_numpy(dtype=float)
    y_val = frame.iloc[fold.validation][target].to_numpy(dtype=float)
    scaler = TrainOnlyScaler().fit(train[FEATURES].to_numpy(float))
    x_train = scaler.transform(train[FEATURES].to_numpy(float)).astype(float)
    x_val = scaler.transform(frame.iloc[fold.validation][FEATURES].to_numpy(float)).astype(float)
    x_test = scaler.transform(frame.iloc[fold.test][FEATURES].to_numpy(float)).astype(float)
    n = len(fold.test)
    predictions = {"zero": np.zeros(n), "train_mean": np.full(n, y_train.mean()), "train_median": np.full(n, np.median(y_train)), "persistence": frame.iloc[fold.test][f"return_{horizon}"].to_numpy(float)}
    # Ridge hyperparameter selection is validation-only, never outer test.
    design = np.column_stack([np.ones(len(x_train)), x_train])
    val_design = np.column_stack([np.ones(len(x_val)), x_val])
    best = None
    for alpha in (1.0, 100.0, 1000.0):
        penalty = np.eye(design.shape[1]) * alpha
        penalty[0, 0] = 0
        coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
        val_mae = float(np.mean(np.abs(y_val - val_design @ coefficient)))
        if best is None or val_mae < best[0]:
            best = (val_mae, alpha, coefficient)
    predictions["ridge"] = np.column_stack([np.ones(len(x_test)), x_test]) @ best[2]
    details = {"ridge_alpha": best[1], "ridge_validation_mae": best[0], "tree_status": "DISABLED", "fit_scope": "TRAIN_ONLY", "selected_using": "VALIDATION_ONLY"}
    if include_tree:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except ImportError:
            details["tree_status"] = "UNAVAILABLE_OPTIONAL_SKLEARN"
        else:
            tree = HistGradientBoostingRegressor(max_iter=60, max_leaf_nodes=7, l2_regularization=10, learning_rate=0.05, early_stopping=False, random_state=42)
            try:
                tree.fit(x_train, y_train)
                predictions["tree"] = tree.predict(x_test)
                details["tree_status"] = "EVALUATED"
            except (OSError, RuntimeError) as exc:
                # Resource-limited classroom environments may deny worker
                # pipes/threads. Keep other baseline results; missing tree
                # remains explicit and MUST fail the promotion gate.
                details["tree_status"] = "EXECUTION_UNAVAILABLE"
                details["tree_error_type"] = type(exc).__name__
    return predictions, details


def evaluate_baselines(frame: pd.DataFrame, folds: list[WalkForwardFold], *, include_tree: bool = True) -> dict:
    reports = []
    for fold in folds:
        for horizon in (1, 5):
            predictions, details = baseline_predictions(frame, fold, horizon, include_tree=include_tree)
            actual = frame.iloc[fold.test][f"target_return_{horizon}"].to_numpy(float)
            train_signs = np.sign(frame.iloc[fold.train][f"target_return_{horizon}"].to_numpy(float))
            train_sign = 1 if np.count_nonzero(train_signs > 0) >= np.count_nonzero(train_signs < 0) else -1
            reports.append({"fold_id": fold.fold_id, "horizon": horizon, "metrics": {name: regression_metrics(actual, predicted, train_direction=train_sign) for name, predicted in predictions.items()}, "details": details, "split": fold.manifest()})
    return {"schema_version": "ml-baselines-v3", "evaluation_scope": "DEVELOPMENT_WALK_FORWARD", "same_endpoints": True, "reports": reports}
