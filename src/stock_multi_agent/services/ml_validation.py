"""Expanding, purged chronological validation and train-only transforms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from stock_multi_agent.services.ml_features_v3 import FEATURES


@dataclass
class WalkForwardFold:
    fold_id: int
    train: np.ndarray
    validation: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    boundaries: dict[str, str]

    def manifest(self) -> dict[str, Any]:
        return {"fold_id": self.fold_id, **self.boundaries, "counts": {name: int(len(getattr(self, name))) for name in ("train", "validation", "calibration", "test")}}


def expanding_purged_folds(frame: pd.DataFrame, endpoints: np.ndarray, *, folds: int = 3, gap_sessions: int = 5) -> list[WalkForwardFold]:
    """Each endpoint belongs to at most one outer test; labels cannot cross splits.

    The last target horizon is used for purging. Gap counts use observed market
    dates (sessions), not calendar days. Validation is split into model-selection
    and a later, disjoint calibration block, each purged at its label boundary.
    """
    if folds < 1 or gap_sessions < 0:
        raise ValueError("folds must be positive and gap_sessions nonnegative")
    endpoints = np.asarray(endpoints, dtype=np.int64)
    if not len(endpoints):
        raise ValueError("No endpoints")
    origins = pd.to_datetime(frame.iloc[endpoints]["date"]).to_numpy(dtype="datetime64[ns]")
    target_end = pd.to_datetime(frame.iloc[endpoints]["target_end_5"]).to_numpy(dtype="datetime64[ns]")
    # Session gaps count the observed source calendar, not merely dates with
    # liquid/finite endpoints. Otherwise sparse eligibility compresses time.
    dates = np.sort(pd.to_datetime(frame["date"]).dropna().unique().to_numpy(dtype="datetime64[ns]"))
    if len(dates) < max(80, 20 * folds + gap_sessions * 4):
        raise ValueError("Too few distinct observed sessions for purged walk-forward folds")
    # Initial 45% training; remaining sessions divided into validation/test blocks.
    initial = int(len(dates) * 0.45)
    block = (len(dates) - initial) // (folds * 2)
    result: list[WalkForwardFold] = []
    for number in range(folds):
        train_end_i = initial + number * 2 * block - 1
        val_start_i = train_end_i + gap_sessions + 1
        val_end_i = initial + (number * 2 + 1) * block - 1
        cal_start_i = val_start_i + (val_end_i - val_start_i) // 2 + gap_sessions
        validation_end_i = cal_start_i - gap_sessions - 1
        test_start_i = val_end_i + gap_sessions + 1
        test_end_i = len(dates) - 1 if number == folds - 1 else initial + (number * 2 + 2) * block - 1
        if not (val_start_i <= validation_end_i < cal_start_i <= val_end_i < test_start_i <= test_end_i):
            raise ValueError("Not enough sessions for separate selection/calibration/test blocks; collect longer history")
        masks = {
            "train": (origins <= dates[train_end_i]) & (target_end <= dates[train_end_i]),
            "validation": (origins >= dates[val_start_i]) & (target_end <= dates[validation_end_i]),
            "calibration": (origins >= dates[cal_start_i]) & (target_end <= dates[val_end_i]),
            "test": (origins >= dates[test_start_i]) & (origins <= dates[test_end_i]) & (target_end <= dates[test_end_i]),
        }
        indices = {name: endpoints[mask] for name, mask in masks.items()}
        if any(len(value) == 0 for value in indices.values()):
            raise ValueError("Empty purged split; add history or reduce fold count before registering protocol")
        boundaries = {"train_end": str(pd.Timestamp(dates[train_end_i]).date()), "validation_start": str(pd.Timestamp(dates[val_start_i]).date()), "validation_end": str(pd.Timestamp(dates[validation_end_i]).date()), "calibration_start": str(pd.Timestamp(dates[cal_start_i]).date()), "calibration_end": str(pd.Timestamp(dates[val_end_i]).date()), "test_start": str(pd.Timestamp(dates[test_start_i]).date()), "test_end": str(pd.Timestamp(dates[test_end_i]).date()), "gap_sessions": str(gap_sessions)}
        result.append(WalkForwardFold(number, **indices, boundaries=boundaries))
    return result


class TrainOnlyScaler:
    def fit(self, values: np.ndarray, clip_quantile: float = 0.005) -> "TrainOnlyScaler":
        if not 0 <= clip_quantile < 0.5:
            raise ValueError("clip_quantile must be in [0, 0.5)")
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Scaler requires a finite nonempty train matrix")
        self.lower = np.quantile(values, clip_quantile, axis=0)
        self.upper = np.quantile(values, 1 - clip_quantile, axis=0)
        clipped = np.clip(values, self.lower, self.upper)
        self.mean = clipped.mean(axis=0)
        self.std = clipped.std(axis=0)
        self.std[self.std < 1e-8] = 1
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.clip(values, self.lower, self.upper) - self.mean) / self.std).astype(np.float32)

    def specification(self) -> dict:
        return {"features": FEATURES, "mean": self.mean.tolist(), "std": self.std.tolist(), "lower": self.lower.tolist(), "upper": self.upper.tolist(), "fit_scope": "TRAIN_ONLY"}


def regression_metrics(actual: np.ndarray, predicted: np.ndarray, *, train_direction: int = 1) -> dict:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if actual.shape != predicted.shape or not actual.size or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require aligned, finite, nonempty arrays")
    signs = np.sign(actual)
    nonflat = signs != 0
    classes = np.unique(signs[nonflat])
    balanced = np.mean([np.mean(np.sign(predicted[signs == label]) == label) for label in classes]) if len(classes) else None
    return {"mae": float(np.mean(np.abs(actual - predicted))), "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))), "directional_accuracy": float(np.mean(np.sign(predicted) == signs)), "nonflat_directional_accuracy": float(np.mean(np.sign(predicted[nonflat]) == signs[nonflat])) if nonflat.any() else None, "balanced_directional_accuracy": float(balanced) if balanced is not None else None, "nonflat_train_majority_accuracy": float(np.mean(signs[nonflat] == train_direction)) if nonflat.any() else None, "flat_fraction": float(np.mean(~nonflat)), "samples": int(actual.size)}


def paired_date_block_bootstrap(actual: np.ndarray, candidate: np.ndarray, reference: np.ndarray, dates: np.ndarray, *, block_sessions: int = 5, repetitions: int = 500, seed: int = 42) -> dict:
    """CI for reference MAE minus candidate MAE, resampling contiguous date blocks.

    Tickers on a date stay together. Date sums/counts preserve row weighting;
    overlapping horizon labels are handled by block dependence, not iid rows.
    """
    if block_sessions < 1 or repetitions < 20:
        raise ValueError("Invalid bootstrap configuration")
    # The reference and candidate must share arithmetic precision. In legacy
    # training, float32 zero MAE and float64 candidate MAE manufactured tiny wins.
    actual, candidate, reference = [np.asarray(value, dtype=np.float64) for value in (actual, candidate, reference)]
    dates = np.asarray(dates)
    if actual.ndim != 1 or any(value.shape != actual.shape for value in (candidate, reference, dates)):
        raise ValueError("Bootstrap requires aligned one-dimensional arrays")
    if pd.isna(dates).any():
        raise ValueError("Bootstrap dates cannot be missing")
    dates = pd.to_datetime(dates, utc=True).tz_localize(None).normalize().to_numpy()
    frame = pd.DataFrame({"date": dates, "delta": np.abs(actual - reference) - np.abs(actual - candidate)})
    if frame.empty or not np.isfinite(frame.delta).all():
        raise ValueError("Invalid bootstrap observations")
    grouped = frame.groupby("date", sort=True).delta.agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    if len(sums) < block_sessions * 2:
        return {"status": "INSUFFICIENT_DATES", "date_count": len(sums), "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(sums) - block_sessions + 1, size=int(np.ceil(len(sums) / block_sessions)))
        indices = (starts[:, None] + np.arange(block_sessions)[None, :]).reshape(-1)[:len(sums)]
        samples.append(float(sums[indices].sum() / counts[indices].sum()))
    return {"status": "ESTIMATED", "estimate": float(sums.sum() / counts.sum()), "lower": float(np.quantile(samples, 0.025)), "upper": float(np.quantile(samples, 0.975)), "date_count": len(sums), "block_sessions": block_sessions, "repetitions": repetitions}
