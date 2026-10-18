"""Causal, single-ticker V3 features shared by training and serving.

Rows are retained through warm-up. Eligibility applies to window endpoints, never
to future returns. No cross-sectional proxy is silently substituted at serving.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_VERSION = "single-ticker-v3a-1"
FEATURES = [
    "return_1", "return_2", "return_5", "return_10", "return_20",
    "open_gap", "intraday_range", "close_position", "log_volume",
    "volume_ratio_5", "volume_ratio_20", "ma_distance_5", "ma_distance_20",
    "ma_distance_60", "volatility_5", "volatility_20", "active_ratio_20",
]
HORIZONS = (1, 5)


def build_feature_frame(raw: pd.DataFrame, *, max_gap_business_days: int = 5) -> pd.DataFrame:
    """Use only information at/before each row; future rows cannot change features.

    ``segment_id`` from the builder is preserved. Without it, unverified gaps
    exceeding five weekdays create segments; weekends are not fabricated bars.
    This conservative rule is not an exchange-holiday calendar certification.
    """
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/date rows must be reconciled before feature construction")
    if "segment_id" not in frame:
        gap = frame.groupby("ticker")["date"].diff().dt.days.fillna(0)
        previous = frame.groupby("ticker")["date"].shift(1)
        weekdays = np.zeros(len(frame), dtype=np.int64)
        present = previous.notna()
        weekdays[present] = np.busday_count(
            previous[present].to_numpy(dtype="datetime64[D]"),
            frame.loc[present, "date"].to_numpy(dtype="datetime64[D]"),
        )
        breaks = frame["ticker"].ne(frame["ticker"].shift()) | (weekdays > max_gap_business_days) | (gap < 0)
        frame["segment_id"] = breaks.cumsum().astype(int)
    group = frame.groupby(["ticker", "segment_id"], sort=False)
    previous_close = group["close"].shift(1)
    for period in (1, 2, 5, 10, 20):
        frame[f"return_{period}"] = group["close"].pct_change(period, fill_method=None)
    frame["open_gap"] = frame["open"] / previous_close - 1
    frame["intraday_range"] = frame["high"] / frame["low"] - 1
    spread = frame["high"] - frame["low"]
    frame["close_position"] = ((frame["close"] - frame["low"]) / spread.replace(0, np.nan)).where(spread.ne(0), 0.5)
    frame["log_volume"] = np.log1p(frame["volume"])
    for period in (5, 20):
        volume_mean = group["volume"].transform(lambda values: values.rolling(period, min_periods=period).mean())
        frame[f"volume_ratio_{period}"] = frame["volume"] / volume_mean.replace(0, np.nan)
    for period in (5, 20, 60):
        price_mean = group["close"].transform(lambda values: values.rolling(period, min_periods=period).mean())
        frame[f"ma_distance_{period}"] = frame["close"] / price_mean - 1
    for period in (5, 20):
        frame[f"volatility_{period}"] = frame.groupby(["ticker", "segment_id"])["return_1"].transform(
            lambda values: values.rolling(period, min_periods=period).std(ddof=1)
        )
    frame["active_ratio_20"] = group["volume"].transform(lambda values: values.gt(0).rolling(20, min_periods=20).mean())
    finite = np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all(axis=1)
    frame["feature_valid"] = finite
    frame["endpoint_eligible"] = finite & frame["volume"].gt(0) & frame["active_ratio_20"].ge(0.5)
    return frame


def add_targets(frame: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Create labels within segments, never use labels to select liquid endpoints."""
    result = frame.copy()
    group = result.groupby(["ticker", "segment_id"], sort=False)
    for horizon in horizons:
        result[f"target_return_{horizon}"] = group["close"].shift(-horizon) / result["close"] - 1
        result[f"target_end_{horizon}"] = group["date"].shift(-horizon)
    result["forecast_origin"] = result["date"]
    result["feature_end"] = result["date"]
    return result


def window_endpoints(frame: pd.DataFrame, window: int = 30, *, require_targets: bool = True) -> np.ndarray:
    """Return positional endpoints with contiguous, finite same-segment context."""
    if window < 1:
        raise ValueError("window must be positive")
    valid = frame["feature_valid"].astype(int)
    context = valid.groupby([frame["ticker"], frame["segment_id"]]).transform(
        lambda values: values.rolling(window, min_periods=window).sum().eq(window)
    )
    eligible = context & frame["endpoint_eligible"]
    if require_targets:
        eligible &= np.isfinite(frame[[f"target_return_{h}" for h in HORIZONS]].to_numpy(dtype=float)).all(axis=1)
        eligible &= frame[[f"target_end_{h}" for h in HORIZONS]].notna().all(axis=1)
    return np.flatnonzero(eligible.to_numpy())


def window_batch(values: np.ndarray, endpoints: np.ndarray, window: int) -> np.ndarray:
    """Allocate only one batch, not N x window x features for the full dataset."""
    offsets = np.arange(window - 1, -1, -1)
    return values[np.asarray(endpoints)[:, None] - offsets[None, :]]
