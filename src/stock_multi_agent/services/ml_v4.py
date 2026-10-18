"""V4 causal features, eligible-session splits and flat-aware calibration.

No TensorFlow import, downloads, model loading or training at import time.
V3 source data/targets and V2 artifacts are never modified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stock_multi_agent.services.ml_features_v3 import FEATURES as V3_FEATURES, build_feature_frame, window_endpoints
from stock_multi_agent.services.ml_validation import WalkForwardFold, TrainOnlyScaler, regression_metrics

FEATURE_VERSION = 'single-ticker-v4-1'
FEATURES = [*V3_FEATURES, 'flat_ratio_5', 'flat_ratio_20', 'volume_cv_20', 'turnover_ratio_20',
    'candle_body_return', 'upper_wick_ratio', 'lower_wick_ratio', 'return_mean_5']


def feature_frame(raw: pd.DataFrame, *, reuse_verified_v3_features=False) -> pd.DataFrame:
    if reuse_verified_v3_features:
        # Only the trainer may opt in after hashing the frozen V3 CSV. Runtime
        # builds from raw bars by default; tests prove both paths are identical.
        required = set(V3_FEATURES) | {'ticker', 'date', 'segment_id', 'open', 'high', 'low', 'close', 'volume'}
        if required.difference(raw.columns):
            raise ValueError('Verified V3 cache is missing feature/source columns')
        frame = raw.copy()
        frame['date'] = pd.to_datetime(frame.date).dt.tz_localize(None).dt.normalize()
        frame = frame.sort_values(['ticker', 'date']).reset_index(drop=True)
        if frame.duplicated(['ticker', 'date']).any():
            raise ValueError('Duplicate ticker/date in V3 cache')
    else:
        frame = build_feature_frame(raw)
    keys = [frame.ticker, frame.segment_id]
    previous_close = frame.groupby(['ticker', 'segment_id'], sort=False).close.shift(1)
    flat = frame.return_1.abs().le(1e-12).astype(float).where(frame.return_1.notna())
    for period in (5, 20):
        frame[f'flat_ratio_{period}'] = flat.groupby(keys).transform(lambda x: x.rolling(period, min_periods=period).mean())
    grouped = frame.groupby(['ticker', 'segment_id'], sort=False)
    vol_mean = grouped.volume.transform(lambda x: x.rolling(20, min_periods=20).mean())
    vol_std = grouped.volume.transform(lambda x: x.rolling(20, min_periods=20).std(ddof=1))
    frame['volume_cv_20'] = vol_std / vol_mean.replace(0, np.nan)
    turnover = frame.close * frame.volume
    frame['turnover_ratio_20'] = turnover / turnover.groupby(keys).transform(lambda x: x.rolling(20, min_periods=20).mean()).replace(0, np.nan)
    frame['candle_body_return'] = (frame.close - frame.open) / previous_close
    spread = frame.high - frame.low
    frame['upper_wick_ratio'] = ((frame.high - frame[['open', 'close']].max(axis=1)) / spread.replace(0, np.nan)).where(spread.ne(0), 0.)
    frame['lower_wick_ratio'] = ((frame[['open', 'close']].min(axis=1) - frame.low) / spread.replace(0, np.nan)).where(spread.ne(0), 0.)
    frame['return_mean_5'] = frame.return_1.groupby(keys).transform(lambda x: x.rolling(5, min_periods=5).mean())
    finite = np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all(axis=1)
    frame['feature_valid'] = finite
    frame['endpoint_eligible'] = finite & frame.volume.gt(0) & frame.active_ratio_20.ge(.5)
    return frame


def eligible_session_folds(frame, endpoints, *, folds=3, gap_sessions=5, minimum_sessions=None):
    """Allocate by actual eligible dates; still count gap on full source calendar.

    No future-return value is used for boundaries. Availability of target dates is
    required for purging only. Sparse validation never silently passes as months.
    """
    minimum_sessions = minimum_sessions or {'train': 120, 'validation': 30, 'calibration': 30, 'test': 40}
    if folds < 1 or gap_sessions < 0:
        raise ValueError('Invalid folds/gap')
    endpoints = np.asarray(endpoints, dtype=np.int64)
    if not len(endpoints):
        raise ValueError('No eligible endpoints')
    origins = pd.to_datetime(frame.iloc[endpoints].date).to_numpy(dtype='datetime64[ns]')
    labels = pd.to_datetime(frame.iloc[endpoints].target_end_5).to_numpy(dtype='datetime64[ns]')
    eligible_dates = np.unique(origins)
    calendar = np.sort(pd.to_datetime(frame.date).dropna().unique().to_numpy(dtype='datetime64[ns]'))
    initial = int(len(eligible_dates) * .45)
    block = (len(eligible_dates) - initial) // (folds * 2)
    if block < 4:
        raise ValueError('Too few eligible dates; collect more history')
    def after_gap(value):
        index = np.searchsorted(calendar, value) + gap_sessions + 1
        if index >= len(calendar):
            raise ValueError('Not enough source sessions for gap')
        return calendar[index]
    result = []
    for number in range(folds):
        train_end = eligible_dates[initial + number * 2 * block - 1]
        val_end = eligible_dates[initial + (number * 2 + 1) * block - 1]
        selection_end = eligible_dates[initial + number * 2 * block + block // 2 - 1]
        test_end = eligible_dates[-1] if number == folds - 1 else eligible_dates[initial + (number * 2 + 2) * block - 1]
        val_start, cal_start, test_start = after_gap(train_end), after_gap(selection_end), after_gap(val_end)
        masks = {'train': (origins <= train_end) & (labels <= train_end),
            'validation': (origins >= val_start) & (labels <= selection_end),
            'calibration': (origins >= cal_start) & (labels <= val_end),
            'test': (origins >= test_start) & (origins <= test_end) & (labels <= test_end)}
        indices = {name: endpoints[mask] for name, mask in masks.items()}
        counts = {name: int(frame.iloc[value].date.nunique()) for name, value in indices.items()}
        if any(counts[name] < minimum_sessions[name] for name in indices):
            raise ValueError(f'Fold {number} has insufficient eligible sessions: {counts}; minimum {minimum_sessions}. Do not shrink the gate after test results.')
        boundaries = {'train_end': str(pd.Timestamp(train_end).date()), 'validation_start': str(pd.Timestamp(val_start).date()),
            'validation_end': str(pd.Timestamp(selection_end).date()), 'calibration_start': str(pd.Timestamp(cal_start).date()),
            'calibration_end': str(pd.Timestamp(val_end).date()), 'test_start': str(pd.Timestamp(test_start).date()),
            'test_end': str(pd.Timestamp(test_end).date()), 'gap_sessions': str(gap_sessions),
            'eligible_session_counts': counts, 'boundary_policy': 'ELIGIBLE_ORIGIN_DATES_GAP_ON_SOURCE_CALENDAR'}
        result.append(WalkForwardFold(number, **indices, boundaries=boundaries))
    return result


def origin_scales(frame, floor=.005, ceiling=.10):
    """Known-at-origin volatility only; no target or per-ticker future statistic."""
    vol = frame.volatility_20.to_numpy(dtype=np.float64)
    # Warm-up rows are never endpoints. Fixed floor/ceiling belong to protocol.
    vol = np.clip(np.nan_to_num(vol, nan=floor), floor, ceiling)
    return np.column_stack([vol, vol * np.sqrt(5.)])


def direction_labels(actual):
    values = np.asarray(actual, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Direction labels require finite targets')
    return np.where(values < -1e-12, 0, np.where(values > 1e-12, 2, 1)).astype(np.int32)


def direction_class_weights(train_labels):
    labels = np.asarray(train_labels)
    if not len(labels) or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError('Invalid training direction labels')
    counts = np.bincount(labels, minlength=3)
    return np.clip(len(labels) / (3 * np.maximum(counts, 1)), .5, 3.).astype(np.float32)


def split_calibration(frame, indices):
    indices = np.asarray(indices)
    dates = np.sort(pd.to_datetime(frame.iloc[indices].date).unique())
    boundary = dates[int(len(dates) * .65)]
    rows = frame.iloc[indices]
    fit = indices[(pd.to_datetime(rows.target_end_5) < boundary).to_numpy()]
    interval = indices[(pd.to_datetime(rows.date) >= boundary).to_numpy()]
    if min(frame.iloc[fit].date.nunique(), frame.iloc[interval].date.nunique()) < 10:
        raise ValueError('Need >=10 eligible dates each for purged calibration-fit and interval blocks')
    return fit, interval


def _validate_calibration_arrays(actual, raw, flat_probability):
    arrays = [np.asarray(x, dtype=np.float64) for x in (actual, raw, flat_probability)]
    if arrays[0].ndim != 1 or not arrays[0].size or any(x.shape != arrays[0].shape or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Calibration arrays must be finite, aligned and nonempty')
    if ((arrays[2] < 0) | (arrays[2] > 1)).any():
        raise ValueError('Flat probabilities must be in [0,1]')
    return arrays


def apply_calibration(raw, flat_probability, specification):
    raw, probability = [np.asarray(x, dtype=np.float64) for x in (raw, flat_probability)]
    if raw.shape != probability.shape or not np.isfinite(raw).all() or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Invalid calibrated forecast inputs')
    alpha, bias, threshold = specification['alpha'], specification['bias'], specification['flat_threshold']
    if not np.isfinite([alpha, bias]).all() or not 0 <= alpha <= 1 or (threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1)):
        raise ValueError('Invalid calibration parameters')
    forecast = alpha * raw + bias
    if threshold is not None:
        forecast = np.where(probability >= threshold, 0., forecast)
    return forecast


def fit_calibration(actual, raw, flat_probability, *, allow_flat_gate=True):
    """Fit only on the registered calibration-fit block, never test/interval."""
    actual, raw, probability = _validate_calibration_arrays(actual, raw, flat_probability)
    options = []
    for threshold in ([None, .5, .65, .8] if allow_flat_gate else [None]):
        keep = np.ones(len(actual), dtype=bool) if threshold is None else probability < threshold
        if not keep.any():
            continue
        for alpha in np.linspace(0, 1, 11):
            bias = float(np.median(actual[keep] - alpha * raw[keep]))
            spec = {'alpha': float(alpha), 'bias': bias, 'flat_threshold': threshold}
            prediction = apply_calibration(raw, probability, spec)
            loss = float(np.mean(np.abs(actual - prediction), dtype=np.float64))
            options.append((loss, spec))
    loss, best = min(options, key=lambda x: x[0])
    return {**best, 'fit_mae': loss, 'fit_scope': 'PURGED_CALIBRATION_FIT_ONLY'}


def classification_metrics(actual, probabilities):
    labels = direction_labels(actual)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), 3) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError('Invalid direction probabilities')
    predicted = probabilities.argmax(axis=1)
    recall = [float(np.mean(predicted[labels == label] == label)) if (labels == label).any() else None for label in range(3)]
    return {'three_class_accuracy': float(np.mean(predicted == labels)),
        'balanced_three_class_accuracy': float(np.mean([x for x in recall if x is not None])),
        'per_class_recall_down_flat_up': recall,
        'probability_status': 'UNCALIBRATED_CLASSIFIER_NOT_TRADE_CONFIDENCE'}


def baseline_predictions(frame, fold, horizon, *, include_tree=True):
    """25-feature Ridge/tree, same endpoints and original unfiltered test labels."""
    train = frame.iloc[fold.train]
    scaler = TrainOnlyScaler().fit(train[FEATURES].to_numpy(float))
    x, v, t = [scaler.transform(frame.iloc[index][FEATURES].to_numpy(float)).astype(np.float64) for index in (fold.train, fold.validation, fold.test)]
    y, vy = [frame.iloc[index][f'target_return_{horizon}'].to_numpy(float) for index in (fold.train, fold.validation)]
    values = {'zero': np.zeros(len(t)), 'train_mean': np.full(len(t), y.mean()),
        'train_median': np.full(len(t), np.median(y)), 'persistence': frame.iloc[fold.test][f'return_{horizon}'].to_numpy(float)}
    x, v, t = [np.column_stack([np.ones(len(a)), a]) for a in (x, v, t)]
    best = None
    for alpha in (1., 100., 1000.):
        penalty = np.eye(x.shape[1]) * alpha
        penalty[0, 0] = 0
        coefficient = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        loss = np.mean(np.abs(vy - v @ coefficient))
        if best is None or loss < best[0]:
            best = loss, alpha, coefficient
    values['ridge'] = t @ best[2]
    details = {'ridge_alpha': best[1], 'tree_status': 'DISABLED', 'feature_count': len(FEATURES), 'selection_scope': 'VALIDATION_ONLY'}
    if include_tree:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(max_iter=60, max_leaf_nodes=7, l2_regularization=10,
                learning_rate=.05, early_stopping=False, random_state=42)
            model.fit(x[:, 1:], y)
            values['tree'] = model.predict(t[:, 1:])
            details['tree_status'] = 'EVALUATED'
        except (ImportError, OSError, RuntimeError) as exc:
            details.update(tree_status='UNAVAILABLE', tree_error_type=type(exc).__name__)
    return values, details


def scaler_specification(scaler):
    spec = scaler.specification()
    # The generic V3 scaler's mathematical implementation is reused, not its
    # 17-feature metadata. Pin the actual 25-feature order for future inference.
    spec['features'] = list(FEATURES)
    return spec
