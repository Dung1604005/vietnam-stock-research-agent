"""Kaggle-ready multi-horizon LSTM for an educational agent demo.

Paste this file into one Kaggle code cell after attaching the 1M-row CSV.
The artifact is always exported, but its specification says whether it passed
the holdout gate. A rejected model must not influence investment synthesis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
WINDOW = 30
CLIP_Q = 0.005
MIN_ACTIVE_RATIO_20 = 0.80
MIN_LIQUIDITY_PERCENTILE = 0.35
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

matches = sorted(
    Path("/kaggle/input").rglob("vci_ohlcv_lstm_*.csv"),
    key=lambda item: item.stat().st_size,
    reverse=True,
)
if not matches:
    raise FileNotFoundError("Attach vci_ohlcv_lstm_1m.csv to this Kaggle notebook.")
DATA_PATH = matches[0]
OUTPUT_DIR = Path("/kaggle/working")
print("Dataset:", DATA_PATH)
print("TensorFlow:", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))

raw = pd.read_csv(DATA_PATH, parse_dates=["date", "target_date"])
required = {
    "ticker", "date", "open", "high", "low", "close", "volume",
    "target_date", "next_session_return", "price_adjustment",
}
missing = required.difference(raw.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")
if raw.duplicated(["ticker", "date"]).any():
    raise ValueError("Duplicate ticker/date pairs detected.")
print({"rows": len(raw), "tickers": int(raw.ticker.nunique())})
if not raw["price_adjustment"].eq("ADJUSTED").all():
    print("WARNING: price adjustment is unverified; this is a classroom experiment.")

raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)
group = raw.groupby("ticker", group_keys=False)
previous_close = group["close"].shift(1)
raw["return_1"] = raw["close"] / previous_close - 1
for period in (2, 5, 10, 20):
    raw[f"return_{period}"] = group["close"].pct_change(period, fill_method=None)
raw["open_gap"] = raw["open"] / previous_close - 1
raw["intraday_range"] = raw["high"] / raw["low"] - 1
raw["close_position"] = (
    (raw["close"] - raw["low"]) / (raw["high"] - raw["low"]).replace(0, np.nan)
)
raw["log_volume"] = np.log1p(raw["volume"])
raw["volume_change_1"] = raw.groupby("ticker")["log_volume"].diff()
raw["traded_value"] = raw["close"] * raw["volume"]
raw["log_traded_value"] = np.log1p(raw["traded_value"])
raw["active_ratio_20"] = group["volume"].transform(
    lambda values: values.gt(0).rolling(20, min_periods=20).mean()
)
raw["avg_traded_value_20"] = group["traded_value"].transform(
    lambda values: values.rolling(20, min_periods=20).mean()
)
raw["liquidity_percentile"] = raw.groupby("date")["avg_traded_value_20"].rank(pct=True)
for period in (5, 20):
    rolling = group["volume"].transform(
        lambda values, size=period: values.rolling(size, min_periods=size).mean()
    )
    raw[f"volume_ratio_{period}"] = raw["volume"] / rolling.replace(0, np.nan)
for period in (5, 20, 60):
    rolling = group["close"].transform(
        lambda values, size=period: values.rolling(size, min_periods=size).mean()
    )
    raw[f"ma_distance_{period}"] = raw["close"] / rolling - 1
for period in (5, 20):
    raw[f"volatility_{period}"] = group["return_1"].transform(
        lambda values, size=period: values.rolling(size, min_periods=size).std()
    )

# This is a free, close-time market proxy—not VN-Index.
market = (
    raw.groupby("date", as_index=False)["return_1"].median()
    .rename(columns={"return_1": "cross_section_median_return_1"})
)
raw = raw.merge(market, on="date", how="left", validate="many_to_one")
raw["relative_return_1"] = raw["return_1"] - raw["cross_section_median_return_1"]
raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)

# Auxiliary 5-session target is less noisy and improves the course demo.
group = raw.groupby("ticker", group_keys=False)
raw["target_return_5"] = group["close"].shift(-5) / raw["close"] - 1
raw["target_date_5"] = group["date"].shift(-5)

FEATURES = [
    "return_1", "return_2", "return_5", "return_10", "return_20",
    "open_gap", "intraday_range", "close_position", "log_volume",
    "log_traded_value", "volume_change_1", "volume_ratio_5", "volume_ratio_20",
    "ma_distance_5", "ma_distance_20", "ma_distance_60",
    "volatility_5", "volatility_20", "cross_section_median_return_1",
    "relative_return_1", "active_ratio_20", "liquidity_percentile",
]
TARGETS = ["next_session_return", "target_return_5"]
frame = raw.dropna(subset=[*FEATURES, *TARGETS, "target_date_5"]).copy()
frame = frame[np.isfinite(frame[FEATURES + TARGETS]).all(axis=1)].copy()
frame = frame[
    frame["active_ratio_20"].ge(MIN_ACTIVE_RATIO_20)
    & frame["liquidity_percentile"].ge(MIN_LIQUIDITY_PERCENTILE)
].copy()
if len(frame) < 50_000:
    raise ValueError(f"Only {len(frame)} usable rows remain.")

# Split on the date when the five-session label becomes available.
dates = np.array(sorted(frame["target_date_5"].unique()))
train_cutoff = dates[int(len(dates) * 0.70)]
validation_cutoff = dates[int(len(dates) * 0.85)]
train_mask_rows = frame["target_date_5"] <= train_cutoff
train_rows = frame[train_mask_rows]
ticker_vocabulary = sorted(train_rows["ticker"].unique().tolist())
ticker_to_id = {ticker: index + 1 for index, ticker in enumerate(ticker_vocabulary)}
frame["ticker_id"] = frame["ticker"].map(ticker_to_id).fillna(0).astype(np.int32)

# All preprocessing statistics are fitted on training data only.
feature_low = train_rows[FEATURES].quantile(CLIP_Q)
feature_high = train_rows[FEATURES].quantile(1 - CLIP_Q)
feature_high = feature_high.where(feature_high > feature_low, feature_low + 1.0)
frame.loc[:, FEATURES] = frame[FEATURES].clip(feature_low, feature_high, axis=1)
scaled_train = frame[train_mask_rows]
feature_mean = scaled_train[FEATURES].mean()
feature_std = scaled_train[FEATURES].std().replace(0, 1)
frame.loc[:, FEATURES] = (frame[FEATURES] - feature_mean) / feature_std

target_low = train_rows[TARGETS].quantile(CLIP_Q)
target_high = train_rows[TARGETS].quantile(1 - CLIP_Q)
target_high = target_high.where(target_high > target_low, target_low + 1.0)
fit_targets = frame[TARGETS].clip(target_low, target_high, axis=1)
target_mean = fit_targets.loc[train_mask_rows].mean()
target_std = fit_targets.loc[train_mask_rows].std().replace(0, 1)
normalized_targets = (fit_targets - target_mean) / target_std


def make_sequences(data: pd.DataFrame, normalized: pd.DataFrame):
    groups = [indices.to_numpy() for indices in data.groupby("ticker", sort=False).groups.values()]
    count = sum(max(0, len(indices) - WINDOW + 1) for indices in groups)
    x = np.empty((count, WINDOW, len(FEATURES)), dtype=np.float32)
    y_fit = np.empty((count, len(TARGETS)), dtype=np.float32)
    y_raw = np.empty((count, len(TARGETS)), dtype=np.float32)
    ticker_ids = np.empty(count, dtype=np.int32)
    label_dates = np.empty(count, dtype="datetime64[ns]")
    feature_values = data[FEATURES].to_numpy(dtype=np.float32)
    fit_values = normalized.to_numpy(dtype=np.float32)
    raw_values = data[TARGETS].to_numpy(dtype=np.float32)
    dates_values = data["target_date_5"].to_numpy(dtype="datetime64[ns]")
    ticker_values = data["ticker_id"].to_numpy(dtype=np.int32)
    cursor = 0
    for indices in groups:
        for endpoint in range(WINDOW - 1, len(indices)):
            window_indices = indices[endpoint - WINDOW + 1:endpoint + 1]
            label_index = indices[endpoint]
            x[cursor] = feature_values[window_indices]
            y_fit[cursor] = fit_values[label_index]
            y_raw[cursor] = raw_values[label_index]
            ticker_ids[cursor] = ticker_values[label_index]
            label_dates[cursor] = dates_values[label_index]
            cursor += 1
    return x, ticker_ids, y_fit, y_raw, label_dates


frame = frame.reset_index(drop=True)
normalized_targets = normalized_targets.reset_index(drop=True)
X, ticker_ids, y_fit, y_raw, sequence_dates = make_sequences(frame, normalized_targets)
train_mask = sequence_dates <= train_cutoff
validation_mask = (sequence_dates > train_cutoff) & (sequence_dates <= validation_cutoff)
test_mask = sequence_dates > validation_cutoff
if min(train_mask.sum(), validation_mask.sum(), test_mask.sum()) == 0:
    raise ValueError("A chronological split is empty.")
print({
    "usable_rows": len(frame), "features": len(FEATURES), "window": WINDOW,
    "train": int(train_mask.sum()), "validation": int(validation_mask.sum()),
    "test": int(test_mask.sum()),
})

sequence_input = tf.keras.layers.Input((WINDOW, len(FEATURES)), name="market_sequence")
ticker_input = tf.keras.layers.Input((), dtype="int32", name="ticker_id")
x = tf.keras.layers.Conv1D(32, 3, padding="causal", activation="relu")(sequence_input)
x = tf.keras.layers.LSTM(64, dropout=0.10)(x)
ticker_embedding = tf.keras.layers.Embedding(len(ticker_vocabulary) + 1, 8, name="ticker_embedding")(ticker_input)
ticker_embedding = tf.keras.layers.Flatten()(ticker_embedding)
x = tf.keras.layers.Concatenate()([x, ticker_embedding])
x = tf.keras.layers.Dense(48, activation="relu")(x)
x = tf.keras.layers.Dropout(0.15)(x)
outputs = tf.keras.layers.Dense(2, name="normalized_returns")(x)
model = tf.keras.Model([sequence_input, ticker_input], outputs, name="stock_lstm_multihorizon_v2")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=7.5e-4, clipnorm=1.0),
    loss=tf.keras.losses.MeanAbsoluteError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
)
model.summary()
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=7, min_delta=1e-4, restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1,
    ),
]
history = model.fit(
    [X[train_mask], ticker_ids[train_mask]], y_fit[train_mask],
    validation_data=([X[validation_mask], ticker_ids[validation_mask]], y_fit[validation_mask]),
    epochs=35, batch_size=512, callbacks=callbacks, shuffle=True, verbose=2,
)

validation_normalized_prediction = model.predict(
    [X[validation_mask], ticker_ids[validation_mask]], batch_size=1024, verbose=0,
)
validation_prediction = (
    validation_normalized_prediction * target_std.to_numpy(np.float32)
    + target_mean.to_numpy(np.float32)
)
normalized_prediction = model.predict(
    [X[test_mask], ticker_ids[test_mask]], batch_size=1024, verbose=0,
)
raw_prediction = normalized_prediction * target_std.to_numpy(np.float32) + target_mean.to_numpy(np.float32)
actual = y_raw[test_mask]
naive = np.zeros_like(actual)


def fit_shrinkage(observed, predicted):
    """Select forecast shrinkage and bias using validation only."""
    best = None
    for alpha in np.linspace(0.0, 1.0, 21):
        bias = float(np.median(observed - alpha * predicted))
        calibrated = alpha * predicted + bias
        score = float(np.mean(np.abs(observed - calibrated)))
        candidate = (score, float(alpha), bias)
        if best is None or candidate < best:
            best = candidate
    return {"validation_mae": best[0], "alpha": best[1], "bias": best[2]}


calibration = [
    fit_shrinkage(y_raw[validation_mask, index], validation_prediction[:, index])
    for index in range(len(TARGETS))
]
prediction = np.column_stack([
    calibration[index]["alpha"] * raw_prediction[:, index] + calibration[index]["bias"]
    for index in range(len(TARGETS))
])


def evaluate(observed, predicted, baseline):
    model_mae = float(np.mean(np.abs(observed - predicted)))
    baseline_mae = float(np.mean(np.abs(observed - baseline)))
    return {
        "mae": model_mae,
        "naive_mae": baseline_mae,
        "mae_relative_improvement": float((baseline_mae - model_mae) / baseline_mae),
        "rmse": float(np.sqrt(np.mean((observed - predicted) ** 2))),
        "naive_rmse": float(np.sqrt(np.mean((observed - baseline) ** 2))),
        "directional_accuracy": float(np.mean(np.sign(predicted) == np.sign(observed))),
        "beats_naive_mae": bool(model_mae < baseline_mae),
    }


metrics = {
    "next_session": evaluate(actual[:, 0], prediction[:, 0], naive[:, 0]),
    "five_session": evaluate(actual[:, 1], prediction[:, 1], naive[:, 1]),
    "test_samples": int(test_mask.sum()),
    "train_cutoff": str(pd.Timestamp(train_cutoff).date()),
    "validation_cutoff": str(pd.Timestamp(validation_cutoff).date()),
    "validation_calibration": {
        TARGETS[index]: calibration[index] for index in range(len(TARGETS))
    },
}
for index, horizon in enumerate(("next_session", "five_session")):
    metrics[horizon]["calibration_alpha"] = calibration[index]["alpha"]
    metrics[horizon]["uses_lstm_signal"] = calibration[index]["alpha"] > 0
passed_horizons = [
    name for name in ("next_session", "five_session")
    if metrics[name]["beats_naive_mae"] and metrics[name]["uses_lstm_signal"]
]
eligible = bool(passed_horizons)
status = "RESEARCH_CANDIDATE" if eligible else "EXPERIMENTAL_REJECTED"
print(json.dumps(metrics, indent=2))
print("Artifact status:", status, "| Passed horizons:", passed_horizons)

# Always export for the course demo; the status controls later app behaviour.
model_path = OUTPUT_DIR / "stock_lstm_multihorizon_demo.keras"
spec_path = OUTPUT_DIR / "stock_lstm_multihorizon_demo.spec.json"
metrics_path = OUTPUT_DIR / "stock_lstm_multihorizon_demo.metrics.json"
history_path = OUTPUT_DIR / "stock_lstm_multihorizon_demo.history.csv"
model.save(model_path)
pd.DataFrame(history.history).to_csv(history_path, index=False)
spec = {
    "schema_version": "stock-lstm-artifact-v2",
    "model_version": "lstm-multihorizon-v2",
    "artifact_status": status,
    "integration_eligible": eligible,
    "passed_horizons": passed_horizons,
    "window": WINDOW,
    "features": FEATURES,
    "ticker_vocabulary": ticker_vocabulary,
    "feature_clip_lower": feature_low.to_dict(),
    "feature_clip_upper": feature_high.to_dict(),
    "feature_mean": feature_mean.to_dict(),
    "feature_std": feature_std.to_dict(),
    "targets": TARGETS,
    "target_clip_lower": target_low.to_dict(),
    "target_clip_upper": target_high.to_dict(),
    "target_mean": target_mean.to_dict(),
    "target_std": target_std.to_dict(),
    "metrics": metrics,
    "eligibility_scope": {
        "minimum_active_ratio_20": MIN_ACTIVE_RATIO_20,
        "minimum_liquidity_percentile": MIN_LIQUIDITY_PERCENTILE,
    },
    "data_warnings": [
        "VCI price-adjustment status is UNKNOWN.",
        "The current listing catalogue has survivorship bias.",
        "Educational demo only; never a trade instruction.",
    ],
}
spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("Saved:", model_path, spec_path, metrics_path, history_path, sep="\n- ")
