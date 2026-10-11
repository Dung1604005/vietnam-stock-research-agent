"""Paste this file's Python code into a Kaggle Notebook code cell, then Run All.

Do not paste train_lstm_kaggle.ipynb's JSON into a code cell: that file is a
notebook container, whereas this file is executable Python.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


SEED = 42
WINDOW = 30
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

matches = sorted(Path("/kaggle/input").rglob("vci_ohlcv_lstm_*.csv"), key=lambda item: item.stat().st_size, reverse=True)
if not matches:
    raise FileNotFoundError(
        "Attach the uploaded vci_ohlcv_lstm_1m.csv Kaggle Dataset to this notebook."
    )
DATA_PATH = matches[0]
OUTPUT_DIR = Path("/kaggle/working")
print("Dataset:", DATA_PATH)
print("TensorFlow:", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))

raw = pd.read_csv(DATA_PATH, parse_dates=["date", "target_date"])
required = {
    "ticker", "date", "open", "high", "low", "close", "volume",
    "target_date", "next_session_return",
}
missing = required.difference(raw.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")
if raw.duplicated(["ticker", "date"]).any():
    raise ValueError("Duplicate ticker/date pairs: regenerate or deduplicate before training.")
if (raw.get("price_adjustment") != "ADJUSTED").any():
    print("WARNING: adjusted-price status is not verified. Do not use results as a performance claim.")

raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)
group = raw.groupby("ticker", group_keys=False)
previous_close = group["close"].shift(1)
raw["return_1"] = raw["close"] / previous_close - 1
raw["return_5"] = group["close"].pct_change(5)
raw["return_20"] = group["close"].pct_change(20)
raw["open_gap"] = raw["open"] / previous_close - 1
raw["intraday_range"] = raw["high"] / raw["low"] - 1
raw["close_position"] = (raw["close"] - raw["low"]) / (raw["high"] - raw["low"]).replace(0, np.nan)
raw["log_volume"] = np.log1p(raw["volume"])
raw["volume_ratio_20"] = group["volume"].transform(
    lambda value: value / value.rolling(20, min_periods=5).mean()
)
raw["volatility_20"] = group["return_1"].transform(
    lambda value: value.rolling(20, min_periods=20).std()
)

FEATURES = [
    "return_1", "return_5", "return_20", "open_gap", "intraday_range",
    "close_position", "log_volume", "volume_ratio_20", "volatility_20",
]
frame = raw.dropna(subset=[*FEATURES, "next_session_return"]).copy()
frame = frame[np.isfinite(frame[FEATURES + ["next_session_return"]]).all(axis=1)].copy()
if len(frame) < 5_000:
    raise ValueError(f"Only {len(frame)} usable rows remain; dataset is too small for this LSTM experiment.")
print({"raw_rows": len(raw), "usable_rows": len(frame), "tickers": frame.ticker.nunique(), "features": len(FEATURES)})

# Split by target availability time; never randomly shuffle stock time series.
dates = np.array(sorted(frame["target_date"].unique()))
train_cutoff = dates[int(len(dates) * 0.70)]
validation_cutoff = dates[int(len(dates) * 0.85)]
train_rows = frame[frame["target_date"] <= train_cutoff]
mean = train_rows[FEATURES].mean().to_numpy(dtype=np.float32)
std = train_rows[FEATURES].std().replace(0, 1).to_numpy(dtype=np.float32)
frame.loc[:, FEATURES] = (frame[FEATURES].to_numpy(dtype=np.float32) - mean) / std


def make_sequences(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences, labels, label_dates = [], [], []
    for _, ticker_frame in data.groupby("ticker", sort=False):
        values = ticker_frame[FEATURES].to_numpy(dtype=np.float32)
        targets = ticker_frame["next_session_return"].to_numpy(dtype=np.float32)
        target_dates = ticker_frame["target_date"].to_numpy()
        for endpoint in range(WINDOW - 1, len(ticker_frame)):
            sequences.append(values[endpoint - WINDOW + 1:endpoint + 1])
            labels.append(targets[endpoint])
            label_dates.append(target_dates[endpoint])
    return np.asarray(sequences), np.asarray(labels), np.asarray(label_dates)


X, y, sequence_dates = make_sequences(frame)
train_mask = sequence_dates <= train_cutoff
validation_mask = (sequence_dates > train_cutoff) & (sequence_dates <= validation_cutoff)
test_mask = sequence_dates > validation_cutoff
if min(train_mask.sum(), validation_mask.sum(), test_mask.sum()) == 0:
    raise ValueError("A chronological split is empty; collect a longer data history.")
print({"train": int(train_mask.sum()), "validation": int(validation_mask.sum()), "test": int(test_mask.sum()), "window": WINDOW})

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW, len(FEATURES))),
    tf.keras.layers.LSTM(64, dropout=0.15),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.10),
    tf.keras.layers.Dense(1),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.Huber(),
    metrics=["mae"],
)
history = model.fit(
    X[train_mask], y[train_mask],
    validation_data=(X[validation_mask], y[validation_mask]),
    epochs=40,
    batch_size=256,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)],
    verbose=2,
)

prediction = model.predict(X[test_mask], verbose=0).reshape(-1)
actual = y[test_mask]
naive = np.zeros_like(actual)  # predicted next-session return = 0
mae = lambda observed, predicted: float(np.mean(np.abs(observed - predicted)))
rmse = lambda observed, predicted: float(np.sqrt(np.mean((observed - predicted) ** 2)))
metrics = {
    "test_mae": mae(actual, prediction),
    "test_rmse": rmse(actual, prediction),
    "naive_test_mae": mae(actual, naive),
    "naive_test_rmse": rmse(actual, naive),
    "directional_accuracy": float(np.mean(np.sign(prediction) == np.sign(actual))),
    "beats_naive_mae": bool(mae(actual, prediction) < mae(actual, naive)),
    "train_cutoff": str(pd.Timestamp(train_cutoff).date()),
    "validation_cutoff": str(pd.Timestamp(validation_cutoff).date()),
    "test_samples": int(test_mask.sum()),
}
print(json.dumps(metrics, indent=2))

if metrics["beats_naive_mae"]:
    model_path = OUTPUT_DIR / "stock_lstm_next_return.keras"
    model.save(model_path)
    specification = {
        "model_version": "lstm-next-session-return-v1",
        "window": WINDOW,
        "features": FEATURES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "target": "next_session_return",
        "metrics": metrics,
        "integration_gate": "passed_naive_mae_holdout",
    }
    (OUTPUT_DIR / "stock_lstm_next_return.spec.json").write_text(json.dumps(specification, indent=2))
    print("Artifact approved for review:", model_path)
else:
    print("Artifact REJECTED: LSTM did not beat the naive baseline on held-out data. Do not integrate it into the agent.")
