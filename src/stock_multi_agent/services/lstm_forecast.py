from __future__ import annotations

import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_multi_agent.models.analysis import MarketData
from stock_multi_agent.models.data import PriceAdjustment
from stock_multi_agent.models.forecast import ForecastStatus, PriceForecast


class LstmArtifactError(RuntimeError):
    """The selected artifact cannot safely produce a runtime forecast."""


class LstmV2ForecastModel:
    """Deterministic adapter for the frozen Kaggle V2 Keras artifact.

    The training pipeline used two cross-sectional inputs that are unavailable
    in the single-ticker runtime contract. Their train means are used so their
    standardized values are zero. This limitation is always disclosed.
    """

    MODEL_FILENAME = "stock_lstm_multihorizon_demo.keras"
    SPEC_FILENAME = "stock_lstm_multihorizon_demo.spec.json"
    METRICS_FILENAME = "stock_lstm_multihorizon_demo.metrics.json"
    MANIFEST_FILENAME = "manifest.json"

    def __init__(self, artifact_dir: Path, keras_model: Any | None = None) -> None:
        self.artifact_dir = artifact_dir.resolve()
        self.spec = self._read_json(self.artifact_dir / self.SPEC_FILENAME)
        self.metrics = self._read_json(self.artifact_dir / self.METRICS_FILENAME)
        self.manifest = self._read_json(self.artifact_dir / self.MANIFEST_FILENAME)
        self._validate_artifact_files()
        self._validate_spec()
        self.version = str(self.spec["model_version"])
        self.window = int(self.spec["window"])
        self.features = list(self.spec["features"])
        self._ticker_to_id = {
            ticker: index + 1
            for index, ticker in enumerate(self.spec["ticker_vocabulary"])
        }
        self._model = keras_model if keras_model is not None else self._load_keras_model()

    @classmethod
    def default_artifact_dir(cls) -> Path:
        return Path(__file__).resolve().parents[3] / "notebooks" / "trainResult"

    def predict_next_session(self, market_data: MarketData) -> PriceForecast:
        latest = market_data.bars[-1]
        feature_matrix = self._build_feature_matrix(market_data)
        ticker_id = self._ticker_to_id.get(market_data.ticker.upper(), 0)
        try:
            normalized = np.asarray(
                self._model.predict(
                    [feature_matrix[None, :, :], np.asarray([ticker_id], dtype=np.int32)],
                    verbose=0,
                ),
                dtype=np.float32,
            )
        except Exception as exc:  # TensorFlow errors are normalized at this boundary.
            raise LstmArtifactError(f"Keras inference failed: {exc}") from exc
        if normalized.shape != (1, 2) or not np.isfinite(normalized).all():
            raise LstmArtifactError(
                f"Unexpected model output shape/value: {normalized.shape}."
            )

        target_names = list(self.spec["targets"])
        target_mean = np.asarray(
            [self.spec["target_mean"][name] for name in target_names], dtype=np.float32
        )
        target_std = np.asarray(
            [self.spec["target_std"][name] for name in target_names], dtype=np.float32
        )
        raw_prediction = normalized[0] * target_std + target_mean
        calibration = self.metrics["validation_calibration"]["next_session_return"]
        predicted_return = (
            float(calibration["alpha"]) * float(raw_prediction[0])
            + float(calibration["bias"])
        )
        forecast_price = max(0.01, latest.close * (1.0 + predicted_return))

        # Diagnostic dispersion only: the training run did not calibrate a
        # prediction interval, so this must not be presented as coverage.
        rmse = float(self.metrics["next_session"]["rmse"])
        spread = 1.28 * rmse
        lower = max(0.01, latest.close * (1.0 + predicted_return - spread))
        upper = max(forecast_price, latest.close * (1.0 + predicted_return + spread))
        warnings = [
            "LSTM V2 is a classroom research candidate; its next-session holdout edge over zero return was only 0.0842% by MAE.",
            "Cross-sectional market return and liquidity percentile are unavailable in single-ticker inference and use frozen train-mean proxies.",
            "The displayed range is RMSE-based diagnostic dispersion, not a calibrated prediction interval.",
            "This forecast contributes at most 20% to the Technical Agent score and cannot independently determine BUY/SELL/HOLD.",
        ]
        if ticker_id == 0:
            warnings.append(
                f"{market_data.ticker} was outside the V2 training vocabulary; unknown ticker embedding id 0 was used."
            )
        if market_data.price_adjustment is not PriceAdjustment.ADJUSTED:
            warnings.append(
                "Runtime price-adjustment status is not verified; corporate actions may distort features."
            )
        return PriceForecast(
            ticker=market_data.ticker,
            as_of=latest.date,
            current_price=latest.close,
            horizon_sessions=1,
            forecast_price=forecast_price,
            lower_bound=lower,
            upper_bound=upper,
            status=ForecastStatus.TRAINED_MODEL,
            method="Frozen LSTM V2 next-session return with validation-only shrinkage",
            model_version=self.version,
            # Deliberately low: the artifact's direction accuracy is below 50%
            # and its MAE advantage is economically tiny.
            confidence=0.10,
            warnings=warnings,
        )

    def _build_feature_matrix(self, market_data: MarketData) -> np.ndarray:
        bars = market_data.bars
        minimum_bars = 60 + self.window - 1
        if len(bars) < minimum_bars:
            raise LstmArtifactError(
                f"LSTM V2 needs at least {minimum_bars} consecutive bars; received {len(bars)}."
            )
        frame = pd.DataFrame(
            {
                "open": [bar.open for bar in bars],
                "high": [bar.high for bar in bars],
                "low": [bar.low for bar in bars],
                "close": [bar.close for bar in bars],
                "volume": [bar.volume for bar in bars],
            },
            dtype=np.float64,
        )
        previous_close = frame["close"].shift(1)
        frame["return_1"] = frame["close"].pct_change(1, fill_method=None)
        for period in (2, 5, 10, 20):
            frame[f"return_{period}"] = frame["close"].pct_change(
                period, fill_method=None
            )
        frame["open_gap"] = frame["open"] / previous_close - 1.0
        frame["intraday_range"] = frame["high"] / frame["low"] - 1.0
        frame["close_position"] = (
            (frame["close"] - frame["low"])
            / (frame["high"] - frame["low"]).replace(0, np.nan)
        )
        frame["log_volume"] = np.log1p(frame["volume"])
        frame["volume_change_1"] = frame["log_volume"].diff()
        frame["traded_value"] = frame["close"] * frame["volume"]
        frame["log_traded_value"] = np.log1p(frame["traded_value"])
        frame["active_ratio_20"] = frame["volume"].gt(0).rolling(20).mean()
        for period in (5, 20):
            rolling_volume = frame["volume"].rolling(period).mean()
            frame[f"volume_ratio_{period}"] = frame["volume"] / rolling_volume.replace(
                0, np.nan
            )
        for period in (5, 20, 60):
            moving_average = frame["close"].rolling(period).mean()
            frame[f"ma_distance_{period}"] = frame["close"] / moving_average - 1.0
        for period in (5, 20):
            frame[f"volatility_{period}"] = frame["return_1"].rolling(period).std()

        feature_mean = self.spec["feature_mean"]
        market_proxy = float(feature_mean["cross_section_median_return_1"])
        frame["cross_section_median_return_1"] = market_proxy
        frame["relative_return_1"] = frame["return_1"] - market_proxy
        # A cross-sectional percentile cannot be derived from one ticker. Its
        # train mean maps to exactly zero after standardization.
        frame["liquidity_percentile"] = float(feature_mean["liquidity_percentile"])

        window = frame.iloc[-self.window :].copy()
        if window[self.features].isna().any(axis=None):
            raise LstmArtifactError("Insufficient finite history to build all V2 features.")
        lower = pd.Series(self.spec["feature_clip_lower"])
        upper = pd.Series(self.spec["feature_clip_upper"])
        mean = pd.Series(feature_mean)
        std = pd.Series(self.spec["feature_std"]).replace(0, 1)
        scaled = (window[self.features].clip(lower, upper, axis=1) - mean) / std
        values = scaled.to_numpy(dtype=np.float32)
        if values.shape != (self.window, len(self.features)) or not np.isfinite(values).all():
            raise LstmArtifactError(f"Invalid V2 feature matrix: {values.shape}.")
        return values

    def _validate_artifact_files(self) -> None:
        if self.manifest.get("selected_model") != "lstm-multihorizon-v2":
            raise LstmArtifactError("Manifest does not select LSTM V2.")
        declared = self.manifest.get("files", {})
        for filename in (self.MODEL_FILENAME, self.SPEC_FILENAME, self.METRICS_FILENAME):
            path = self.artifact_dir / filename
            if not path.is_file():
                raise LstmArtifactError(f"Missing finalized artifact: {path}")
            expected = declared.get(filename)
            observed = self._sha256(path)
            if not expected or observed != expected:
                raise LstmArtifactError(f"Checksum mismatch for finalized artifact: {filename}")

    def _validate_spec(self) -> None:
        if self.spec.get("schema_version") != "stock-lstm-artifact-v2":
            raise LstmArtifactError("Unsupported LSTM artifact schema.")
        if self.spec.get("model_version") != "lstm-multihorizon-v2":
            raise LstmArtifactError("Unexpected LSTM model version.")
        if not self.spec.get("integration_eligible"):
            raise LstmArtifactError("Artifact is not marked eligible by its training pipeline.")
        if "next_session" not in self.spec.get("passed_horizons", []):
            raise LstmArtifactError("V2 next-session horizon did not pass its declared gate.")
        if self.spec.get("metrics") != self.metrics:
            raise LstmArtifactError("Spec and standalone metrics JSON disagree.")
        if len(self.features if hasattr(self, "features") else self.spec.get("features", [])) != 22:
            raise LstmArtifactError("V2 must contain exactly 22 ordered features.")

    def _load_keras_model(self):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise LstmArtifactError(
                "TensorFlow is not installed in the workspace virtual environment. "
                "Install the project 'ml' extra on drive D."
            ) from exc
        try:
            return tf.keras.models.load_model(
                self.artifact_dir / self.MODEL_FILENAME, compile=False
            )
        except Exception as exc:
            raise LstmArtifactError(f"Cannot load finalized Keras model: {exc}") from exc

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise LstmArtifactError(f"Cannot read artifact metadata {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise LstmArtifactError(f"Artifact metadata must be an object: {path}")
        return value

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
