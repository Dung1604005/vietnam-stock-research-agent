"""Optional V3 shadow adapter; loading never changes the frozen V2 default."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_multi_agent.models.analysis import MarketData
from stock_multi_agent.models.forecast import ForecastStatus, PriceForecast
from stock_multi_agent.services.ml_features_v3 import FEATURES, build_feature_frame, window_endpoints
from stock_multi_agent.services.ml_model_registry import evaluate_model_gate, verify_artifact_directory


class LstmV3ArtifactError(RuntimeError):
    pass


class LstmV3ForecastModel:
    def __init__(self, artifact_dir: Path, keras_model: Any | None = None) -> None:
        self.artifact_dir = Path(artifact_dir).resolve()
        try:
            self.manifest, self.spec, self.report, self.protocol = verify_artifact_directory(self.artifact_dir)
            self.window = int(self.spec["window"])
            self.version = self.spec["model_version"]
            scaler = self.spec["scaler"]
            self.mean, self.std, self.lower, self.upper = [np.asarray(scaler[key], dtype=np.float32) for key in ("mean", "std", "lower", "upper")]
            if scaler.get("fit_scope") != "TRAIN_ONLY" or scaler.get("features") != FEATURES:
                raise ValueError("Unverified scaler fit scope/order")
            if any(array.shape != (len(FEATURES),) or not np.isfinite(array).all() for array in (self.mean, self.std, self.lower, self.upper)) or (self.std <= 0).any() or (self.lower > self.upper).any():
                raise ValueError("Invalid scaler values")
            self.gate = evaluate_model_gate(self.report, self.protocol, horizon=1)
            self._model = keras_model if keras_model is not None else self._load_model()
        except Exception as exc:
            raise LstmV3ArtifactError(f"V3 artifact rejected: {type(exc).__name__}: {exc}") from exc

    def _load_model(self) -> Any:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise LstmV3ArtifactError("TensorFlow is optional and not installed; V3 unavailable") from exc
        return tf.keras.models.load_model(self.artifact_dir / "model.keras", compile=False, safe_mode=True)

    def predict_next_session(self, market_data: MarketData) -> PriceForecast:
        rows = [{"ticker": market_data.ticker, "date": bar.date, "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close, "volume": bar.volume} for bar in market_data.bars]
        frame = build_feature_frame(pd.DataFrame(rows), max_gap_business_days=int(self.spec.get("max_gap_business_days", 5)))
        endpoints = window_endpoints(frame, self.window, require_targets=False)
        if not len(endpoints) or endpoints[-1] != len(frame) - 1:
            raise LstmV3ArtifactError("Latest bar lacks a contiguous eligible V3 feature window")
        values = frame.iloc[-self.window:][FEATURES].to_numpy(dtype=np.float32)
        transformed = (np.clip(values, self.lower, self.upper) - self.mean) / self.std
        output = np.asarray(self._model.predict(transformed[None, :, :], verbose=0), dtype=float)
        if output.shape != (1, 2) or not np.isfinite(output).all():
            raise LstmV3ArtifactError("Invalid V3 model output shape/value")
        calibration = self.spec["calibration"]["1"]
        if any(not isinstance(calibration.get(key), (int, float)) or not np.isfinite(calibration[key]) for key in ("alpha", "bias")) or not 0 <= calibration["alpha"] <= 1:
            raise LstmV3ArtifactError("Invalid V3 calibration alpha/bias")
        target_scale = self.spec.get("target_scale", [1.0, 1.0])
        if len(target_scale) != 2 or not np.isfinite(target_scale).all() or min(target_scale) <= 0:
            raise LstmV3ArtifactError("Invalid V3 target scale")
        predicted_return = float(output[0, 0]) * float(target_scale[0]) * float(calibration["alpha"]) + float(calibration["bias"])
        if not np.isfinite(predicted_return) or predicted_return <= -1:
            raise LstmV3ArtifactError("Invalid V3 calibrated return")
        current = market_data.bars[-1].close
        forecast = current * (1 + predicted_return)
        adjustment = getattr(market_data.price_adjustment, "value", str(market_data.price_adjustment))
        input_ok = adjustment == "ADJUSTED" and self.spec.get("price_adjustment") == "ADJUSTED"
        input_ok &= str(getattr(market_data.price_unit, "value", market_data.price_unit)) == self.spec.get("price_unit")
        eligible = self.gate.policy_eligibility and bool(input_ok) and calibration["alpha"] > 0
        warnings = list(self.gate.reasons)
        if not input_ok:
            warnings.append("Runtime adjustment/price unit does not meet the V3 blend contract; shadow only")
        if calibration["alpha"] == 0:
            warnings.append("Calibration selected a constant baseline (alpha=0); no LSTM Technical weight")
        # Calibration uses its own block; coverage is evaluated, not guaranteed.
        radius = calibration.get("absolute_residual_quantile_90")
        if radius is not None and (not np.isfinite(radius) or radius < 0):
            raise LstmV3ArtifactError("Invalid V3 calibration interval")
        payload = {"ticker": market_data.ticker, "as_of": market_data.bars[-1].date, "current_price": current, "forecast_price": forecast, "lower_bound": max(0.000001, forecast - current * radius) if radius is not None else None, "upper_bound": forecast + current * radius if radius is not None else None, "horizon_sessions": 1, "status": ForecastStatus.TRAINED_MODEL, "method": "Shared-feature V3 LSTM; disjoint validation calibration; policy-gated", "model_version": self.version, "confidence": 0.1, "warnings": warnings + ["One-session research estimate, not a three-month target or trade instruction", "Calibration-block interval does not guarantee future coverage"], "artifact_status": self.gate.artifact_status, "validation_status": self.gate.validation_status, "policy_eligibility": eligible, "technical_blend_weight": self.gate.technical_blend_weight if eligible else 0.0}
        return PriceForecast(**payload)
