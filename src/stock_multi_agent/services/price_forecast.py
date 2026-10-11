from __future__ import annotations

from math import sqrt
import os
from pathlib import Path
from statistics import fmean, pstdev
from typing import Protocol

from stock_multi_agent.models.analysis import MarketData
from stock_multi_agent.models.data import PriceAdjustment
from stock_multi_agent.models.forecast import ForecastStatus, PriceForecast


class PriceForecastModel(Protocol):
    """Future adapter boundary for a trained ML/deep-learning artifact."""

    version: str

    def predict_next_session(self, market_data: MarketData) -> PriceForecast: ...


class PriceForecastService:
    """Serves a trained model when supplied, otherwise an explicitly untrained baseline."""

    BASELINE_VERSION = "rolling-mean-return-baseline-v1"

    def __init__(
        self,
        trained_model: PriceForecastModel | None = None,
        load_warning: str | None = None,
    ) -> None:
        self.trained_model = trained_model
        self.load_warning = load_warning

    @classmethod
    def from_environment(cls, default_enabled: bool = False) -> "PriceForecastService":
        """Load the finalized V2 artifact, with a safe baseline on any failure."""
        configured = os.getenv("STOCK_AGENT_LSTM_ENABLED")
        enabled = default_enabled if configured is None else configured.strip().lower() not in {
            "0", "false", "off", "no",
        }
        if not enabled:
            return cls()
        from stock_multi_agent.services.lstm_forecast import (
            LstmArtifactError,
            LstmV2ForecastModel,
        )

        configured_path = os.getenv("STOCK_AGENT_LSTM_ARTIFACT_DIR", "").strip()
        artifact_dir = (
            Path(configured_path).expanduser()
            if configured_path
            else LstmV2ForecastModel.default_artifact_dir()
        )
        try:
            return cls(trained_model=LstmV2ForecastModel(artifact_dir))
        except LstmArtifactError as exc:
            return cls(load_warning=f"Finalized LSTM V2 unavailable: {exc}")

    def forecast_next_session(self, market_data: MarketData) -> PriceForecast:
        if self.trained_model is not None:
            try:
                forecast = self.trained_model.predict_next_session(market_data)
            except Exception as exc:
                return self._untrained_baseline(
                    market_data,
                    extra_warning=f"LSTM V2 inference unavailable; baseline used instead: {exc}",
                )
            if forecast.ticker != market_data.ticker or forecast.as_of != market_data.bars[-1].date:
                raise ValueError("Trained forecast model returned a ticker or as-of date inconsistent with market data")
            if forecast.status is not ForecastStatus.TRAINED_MODEL:
                raise ValueError("Trained forecast model must label its result TRAINED_MODEL")
            return forecast
        return self._untrained_baseline(market_data)

    def _untrained_baseline(
        self,
        market_data: MarketData,
        extra_warning: str | None = None,
    ) -> PriceForecast:
        closes = [bar.close for bar in market_data.bars]
        latest = closes[-1]
        returns = [closes[index] / closes[index - 1] - 1 for index in range(1, len(closes))]
        window = returns[-20:]
        mean_return = fmean(window) if window else 0.0
        volatility = pstdev(window) if len(window) > 1 else 0.0
        estimate = max(0.01, latest * (1 + mean_return))
        # A broad 80%-style diagnostic interval. It is a baseline uncertainty
        # display, not a calibrated prediction interval.
        spread = 1.28 * volatility * sqrt(1)
        lower = max(0.01, latest * (1 + mean_return - spread))
        upper = max(lower, latest * (1 + mean_return + spread))
        warnings = [
            "No trained forecasting model is installed; this rolling-return baseline is not a validated price prediction.",
            "Do not use this estimate as a trade instruction or performance claim.",
        ]
        if self.load_warning:
            warnings.insert(0, self.load_warning)
        if extra_warning:
            warnings.insert(0, extra_warning)
        if market_data.price_adjustment is not PriceAdjustment.ADJUSTED:
            warnings.append("Price-adjustment status is not verified; corporate actions can distort this baseline.")
        return PriceForecast(
            ticker=market_data.ticker,
            as_of=market_data.bars[-1].date,
            current_price=latest,
            forecast_price=estimate,
            lower_bound=lower,
            upper_bound=upper,
            status=ForecastStatus.BASELINE_UNTRAINED,
            method="20-session rolling mean return with realized-volatility range",
            model_version=self.BASELINE_VERSION,
            confidence=0.0,
            warnings=warnings,
        )
