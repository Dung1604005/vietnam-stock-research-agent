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
        shadow_model: PriceForecastModel | None = None,
    ) -> None:
        self.trained_model = trained_model
        self.load_warning = load_warning
        self.shadow_model = shadow_model

    @classmethod
    def from_environment(cls, default_enabled: bool = False, *, model_version: str | None = None,
                         experimental_blend: bool = False, defer_load: bool = False) -> "PriceForecastService":
        """Load the selected local artifact; V4 demo blend requires explicit opt-in.

        UI selection is session-local, never a process-wide environment change.
        Missing/invalid models use an unscored, clearly labelled baseline.
        """
        if defer_load:
            return DeferredPriceForecastService(default_enabled=default_enabled, model_version=model_version,
                                                 experimental_blend=experimental_blend)
        configured = os.getenv("STOCK_AGENT_LSTM_ENABLED")
        enabled = default_enabled if configured is None else configured.strip().lower() not in {
            "0", "false", "off", "no",
        }
        if not enabled:
            return cls()
        selected = (model_version or os.getenv('STOCK_AGENT_FORECAST_MODEL_VERSION', 'v2')).strip().lower()
        if selected == 'v4':
            from stock_multi_agent.services.lstm_forecast_v4 import LstmV4ForecastModel
            try:
                configured_archive = os.getenv('STOCK_AGENT_LSTM_V4_ARCHIVE', '').strip()
                archive = Path(configured_archive) if configured_archive else LstmV4ForecastModel.default_archive()
                return cls(trained_model=LstmV4ForecastModel(archive, experimental_blend=experimental_blend))
            except (RuntimeError, ValueError, OSError) as exc:
                return cls(load_warning=f'V4 không khả dụng ({exc}); baseline không tham gia điểm Technical.')
        if selected == 'v3':
            from stock_multi_agent.services.lstm_forecast_v3 import LstmV3ForecastModel
            directory = os.getenv('STOCK_AGENT_LSTM_V3_ARTIFACT_DIR', '').strip()
            if not directory:
                return cls(load_warning='V3 artifact directory is not configured; baseline used. V2 is unchanged.')
            try:
                return cls(trained_model=LstmV3ForecastModel(Path(directory)))
            except (RuntimeError, ValueError, OSError) as exc:
                return cls(load_warning=f'V3 artifact unavailable ({type(exc).__name__}); baseline used.')
        if selected != 'v2':
            return cls(load_warning='Unsupported forecast model version; baseline used.')
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
            service = cls(trained_model=LstmV2ForecastModel(artifact_dir))
            shadow_dir = os.getenv('STOCK_AGENT_LSTM_V3_SHADOW_DIR', '').strip()
            if shadow_dir:
                try:
                    from stock_multi_agent.services.lstm_forecast_v3 import LstmV3ForecastModel
                    service.shadow_model = LstmV3ForecastModel(Path(shadow_dir))
                except (RuntimeError, ValueError, OSError):
                    service.load_warning = 'V3 shadow unavailable; selected V2 forecast is unchanged.'
            return service
        except LstmArtifactError as exc:
            return cls(load_warning=f"Finalized LSTM V2 unavailable: {exc}")

    def forecast_next_session(self, market_data: MarketData) -> PriceForecast:
        if self.trained_model is not None:
            try:
                forecast = self.trained_model.predict_next_session(market_data)
            except Exception as exc:
                return self._untrained_baseline(
                    market_data,
                    extra_warning=f"Selected LSTM inference unavailable ({type(exc).__name__}); baseline used instead.",
                )
            if forecast.ticker != market_data.ticker or forecast.as_of != market_data.bars[-1].date:
                raise ValueError("Trained forecast model returned a ticker or as-of date inconsistent with market data")
            if forecast.status is not ForecastStatus.TRAINED_MODEL:
                raise ValueError("Trained forecast model must label its result TRAINED_MODEL")
            if self.shadow_model is not None:
                try:
                    shadow = self.shadow_model.predict_next_session(market_data)
                    if shadow.ticker != forecast.ticker or shadow.as_of != forecast.as_of:
                        raise ValueError('Shadow ticker/as-of mismatch')
                    forecast = forecast.model_copy(update={'shadow_forecast': {
                        'model_version': shadow.model_version, 'as_of': str(shadow.as_of),
                        'forecast_price': shadow.forecast_price, 'validation_status': shadow.validation_status,
                        'technical_blend_weight': 0, 'mode': 'SHADOW_NOT_SCORED'}})
                except Exception as exc:
                    forecast = forecast.model_copy(update={'shadow_forecast': {
                        'mode': 'SHADOW_UNAVAILABLE', 'error_type': type(exc).__name__, 'technical_blend_weight': 0}})
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


class DeferredPriceForecastService:
    """Keep model imports off the UI startup path; load only for a forecast."""

    def __init__(self, **configuration):
        from threading import RLock
        self.configuration = configuration
        self._service = None
        self._lock = RLock()

    @property
    def load_warning(self):
        return self._service.load_warning if self._service else None

    @property
    def trained_model(self):
        return self._service.trained_model if self._service else None

    def forecast_next_session(self, market_data):
        from stock_multi_agent.services.execution_profiles import current_profile
        from stock_multi_agent.services.request_telemetry import check_deadline
        check_deadline()
        if current_profile().mode.value in {'QUICK', 'OFFLINE'}:
            return PriceForecastService().forecast_next_session(market_data)
        with self._lock:
            check_deadline()
            if self._service is None:
                self._service = PriceForecastService.from_environment(**self.configuration)
        check_deadline()
        return self._service.forecast_next_session(market_data)
