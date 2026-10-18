from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ForecastStatus(str, Enum):
    BASELINE_UNTRAINED = "BASELINE_UNTRAINED"
    TRAINED_MODEL = "TRAINED_MODEL"
    UNAVAILABLE = "UNAVAILABLE"


class PriceForecast(BaseModel):
    """A transparent next-session estimate, separate from an investment recommendation."""
    model_config = ConfigDict(allow_inf_nan=False)

    ticker: str = Field(pattern=r"^[A-Z][A-Z0-9]{1,9}$")
    as_of: date
    current_price: float = Field(gt=0)
    horizon_sessions: int = Field(default=1, ge=1, le=20)
    forecast_price: float | None = Field(default=None, gt=0)
    lower_bound: float | None = Field(default=None, gt=0)
    upper_bound: float | None = Field(default=None, gt=0)
    status: ForecastStatus
    method: str
    model_version: str
    confidence: float = Field(ge=0, le=1)
    artifact_status: str = 'UNKNOWN'
    validation_status: str = 'LEGACY_NOT_REVALIDATED'
    policy_eligibility: bool | None = None
    technical_blend_weight: float | None = Field(default=None, ge=0, le=0.2)
    experimental_blend_enabled: bool = False
    shadow_forecast: dict | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def bounds_must_wrap_forecast(self) -> "PriceForecast":
        if self.forecast_price is None:
            if self.lower_bound is not None or self.upper_bound is not None:
                raise ValueError("Forecast bounds require a forecast_price")
            return self
        if self.lower_bound is not None and self.lower_bound > self.forecast_price:
            raise ValueError("lower_bound cannot exceed forecast_price")
        if self.upper_bound is not None and self.upper_bound < self.forecast_price:
            raise ValueError("upper_bound cannot be below forecast_price")
        return self
