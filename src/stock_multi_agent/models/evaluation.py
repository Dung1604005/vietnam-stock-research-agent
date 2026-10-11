from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from stock_multi_agent.models.data import SourceReference


class PointInTimeUniverseSnapshot(BaseModel):
    """Membership known at a historical rebalance date, never a current watchlist."""

    as_of: date
    tickers: list[str] = Field(min_length=1, max_length=5000)
    source_reference: SourceReference
    point_in_time_supported: bool = True


class SignalSnapshot(BaseModel):
    """A deterministic signal frozen before its rebalance date."""

    ticker: str = Field(pattern=r"^[A-Z][A-Z0-9]{1,9}$")
    as_of: date
    available_at: datetime
    score: float = Field(ge=0, le=10)
    policy_version: str
    source_ids: list[str] = Field(min_length=1)

    @field_validator("available_at")
    @classmethod
    def available_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("available_at must be timezone-aware")
        return value


class HistoricalPrice(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z][A-Z0-9]{1,9}$")
    session: date
    close: float = Field(gt=0)
    adjusted: bool
    source_reference: SourceReference


class BacktestDataset(BaseModel):
    """A portable manifest of only data available at the historical decision time."""

    dataset_version: str = Field(min_length=1)
    universe_snapshots: list[PointInTimeUniverseSnapshot] = Field(min_length=2)
    signals: list[SignalSnapshot] = Field(min_length=1)
    prices: list[HistoricalPrice] = Field(min_length=1)
    benchmark_ticker: str = "VNINDEX"
    benchmark_prices: list[HistoricalPrice] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def distinct_universe_dates(self) -> "BacktestDataset":
        dates = [snapshot.as_of for snapshot in self.universe_snapshots]
        if len(set(dates)) != len(dates):
            raise ValueError("universe_snapshots must have one snapshot per as_of date")
        return self


class ForwardEvaluationCapture(BaseModel):
    """One real-time observation used only for forward evaluation from capture onward."""

    schema_version: str = "forward-evaluation-capture-v1"
    captured_at: datetime
    universe: PointInTimeUniverseSnapshot
    signals: list[SignalSnapshot] = Field(default_factory=list)
    prices: list[HistoricalPrice] = Field(default_factory=list)
    benchmark_prices: list[HistoricalPrice] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)

    @field_validator("captured_at")
    @classmethod
    def captured_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        return value


class BacktestConfig(BaseModel):
    strategy_version: str = "rank-top-n-v1"
    top_n: int = Field(default=10, ge=1, le=100)
    execution_lag_sessions: int = Field(default=1, ge=1, le=5)
    transaction_cost_bps: float = Field(default=15.0, ge=0, le=1000)
    require_adjusted_prices: bool = True
    bootstrap_samples: int = Field(default=500, ge=100, le=5000)
    random_seed: int = Field(default=7, ge=0)


class WalkForwardConfig(BaseModel):
    training_windows: int = Field(ge=1)
    test_windows: int = Field(ge=1)
    step_windows: int = Field(ge=1)


class WalkForwardWindow(BaseModel):
    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date


class BacktestPeriod(BaseModel):
    signal_as_of: date
    entry_session: date
    exit_session: date
    selected_tickers: list[str]
    gross_return: float
    transaction_cost: float
    net_return: float
    benchmark_return: float
    turnover: float


class BacktestReport(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_version: str
    strategy_version: str
    input_sha256: str
    config: BacktestConfig
    periods: list[BacktestPeriod]
    metrics: dict[str, float | int | None]
    confidence_intervals: dict[str, tuple[float, float] | None]
    failures: list[str] = Field(default_factory=list)
    walk_forward: list[WalkForwardWindow] = Field(default_factory=list)
    disclaimer: str = "Historical evaluation only; not a prediction, recommendation, or trade instruction."
