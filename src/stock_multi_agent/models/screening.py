from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field

from stock_multi_agent.models.analysis import Signal
from stock_multi_agent.models.data import SourceReference


class ScreenStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNAVAILABLE = "UNAVAILABLE"


class MissingDataPolicy(str, Enum):
    EXCLUDE_FROM_RANKING = "EXCLUDE_FROM_RANKING"
    SHOW_AS_UNAVAILABLE = "SHOW_AS_UNAVAILABLE"


class RankingDirection(str, Enum):
    HIGHER_SCORE_FIRST = "HIGHER_SCORE_FIRST"
    LOWER_SCORE_FIRST = "LOWER_SCORE_FIRST"


class UniverseDefinition(BaseModel):
    """A disclosed universe. Current captures are never historical membership evidence."""

    key: str = Field(pattern=r"^[a-z0-9_]+$")
    tickers: list[str] = Field(min_length=1, max_length=3000)
    source: str
    membership_as_of: date | None = None
    captured_at: datetime | None = None
    source_reference: SourceReference | None = None
    point_in_time_supported: bool = False


class ScreenSpec(BaseModel):
    """Versioned, deterministic screening contract for a disclosed universe."""

    universe_key: str = Field(pattern=r"^[a-z0-9_]+$")
    as_of: date | None = None
    maximum_candidates: int = Field(default=30, ge=1, le=100)
    minimum_technical_score: float = Field(default=4.0, ge=0, le=10)
    eligible_signals: list[Signal] = Field(default_factory=lambda: [Signal.BUY, Signal.HOLD])
    missing_data_policy: MissingDataPolicy = MissingDataPolicy.EXCLUDE_FROM_RANKING
    require_same_price_as_of: bool = True
    candidate_selection_policy: str = "configured_order_v1"
    factor_policy_version: str = "screen-v2"
    ranking_direction: RankingDirection = RankingDirection.HIGHER_SCORE_FIRST


class ScreenCandidate(BaseModel):
    ticker: str
    status: ScreenStatus
    technical_signal: Signal | None = None
    technical_score: float | None = Field(default=None, ge=0, le=10)
    as_of: date | None = None
    source_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    rank: int | None = Field(default=None, ge=1)


class ComparisonSpec(BaseModel):
    """Field-alignment policy for an explicitly named comparison set."""

    tickers: list[str] = Field(min_length=2, max_length=4)
    require_same_currency: bool = True
    require_same_price_as_of: bool = True
    require_same_financial_period: bool = True
    policy_version: str = "comparison-v1"
