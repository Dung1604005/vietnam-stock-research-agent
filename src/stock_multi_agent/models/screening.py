from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field

from stock_multi_agent.models.analysis import Recommendation, RiskLevel, Signal
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


class RecommendationBucket(str, Enum):
    HIGH_REWARD_HIGH_RISK = "HIGH_REWARD_HIGH_RISK"
    SAFE_MODERATE_RETURN = "SAFE_MODERATE_RETURN"
    UNCLASSIFIED = "UNCLASSIFIED"


class CandidateSelectionTier(str, Enum):
    QUALIFIED = "QUALIFIED"
    REFERENCE_ONLY = "REFERENCE_ONLY"


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
    filters: dict[str, float] = Field(default_factory=dict)
    exchange: str | None = None
    maximum_candidates: int = Field(default=100, ge=1, le=100)
    minimum_technical_score: float = Field(default=4.0, ge=0, le=10)
    eligible_signals: list[Signal] = Field(default_factory=lambda: [Signal.BUY, Signal.HOLD])
    missing_data_policy: MissingDataPolicy = MissingDataPolicy.EXCLUDE_FROM_RANKING
    require_same_price_as_of: bool = True
    max_price_as_of_lag_days: int = Field(default=3, ge=0, le=10)
    max_price_age_days: int | None = Field(default=None, ge=0, le=30)
    refresh_stale_data: bool = True
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


class ProgressiveScreenRound(BaseModel):
    round_number: int = Field(ge=1)
    input_count: int = Field(ge=0)
    survivor_count: int = Field(ge=0)
    eliminated_tickers: list[str] = Field(default_factory=list)


class RecommendationCandidate(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z][A-Z0-9]{1,9}$")
    bucket: RecommendationBucket
    rank_in_bucket: int = Field(ge=1)
    technical_score: float = Field(ge=0, le=10)
    fundamental_score: float = Field(ge=0, le=10)
    risk_score: float = Field(ge=0, le=10)
    risk_level: RiskLevel
    recommendation: Recommendation
    confidence: float = Field(ge=0, le=1)
    volatility: float = Field(ge=0)
    reasoning: str
    evidence_highlights: list[str] = Field(default_factory=list, max_length=3)
    as_of: date | None = None
    source_ids: list[str] = Field(default_factory=list)
    selection_tier: CandidateSelectionTier = CandidateSelectionTier.QUALIFIED
    policy_gaps: list[str] = Field(default_factory=list)
    veto: bool = False


class DualRecommendation(BaseModel):
    high_reward: list[RecommendationCandidate] = Field(default_factory=list)
    safe_picks: list[RecommendationCandidate] = Field(default_factory=list)
    reference_only: list[RecommendationCandidate] = Field(default_factory=list)
    universe_key: str
    universe_size: int = Field(ge=0)
    screened_count: int = Field(ge=0)
    analyzed_count: int = Field(ge=0)
    requested_count: int = Field(ge=1, le=100)
    returned_count: int = Field(ge=0)
    qualified_count: int = Field(default=0, ge=0)
    reference_count: int = Field(default=0, ge=0)
    horizon: str
    progressive_rounds: list[ProgressiveScreenRound] = Field(default_factory=list)
    deep_analysis_limits: list[int] = Field(default_factory=list)
    refreshed_tickers: list[str] = Field(default_factory=list)
    price_as_of_comparable: bool = False
    quick_screen_seconds: float = Field(default=0.0, ge=0)
    deep_analysis_seconds: float = Field(default=0.0, ge=0)
    total_seconds: float = Field(default=0.0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    policy_version: str = "dual-recommendation-v2-reference-fallback"
    disclaimer: str = "Educational analysis only — not financial advice."


class ComparisonSpec(BaseModel):
    """Field-alignment policy for an explicitly named comparison set."""

    tickers: list[str] = Field(min_length=2, max_length=5)
    require_same_currency: bool = True
    require_same_price_as_of: bool = True
    require_same_financial_period: bool = True
    policy_version: str = "comparison-v1"
