from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from stock_multi_agent.models.data import (
    CorporateAction,
    DataQualityReport,
    FinancialMetricObservation,
    FinancialStatementObservation,
    Instrument,
    IssuerEvent,
    PriceAdjustment,
    SourceReference,
)


class Signal(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    UNKNOWN = "UNKNOWN"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Recommendation(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    WATCH = "WATCH"
    AVOID = "AVOID"


class PriceBar(BaseModel):
    date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_ohlc(self) -> "PriceBar":
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be the greatest OHLC value")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be the smallest OHLC value")
        return self


class FinancialMetrics(BaseModel):
    revenue_growth: float | None = None
    profit_growth: float | None = None
    eps: float | None = None
    pe: float | None = None
    pb: float | None = None
    roe: float | None = None
    roa: float | None = None
    debt_to_equity: float | None = None
    operating_cash_flow: float | None = None
    profit_margin: float | None = None

    def available_count(self) -> int:
        return sum(value is not None for value in self.model_dump().values())


class EvidenceItem(BaseModel):
    """A typed analytical observation with deterministic input lineage."""

    label: str = Field(min_length=1, max_length=160)
    value: str = Field(min_length=1, max_length=500)
    input_fields: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    as_of: date | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)
    note: str | None = Field(default=None, max_length=500)


class RiskSection(BaseModel):
    """One explicit risk dimension; UNKNOWN means its input was unavailable."""

    name: str = Field(min_length=1, max_length=80)
    status: str = Field(pattern=r"^(ASSESSED|PARTIAL|UNKNOWN)$")
    observations: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    coverage: float = Field(default=0, ge=0, le=1)


class MarketData(BaseModel):
    ticker: str
    instrument: Instrument | None = None
    source: str
    access_mode: str = 'UNKNOWN'
    price_unit: str = 'UNKNOWN'
    company_name: str | None = None
    bars: list[PriceBar] = Field(min_length=1)
    financials: FinancialMetrics = Field(default_factory=FinancialMetrics)
    financial_observations: list[FinancialMetricObservation] = Field(default_factory=list)
    financial_statement_observations: list[FinancialStatementObservation] = Field(default_factory=list)
    source_refs: list[SourceReference] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    as_of: date | None = None
    price_adjustment: PriceAdjustment = PriceAdjustment.UNKNOWN
    corporate_actions: list[CorporateAction] = Field(default_factory=list)
    issuer_events: list[IssuerEvent] = Field(default_factory=list)
    quality_report: DataQualityReport = Field(default_factory=DataQualityReport)
    schema_version: str = "market-data-v3"
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def bars_must_be_sorted(self) -> "MarketData":
        dates = [bar.date for bar in self.bars]
        if dates != sorted(dates) or len(dates) != len(set(dates)):
            raise ValueError("price bars must have unique ascending dates")
        if self.fetched_at.tzinfo is None or self.fetched_at.utcoffset() is None:
            raise ValueError("fetched_at must be timezone-aware")
        if self.as_of is None:
            self.as_of = self.bars[-1].date
        if self.instrument is not None and self.instrument.ticker != self.ticker:
            raise ValueError("instrument ticker must match market data ticker")
        return self


class TechnicalAnalysis(BaseModel):
    signal: Signal
    score: float = Field(ge=0, le=10)
    confidence: float = Field(ge=0, le=1)
    evidence: list[EvidenceItem]
    risks: list[str]
    tools_used: list[str]
    reasoning_summary: str
    timeframe: str
    horizon_profile: str = "medium"
    policy_version: str = "technical-v2"
    feature_contributions: dict[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, str | float | bool | None] = Field(default_factory=dict)


class FundamentalAnalysis(BaseModel):
    signal: Signal
    score: float = Field(ge=0, le=10)
    confidence: float = Field(ge=0, le=1)
    strengths: list[str]
    weaknesses: list[str]
    evidence: list[EvidenceItem]
    reasoning_summary: str
    sector_template: str = "UNKNOWN"
    policy_version: str = "fundamental-v3-quality-weighted"
    feature_contributions: dict[str, float] = Field(default_factory=dict)
    unavailable_metrics: list[str] = Field(default_factory=list)
    research_context_count: int = Field(default=0, ge=0)


class RiskAnalysis(BaseModel):
    risk_level: RiskLevel
    risk_score: float = Field(ge=0, le=10)
    criticisms: list[str]
    warnings: list[str]
    veto: bool
    reasoning_summary: str
    sections: list[RiskSection] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
    policy_version: str = "risk-v2"
    feature_contributions: dict[str, float] = Field(default_factory=dict)
    coverage: dict[str, float] = Field(default_factory=dict)
    sector_template: str = "UNKNOWN"
    issuer_risk_level: RiskLevel | None = None
    issuer_risk_score: float | None = Field(default=None, ge=0, le=10)
    data_uncertainty_level: RiskLevel = RiskLevel.MEDIUM
    data_uncertainty_score: float = Field(default=5.0, ge=0, le=10)


class FinalDecision(BaseModel):
    ticker: str
    recommendation: Recommendation
    confidence: float = Field(ge=0, le=1)
    technical_summary: str
    fundamental_summary: str
    risk_summary: str
    agreements: list[str]
    conflicts: list[str]
    final_reasoning: str
    evidence: list[str]
    policy_version: str = "synthesis-v2"
    calibration_status: str = "UNCALIBRATED_NOT_PREDICTIVE"
    evidence_completeness: float = Field(default=0.0, ge=0, le=1)
    disclaimer: str = "Educational analysis only — not financial advice or a trade order."
