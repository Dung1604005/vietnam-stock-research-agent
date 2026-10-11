from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DataKind(str, Enum):
    INSTRUMENT_UNIVERSE = "INSTRUMENT_UNIVERSE"
    OHLCV = "OHLCV"
    FINANCIAL_RATIOS = "FINANCIAL_RATIOS"
    INCOME_STATEMENT = "INCOME_STATEMENT"
    BALANCE_SHEET = "BALANCE_SHEET"
    CASH_FLOW = "CASH_FLOW"
    COMPANY_PROFILE = "COMPANY_PROFILE"
    CORPORATE_ACTIONS = "CORPORATE_ACTIONS"
    ISSUER_EVENTS = "ISSUER_EVENTS"


class InstrumentType(str, Enum):
    EQUITY = "EQUITY"
    INDEX = "INDEX"


class Instrument(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z][A-Z0-9]{1,9}$")
    name: str | None = None
    exchange: str | None = None
    sector: str | None = None
    instrument_type: InstrumentType = InstrumentType.EQUITY
    currency: str = "VND"
    active: bool | None = None


class PriceAdjustment(str, Enum):
    ADJUSTED = "ADJUSTED"
    UNADJUSTED = "UNADJUSTED"
    UNKNOWN = "UNKNOWN"


class QualitySeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class QualityStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class SourceReference(BaseModel):
    provider: str
    source_id: str
    data_kind: DataKind
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    as_of: date | None = None
    source_url: str | None = None
    request_parameters: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1"
    content_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    snapshot_id: str | None = None

    @field_validator("fetched_at")
    @classmethod
    def fetched_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("fetched_at must be timezone-aware")
        return value


class FinancialMetricObservation(BaseModel):
    metric: str
    value: float
    unit: str
    period_label: str
    period_end: date | None = None
    published_at: datetime | None = None
    source_id: str

    @field_validator("published_at")
    @classmethod
    def published_at_must_be_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("published_at must be timezone-aware")
        return value


class FinancialStatementObservation(BaseModel):
    """One reported statement line for a known period; never an imputed value."""

    statement: str = Field(pattern=r"^(income_statement|balance_sheet|cash_flow)$")
    metric: str
    value: float
    unit: str
    period_label: str
    period_end: date | None = None
    frequency: str = Field(default="annual", pattern=r"^(annual|quarter)$")
    published_at: datetime | None = None
    source_id: str

    @field_validator("published_at")
    @classmethod
    def published_at_must_be_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("published_at must be timezone-aware")
        return value


class CorporateAction(BaseModel):
    action_type: str
    ex_date: date
    description: str
    ratio_or_amount: float | None = None
    source_id: str


class IssuerEvent(BaseModel):
    """An issuer event reported by a source; it is not a news-derived claim."""

    event_code: str | None = None
    title: str = Field(min_length=1, max_length=500)
    event_date: date
    published_date: date | None = None
    source_id: str


class PortfolioPosition(BaseModel):
    """User-supplied portfolio weight for deterministic concentration checks."""

    ticker: str = Field(pattern=r"^[A-Z][A-Z0-9]{1,9}$")
    weight: float = Field(gt=0, le=1)


class DataQualityIssue(BaseModel):
    code: str
    severity: QualitySeverity
    message: str
    field: str | None = None
    observed: Any | None = None


class DataQualityReport(BaseModel):
    status: QualityStatus = QualityStatus.PASS
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    checks_run: list[str] = Field(default_factory=list)
    issues: list[DataQualityIssue] = Field(default_factory=list)

    @field_validator("checked_at")
    @classmethod
    def checked_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checked_at must be timezone-aware")
        return value

    @property
    def error_count(self) -> int:
        return sum(issue.severity is QualitySeverity.ERROR for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity is QualitySeverity.WARNING for issue in self.issues)
