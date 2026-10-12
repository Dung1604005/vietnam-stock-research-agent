from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl, field_validator


class ResearchDocumentType(str, Enum):
    FINANCIAL_STATEMENT = "financial_statement"
    ANNUAL_REPORT = "annual_report"
    REGULATORY_DISCLOSURE = "regulatory_disclosure"
    EARNINGS_RELEASE = "earnings_release"
    INVESTOR_PRESENTATION = "investor_presentation"
    NEWS = "news"
    OTHER = "other"


class ResearchDocument(BaseModel):
    document_id: str = Field(min_length=16, max_length=128)
    title: str = Field(min_length=1, max_length=500)
    document_type: ResearchDocumentType
    source_url: HttpUrl
    publisher: str = Field(min_length=1, max_length=200)
    published_at: datetime
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    tickers: list[str] = Field(default_factory=list)
    language: str = "vi"

    @field_validator("published_at", "retrieved_at")
    @classmethod
    def timestamps_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Research document timestamps must include a timezone")
        return value


class DocumentChunk(BaseModel):
    chunk_id: str = Field(min_length=16, max_length=160)
    document_id: str
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    title: str
    source_url: HttpUrl
    publisher: str
    published_at: datetime
    retrieved_at: datetime
    content_sha256: str
    quote: str = Field(min_length=1, max_length=1400)
    retrieval_score: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_modes: list[str] = Field(default_factory=list)


class WebCitation(BaseModel):
    """A live web discovery result, deliberately distinct from verified evidence."""

    title: str = Field(min_length=1, max_length=500)
    source_url: HttpUrl
    source_domain: str = Field(min_length=1, max_length=255)
    snippet: str = Field(min_length=1, max_length=1400)
    search_provider: str = Field(min_length=1, max_length=100)
    retrieved_at: datetime
    rank: int = Field(ge=1)

    @field_validator("retrieved_at")
    @classmethod
    def retrieved_time_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Web citation retrieval timestamps must include a timezone")
        return value


class ResearchQuery(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    purpose: str = Field(min_length=1, max_length=120)


class ResearchPlan(BaseModel):
    goal: str = Field(min_length=1, max_length=1000)
    ticker: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9]{1,9}$")
    queries: list[ResearchQuery] = Field(min_length=1, max_length=3)
    max_searches: int = Field(ge=1, le=3)
    max_pages: int = Field(ge=1, le=4)
    max_iterations: int = Field(ge=1, le=2)
    timeout_seconds: int = Field(ge=1, le=60)
    max_total_bytes: int = Field(ge=1_000, le=8_000_000)


class FetchedDocument(BaseModel):
    requested_url: HttpUrl
    final_url: HttpUrl
    content_type: str = Field(min_length=1, max_length=200)
    status_code: int = Field(ge=200, le=299)
    retrieved_at: datetime
    body: str = Field(min_length=1)

    @field_validator("retrieved_at")
    @classmethod
    def fetched_time_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Fetched document timestamps must include a timezone")
        return value


class ExtractedDocument(BaseModel):
    requested_url: HttpUrl
    final_url: HttpUrl
    canonical_url: HttpUrl
    title: str = Field(min_length=1, max_length=500)
    published_at: datetime | None = None
    retrieved_at: datetime
    text: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("published_at", "retrieved_at")
    @classmethod
    def extracted_times_must_be_aware_when_present(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Extracted document timestamps must include a timezone")
        return value


class ExternalEvidenceCitation(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    source_url: HttpUrl
    canonical_url: HttpUrl
    source_domain: str = Field(min_length=1, max_length=255)
    published_at: datetime | None = None
    retrieved_at: datetime
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    quote: str = Field(min_length=1, max_length=1400)
    evidence_location: str = Field(min_length=1, max_length=200)
    relevance_score: float = Field(ge=0.0, le=1.0)

    @field_validator("published_at", "retrieved_at")
    @classmethod
    def external_times_must_be_aware_when_present(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("External evidence timestamps must include a timezone")
        return value


class EvidenceClaim(BaseModel):
    claim: str = Field(min_length=1, max_length=1000)
    citation: Citation
    confidence: float = Field(ge=0.0, le=1.0)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvidenceBundle(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    citations: list[Citation] = Field(default_factory=list)
    web_citations: list[WebCitation] = Field(default_factory=list)
    external_citations: list[ExternalEvidenceCitation] = Field(default_factory=list)
    retrieval_note: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
