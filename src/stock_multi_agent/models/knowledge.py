from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl, field_validator


class KnowledgeSourceType(str, Enum):
    OFFICIAL_REGULATION = "official_regulation"
    REGULATOR_EDUCATION = "regulator_education"
    EXCHANGE_GUIDANCE = "exchange_guidance"
    ACCOUNTING_STANDARD = "accounting_standard"
    OPEN_EDUCATION = "open_education"
    PROFESSIONAL_EDUCATION = "professional_education"
    ISSUER_IR = "issuer_ir"
    OTHER = "other"


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    REVIEWED = "REVIEWED"
    UNVERIFIED = "UNVERIFIED"


class KnowledgeNodeType(str, Enum):
    DOCUMENT = "DOCUMENT"
    TOPIC = "TOPIC"
    CONCEPT = "CONCEPT"
    SECTION = "SECTION"
    PUBLISHER = "PUBLISHER"
    COMPANY = 'COMPANY'
    TICKER = 'TICKER'
    EXCHANGE = 'EXCHANGE'
    SECTOR = 'SECTOR'
    PERSON = 'PERSON'
    EVENT = 'EVENT'
    METRIC = 'METRIC'
    PERIOD = 'PERIOD'
    RISKFACTOR = 'RISKFACTOR'


class KnowledgeRelationType(str, Enum):
    HAS_TOPIC = "HAS_TOPIC"
    MENTIONS = "MENTIONS"
    HAS_SECTION = "HAS_SECTION"
    PUBLISHED_BY = "PUBLISHED_BY"
    ABOUT_TOPIC = "ABOUT_TOPIC"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"
    LISTED_AS = 'LISTED_AS'
    IN_SECTOR = 'IN_SECTOR'
    PUBLISHED = 'PUBLISHED'
    REPORTS_METRIC = 'REPORTS_METRIC'
    HAS_EVENT = 'HAS_EVENT'
    EXPOSED_TO = 'EXPOSED_TO'
    RELATED_TO = 'RELATED_TO'
    SUPERSEDES = 'SUPERSEDES'


class KnowledgeDocument(BaseModel):
    document_id: str = Field(min_length=16, max_length=128)
    title: str = Field(min_length=1, max_length=500)
    publisher: str = Field(min_length=1, max_length=200)
    source_url: HttpUrl
    canonical_url: HttpUrl
    source_type: KnowledgeSourceType
    language: str = Field(default="vi", min_length=2, max_length=12)
    published_at: datetime | None = None
    effective_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    usage_note: str = Field(min_length=1, max_length=1000)
    verification_status: VerificationStatus
    topics: list[str] = Field(min_length=1)
    version: str = Field(min_length=1, max_length=80)
    time_sensitive: bool = False

    @field_validator("published_at", "effective_at", "retrieved_at")
    @classmethod
    def timestamps_are_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Knowledge timestamps must include a timezone")
        return value


class KnowledgeChunk(BaseModel):
    chunk_id: str = Field(min_length=16, max_length=160)
    document_id: str
    ordinal: int = Field(ge=0)
    heading: str = Field(default="", max_length=500)
    text: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)


class KnowledgeCitation(BaseModel):
    document_id: str
    chunk_id: str
    title: str
    heading: str = ""
    publisher: str
    source_url: HttpUrl
    source_type: KnowledgeSourceType
    published_at: datetime | None = None
    effective_at: datetime | None = None
    retrieved_at: datetime
    content_sha256: str
    verification_status: VerificationStatus
    topics: list[str]
    version: str
    quote: str = Field(min_length=1, max_length=2400)
    score: float = Field(ge=0.0)
    retrieval_modes: list[str] = Field(default_factory=list)
    freshness_warning: str | None = None


class KnowledgeGraphNode(BaseModel):
    node_id: str = Field(min_length=3, max_length=180)
    node_type: KnowledgeNodeType
    label: str = Field(min_length=1, max_length=500)
    aliases: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class KnowledgeGraphPath(BaseModel):
    node_ids: list[str] = Field(min_length=2)
    labels: list[str] = Field(min_length=2)
    relations: list[KnowledgeRelationType] = Field(min_length=1)
    score: float = Field(ge=0.0)
    evidence_document_id: str | None = None
    evidence_chunk_id: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class KnowledgeGraphContext(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    matched_nodes: list[KnowledgeGraphNode] = Field(default_factory=list)
    paths: list[KnowledgeGraphPath] = Field(default_factory=list)
    related_document_ids: list[str] = Field(default_factory=list)
    expansion_terms: list[str] = Field(default_factory=list)
    retrieval_note: str = ""


class KnowledgeSearchResult(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    citations: list[KnowledgeCitation] = Field(default_factory=list)
    retrieval_note: str
    retrieval_mode: str = "lexical"
    requires_freshness_check: bool = False
    graph_context: KnowledgeGraphContext | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

