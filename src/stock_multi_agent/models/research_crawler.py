"""Persistent crawler contracts; defaults also accept legacy source registries."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

from stock_multi_agent.models.research import ResearchDocumentType, ReportingPeriod


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StorageMode(str, Enum):
    FULL_TEXT = "FULL_TEXT"
    EXCERPT_ONLY = "EXCERPT_ONLY"
    LINK_ONLY = "LINK_ONLY"
    BLOCKED = "BLOCKED"


class CandidateStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    FETCHED = "FETCHED"
    QUARANTINED = "QUARANTINED"
    INDEXED = "INDEXED"
    UNCHANGED = "UNCHANGED"
    RETRYABLE_ERROR = "RETRYABLE_ERROR"
    PERMANENT_ERROR = "PERMANENT_ERROR"
    REJECTED = "REJECTED"


class SourceEntrypoint(BaseModel):
    url: HttpUrl
    kind: str = Field(default="listing", pattern="^(listing|rss|sitemap|json)$")


class SourceDefinition(BaseModel):
    source_id: str = ""
    domain: str
    publisher: str
    quality_tier: str = Field(pattern=r"^T[1-4]$")
    usage_note: str = Field(min_length=1)
    refresh_hours: int = Field(default=72, ge=1)
    allow_full_text: bool = False
    entrypoints: list[SourceEntrypoint] = Field(default_factory=list)
    allowed_path_prefixes: list[str] = Field(default_factory=lambda: ["/"])
    asset_domains: dict[str, list[str]] = Field(default_factory=dict)
    document_patterns: list[str] = Field(default_factory=lambda: ["bao-cao", "bctc", "report", "financial", ".pdf", "cong-bo", "disclosure"])
    ticker_map: list[str] = Field(default_factory=list)
    discovery_allowed: bool = False
    fetch_allowed: bool = False
    storage_mode: StorageMode = StorageMode.LINK_ONLY
    min_request_interval_seconds: float = Field(default=3, ge=0)
    max_pages_per_run: int = Field(default=30, ge=1, le=500)
    max_depth: int = Field(default=2, ge=0, le=3)
    max_document_bytes: int = Field(default=8_000_000, ge=1000, le=32_000_000)
    timezone: str = "Asia/Ho_Chi_Minh"
    brave_fallback: bool = False

    @model_validator(mode="after")
    def identity(self):
        self.domain = self.domain.lower().rstrip(".")
        if not self.source_id:
            self.source_id = self.domain
        if not self.domain or "/" in self.domain or ":" in self.domain:
            raise ValueError("domain must be a hostname")
        if not self.allowed_path_prefixes or any(not x.startswith("/") for x in self.allowed_path_prefixes):
            raise ValueError("Explicit absolute path prefixes required")
        for domain, prefixes in self.asset_domains.items():
            if not domain or any(char in domain for char in "/:@") or not prefixes or any(not value.startswith("/") for value in prefixes):
                raise ValueError("Asset domains require hostname and explicit path prefixes")
        if self.allow_full_text and self.storage_mode == StorageMode.LINK_ONLY:
            self.storage_mode = StorageMode.FULL_TEXT
        return self

    def accepts(self, url: str) -> bool:
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError:
            return False
        host = (parsed.hostname or "").lower().rstrip(".")
        path = unquote(parsed.path or "/")
        if parsed.scheme != "https" or parsed.username or parsed.password or port not in (None, 443):
            return False
        main_domain = host == self.domain or host.endswith("." + self.domain)
        if not main_domain and host not in self.asset_domains:
            return False
        if "\\" in path or any(piece in {"..", "."} for piece in path.split("/")):
            return False
        if main_domain and any(path == urlsplit(str(item.url)).path for item in self.entrypoints):
            return True
        prefixes = self.asset_domains.get(host, self.allowed_path_prefixes)
        return any(prefix == "/" or path == prefix.rstrip("/") or path.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)


class DocumentCandidate(BaseModel):
    candidate_id: str = ""
    source_id: str
    source_url: HttpUrl
    parent_url: HttpUrl | None = None
    discovered_at: datetime = Field(default_factory=utcnow)
    title: str = ""
    publisher: str = ""
    tickers: list[str] = Field(default_factory=list)
    document_type: ResearchDocumentType = ResearchDocumentType.OTHER
    published_at: datetime | None = None
    date_evidence: str = ""
    period_end: str | None = None
    reporting_period: ReportingPeriod = Field(default_factory=ReportingPeriod)
    metadata_confidence: float = Field(default=0, ge=0, le=1)
    storage_mode: StorageMode = StorageMode.LINK_ONLY
    status: CandidateStatus = CandidateStatus.DISCOVERED
    reason: str = ""
    excerpt: str = ""
    approved_full_text: bool = False
    approval_note: str = ""
    document_id: str | None = None
    attempts: int = 0
    next_attempt_at: datetime | None = None
    etag: str | None = None
    last_modified: str | None = None
    last_checked_at: datetime | None = None

    @field_validator("published_at", "discovered_at", "next_attempt_at", "last_checked_at")
    @classmethod
    def aware(cls, value):
        if value is not None and value.utcoffset() is None:
            raise ValueError("Crawler timestamps require timezone")
        return value


class IngestionDecision(BaseModel):
    allowed: bool
    storage_mode: StorageMode
    reason: str


class CrawlRun(BaseModel):
    run_id: str
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    status: str = "RUNNING"
    counts: dict[str, int] = Field(default_factory=dict)
    sources: list[dict] = Field(default_factory=list)
    request_count: int = 0
    bytes_downloaded: int = 0
    errors: list[dict[str, str]] = Field(default_factory=list)
