"""Incremental, explicitly approved source-manifest ingestion."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl, field_validator

from stock_multi_agent.models.research import ResearchDocumentType, ExtractedDocument
from stock_multi_agent.services.content_extractor import HtmlContentExtractor
from stock_multi_agent.services.content_fetcher import SafeContentFetcher
from stock_multi_agent.services.research_store import ResearchStore
from stock_multi_agent.services.source_registry import SourceRegistry, canonical_url


class IngestionEntry(BaseModel):
    source_url: HttpUrl
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    published_at: datetime
    document_type: ResearchDocumentType
    tickers: list[str] = Field(default_factory=list)
    approved_full_text: bool = False
    usage_note: str = Field(min_length=1)
    local_path: str | None = None
    refresh_hours: int = Field(default=72, ge=1)

    @field_validator("published_at")
    @classmethod
    def aware(cls, value):
        if value.utcoffset() is None:
            raise ValueError("published_at requires timezone")
        return value


class ResearchIngestionService:
    def __init__(self, store: ResearchStore, fetcher=None, registry=None):
        self.store = store
        self.fetcher = fetcher or SafeContentFetcher(max_bytes=8_000_000)
        self.registry = registry or SourceRegistry()
        with store._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS research_ingest_checks (source_url TEXT PRIMARY KEY, checked_at TEXT NOT NULL, document_id TEXT NOT NULL)")

    def ingest_extracted(self, entry: IngestionEntry, extracted: ExtractedDocument, *, period_end=None) -> dict:
        """Use an already fetched, validated document without a second HTTP request."""
        url = canonical_url(str(entry.source_url))
        policy = self.registry.resolve(url)
        if (policy is None or policy.storage_mode.value == "BLOCKED" or not policy.accepts(url)
                or canonical_url(str(extracted.requested_url)) != url or not policy.accepts(str(extracted.final_url))):
            raise ValueError("Extracted document is outside its registered source")
        if not entry.approved_full_text and not policy.allow_full_text and policy.storage_mode.value != "FULL_TEXT":
            return {"source_url": url, "status": "SKIPPED_PERMISSION"}
        with self.store._connect() as db:
            previous = db.execute("SELECT document_id FROM research_ingest_checks WHERE source_url=?", (url,)).fetchone()
        document = self.store.ingest_text(text=extracted.text, title=entry.title, source_url=url,
            publisher=entry.publisher, published_at=entry.published_at, document_type=entry.document_type,
            tickers=entry.tickers, usage_note=entry.usage_note, quality_tier=policy.quality_tier, period_end=period_end)
        with self.store._connect() as db:
            db.execute("INSERT OR REPLACE INTO research_ingest_checks VALUES (?,?,?)", (url, datetime.now(timezone.utc).isoformat(), document.document_id))
        return {"source_url": url, "status": "UNCHANGED" if previous and previous[0] == document.document_id else "INDEXED",
                "document_id": document.document_id, "version": document.version}

    def ingest(self, entry: IngestionEntry, *, force=False, base_path: Path | None = None) -> dict:
        url = canonical_url(str(entry.source_url))
        policy = self.registry.resolve(url)
        if not entry.approved_full_text and not (policy and policy.allow_full_text):
            return {"source_url": url, "status": "SKIPPED_PERMISSION"}
        # Unregistered sources may be explicitly ingested from a local file;
        # network auto-ingest is limited to the registry allowlist.
        if not policy and not entry.local_path:
            return {"source_url": url, "status": "SKIPPED_UNREGISTERED"}
        now = datetime.now(timezone.utc)
        with self.store._connect() as connection:
            previous = connection.execute("SELECT * FROM research_ingest_checks WHERE source_url=?", (url,)).fetchone()
        ttl = min(entry.refresh_hours, policy.refresh_hours if policy else entry.refresh_hours)
        if previous and not force and not entry.local_path and (now - datetime.fromisoformat(previous["checked_at"])).total_seconds() < ttl * 3600:
            return {"source_url": url, "status": "CACHED", "document_id": previous["document_id"]}
        if entry.local_path:
            path = (base_path or Path.cwd()) / entry.local_path
            text = self.store.read_text_file(path.resolve())
        else:
            fetched = self.fetcher.fetch(url)
            if not self.registry.resolve(str(fetched.final_url)):
                raise ValueError("Redirect left registered sources; ingestion refused")
            extracted = HtmlContentExtractor().extract(fetched)
            text = extracted.text
        document = self.store.ingest_text(text=text, title=entry.title, source_url=url,
            publisher=entry.publisher, published_at=entry.published_at, document_type=entry.document_type,
            tickers=entry.tickers, usage_note=entry.usage_note,
            quality_tier=policy.quality_tier if policy else "T4")
        with self.store._connect() as connection:
            connection.execute("INSERT OR REPLACE INTO research_ingest_checks VALUES (?,?,?)", (url, now.isoformat(), document.document_id))
        return {"source_url": url, "status": "UNCHANGED" if previous and previous["document_id"] == document.document_id else "INDEXED",
                "document_id": document.document_id, "version": document.version}

    def ingest_manifest(self, path: Path, *, force=False) -> list[dict]:
        entries = json.loads(path.read_text(encoding="utf-8"))["documents"]
        results = []
        for raw in entries:
            try:
                results.append(self.ingest(IngestionEntry.model_validate(raw), force=force, base_path=path.parent))
            except Exception as exc:
                results.append({"source_url": str(raw.get("source_url", "")), "status": "ERROR", "error": type(exc).__name__})
        return results
