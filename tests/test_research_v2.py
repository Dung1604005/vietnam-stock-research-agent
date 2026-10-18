from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from stock_multi_agent.models.research import ResearchDocumentType
from stock_multi_agent.models.research import FetchedDocument
from stock_multi_agent.services.evidence_audit import audit_claim
from stock_multi_agent.services.research_ingestion import IngestionEntry, ResearchIngestionService
from stock_multi_agent.services.research_store import ResearchStore
from stock_multi_agent.services.rag_service import RAGService
from stock_multi_agent.services.source_registry import canonical_url, SourceRegistry


@pytest.fixture
def store():
    path = Path(__file__).resolve().parents[1] / ".runtime/test-artifacts/upgrade" / str(uuid4()) / "research.sqlite3"
    return ResearchStore(path)


def ingest(store, **changes):
    arguments = dict(text="FPT revenue increased 12% through technology services.", title="FPT revenue",
        document_type=ResearchDocumentType.EARNINGS_RELEASE, source_url="https://example.invalid/fpt",
        publisher="Evaluation fixture", published_at=datetime(2025, 1, 1, tzinfo=timezone.utc), tickers=["FPT"])
    arguments.update(changes)
    return store.ingest_text(**arguments)


def test_canonical_url_and_idempotent_ingest_preserve_original_retrieved_at(store):
    first = ingest(store, source_url="https://example.invalid/fpt?utm_source=tracker#page=1")
    second = ingest(store)
    assert first.document_id == second.document_id
    assert first.retrieved_at == second.retrieved_at
    with store._connect() as connection:
        assert connection.execute("SELECT count(*) FROM research_documents").fetchone()[0] == 1


def test_revisions_keep_history_and_as_of_hides_future_corrections(store):
    first = ingest(store)
    corrected = ingest(store, text="FPT revenue increased 15% after correction.", published_at=datetime(2025, 2, 1, tzinfo=timezone.utc))
    assert corrected.version == 2 and corrected.supersedes_id == first.document_id
    old = store.search("revenue", ticker="FPT", as_of=date(2025, 1, 15))
    latest = store.search("revenue", ticker="FPT", as_of=date(2025, 2, 15))
    assert [item.document_id for item in old.citations] == [first.document_id]
    assert [item.document_id for item in latest.citations] == [corrected.document_id]
    assert not store.search("revenue", ticker="FPT", as_of=date(2024, 12, 31)).citations


def test_reverting_content_creates_a_new_revision_instead_of_reviving_superseded_id(store):
    first = ingest(store)
    second = ingest(store, text="FPT revenue increased 15% after correction.", published_at=datetime(2025, 2, 1, tzinfo=timezone.utc))
    third = ingest(store, published_at=datetime(2025, 3, 1, tzinfo=timezone.utc))
    assert third.version == 3 and third.supersedes_id == second.document_id
    assert third.document_id != first.document_id
    assert store.search("revenue", ticker="FPT").citations[0].document_id == third.document_id


def test_ticker_and_type_filter_applied_before_candidate_limit(store):
    for index in range(35):
        ingest(store, text=f"ABC revenue forecast report growth {index}", source_url=f"https://example.invalid/abc/{index}", tickers=["ABC"])
    wanted = ingest(store)
    citations = store.search("revenue", ticker="FPT", document_types=[ResearchDocumentType.EARNINGS_RELEASE]).citations
    assert citations and all(item.document_id == wanted.document_id for item in citations)
    assert not store.search("revenue", ticker="FPT", document_types=[ResearchDocumentType.ANNUAL_REPORT]).citations


def test_numeric_grounding_checks_only_cited_sources(store):
    ingest(store)
    citation = store.search("revenue", ticker="FPT").citations[0]
    assert not audit_claim("Doanh thu FPT tăng 12%.", [citation])
    assert "unsupported_numeric_claim" in audit_claim("Doanh thu FPT tăng 99%.", [citation])
    assert "missing_evidence" in audit_claim("Tăng 12%", [])
    unit_source = citation.model_copy(update={"quote": "Doanh thu đạt 20 tỷ VND."})
    assert "unsupported_unit" in audit_claim("Doanh thu đạt 20 triệu VND.", [unit_source])


def test_rag_rejects_hallucinated_numbers_even_with_existing_source_id(store):
    ingest(store, title="FPT earnings Q3 2026", published_at=datetime(2026, 10, 1, tzinfo=timezone.utc))
    class Provider:
        def parse_structured(self, system, payload, output_model):
            return output_model.model_validate({"claims": [{"claim_vi": "Doanh thu tăng 99%.", "source_ids": ["S1"]}]})
    execution = RAGService(store, provider=Provider(), enabled=True).answer("revenue", ticker="FPT")
    assert execution.status == "SYNTHESIS_FAILED"


def test_rag_no_evidence_does_not_call_provider(store):
    class Provider:
        def parse_structured(self, *args):
            raise AssertionError("must not call LLM")
    execution = RAGService(store, provider=Provider(), enabled=True).answer("missing evidence")
    assert execution.status == "NO_EVIDENCE"
    assert execution.answer_contract.status == "INSUFFICIENT_EVIDENCE"


def test_manifest_ingestion_respects_permission_and_reingests_changed_local_file(store):
    path = store.database_path.parent / "report.md"
    path.write_text("FPT revenue increased 12%", encoding="utf-8")
    entry = IngestionEntry(source_url="https://example.invalid/local", title="Local fixture", publisher="Fixture",
        published_at=datetime(2025, 1, 1, tzinfo=timezone.utc), document_type=ResearchDocumentType.OTHER,
        usage_note="Author-created fixture", local_path=str(path))
    service = ResearchIngestionService(store)
    assert service.ingest(entry)["status"] == "SKIPPED_PERMISSION"
    entry.approved_full_text = True
    first = service.ingest(entry)
    assert first["status"] == "INDEXED"
    assert service.ingest(entry)["status"] == "UNCHANGED"
    path.write_text("FPT revenue increased 15%", encoding="utf-8")
    assert service.ingest(entry)["version"] == 2


def test_registry_does_not_match_spoofed_domain():
    registry = SourceRegistry()
    assert registry.resolve("https://www.hnx.vn/issuer").quality_tier == "T1"
    assert registry.resolve("https://hnx.vn.evil.invalid/issuer") is None
    assert canonical_url("https://EXAMPLE.invalid/report?b=2&utm_source=x&a=1#pdf") == "https://example.invalid/report?a=1&b=2"
    with pytest.raises(ValueError):
        canonical_url("https://user:pass@example.invalid/report")


def test_incremental_fetch_uses_ttl_and_does_not_fetch_undue_source(store):
    class Fetcher:
        calls = 0
        def fetch(self, url):
            self.calls += 1
            return FetchedDocument(requested_url=url, final_url=url, content_type="text/html", status_code=200,
                retrieved_at=datetime.now(timezone.utc), body="<h1>FPT report</h1><p>Revenue growth 12%.</p>")
    fetcher = Fetcher()
    entry = IngestionEntry(source_url="https://fpt.com.vn/approved-report", title="Fixture", publisher="FPT IR",
        published_at=datetime(2025, 1, 1, tzinfo=timezone.utc), document_type=ResearchDocumentType.OTHER,
        approved_full_text=True, usage_note="Explicit permission for fixture")
    service = ResearchIngestionService(store, fetcher=fetcher)
    assert service.ingest(entry)["status"] == "INDEXED"
    assert service.ingest(entry)["status"] == "CACHED"
    assert fetcher.calls == 1
    assert service.ingest(entry, force=True)["status"] == "UNCHANGED"
    assert fetcher.calls == 2


def test_vector_failure_preserves_lexical_retrieval(store):
    ingest(store)
    class OfflineEmbedding:
        model_id = "offline-embedding-v1"
        def embed(self, texts):
            raise RuntimeError("unavailable")
    store.embedding_provider = OfflineEmbedding()
    bundle = store.search("revenue", ticker="FPT")
    assert bundle.citations and bundle.citations[0].retrieval_modes == ["fts5"]


def test_missing_ticker_and_out_of_domain_text_abstain(store):
    ingest(store)
    assert not store.search("revenue", ticker="TCB").citations
    assert not store.search("Thời tiết Đà Lạt ngày mai có mưa không?").citations


def test_e5_is_lazy_and_uses_correct_prefixes_and_d_drive_cache(monkeypatch):
    import sys
    import types
    import numpy as np
    from stock_multi_agent.services.embeddings import E5EmbeddingProvider
    calls = []
    class Model:
        def __init__(self, name, **kwargs):
            calls.append(kwargs)
        def encode(self, texts, **kwargs):
            calls.append(texts)
            return np.array([[1.0, 0.0] for _ in texts])
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Model))
    monkeypatch.setenv("STOCK_AGENT_RAG_ALLOW_MODEL_DOWNLOAD", "false")
    provider = E5EmbeddingProvider()
    assert calls == []
    provider.embed(["doanh thu"])
    provider.embed_queries(["doanh thu?"])
    assert calls[0]["local_files_only"] is True
    import os
    if os.name == "nt":
        assert Path(calls[0]["cache_folder"]).drive.upper() == "D:"
    assert calls[1] == ["passage: doanh thu"]
    assert calls[2] == ["query: doanh thu?"]


def test_reinitialization_preserves_rows(store):
    # A second initialization preserves data and adds no duplicate embeddings.
    first = ingest(store)
    reopened = ResearchStore(store.database_path)
    assert reopened.search("revenue", ticker="FPT").citations[0].document_id == first.document_id
    with reopened._connect() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(research_documents)")}
        assert {"quality_tier", "version", "period_end", "supersedes_id"}.issubset(columns)


def test_real_legacy_schema_gets_additive_migration(store):
    legacy_path = store.database_path.parent / "legacy.sqlite3"
    with sqlite3.connect(legacy_path) as connection:
        connection.execute("CREATE TABLE research_documents (document_id TEXT PRIMARY KEY, title TEXT NOT NULL, document_type TEXT NOT NULL, source_url TEXT NOT NULL, publisher TEXT NOT NULL, published_at TEXT NOT NULL, retrieved_at TEXT NOT NULL, content_sha256 TEXT NOT NULL, tickers_json TEXT NOT NULL, language TEXT NOT NULL)")
        connection.execute("INSERT INTO research_documents VALUES (?,?,?,?,?,?,?,?,?,?)", (
            "a" * 64, "Legacy", "other", "https://example.invalid/legacy", "Fixture", "2025-01-01T00:00:00+00:00",
            "2025-01-01T00:00:00+00:00", "b" * 64, '["FPT"]', "vi"))
    migrated = ResearchStore(legacy_path)
    with migrated._connect() as connection:
        row = connection.execute("SELECT * FROM research_documents").fetchone()
        assert row["title"] == "Legacy" and row["version"] == 1 and row["quality_tier"] == "T4"


def test_chunker_preserves_section_boundaries_and_tables():
    text = "# Revenue\n\n" + "technology services " * 35 + "\n\n# Table\n\nYear | Profit\n2025 | 15\n"
    chunks = ResearchStore._chunks("a" * 64, text, chunk_chars=500, overlap_chars=60)
    assert any("# Table" in item.text and "2025 | 15" in item.text for item in chunks)
