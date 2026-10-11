from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from stock_multi_agent.chat.query_parser import QueryParser
from stock_multi_agent.chat.router import IntentRouter, RouteStep
from stock_multi_agent.models.chat import ConversationState, QueryIntent
from stock_multi_agent.models.knowledge import KnowledgeSourceType, VerificationStatus
from stock_multi_agent.services.chat_service import ChatService
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.knowledge_service import KnowledgeService
from stock_multi_agent.services.knowledge_store import KnowledgeIngestionError, KnowledgeStore
from conftest import InMemoryProvider


TEST_ROOT = Path(__file__).resolve().parents[1] / ".runtime" / "test-artifacts" / "knowledge"


def new_store() -> KnowledgeStore:
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    return KnowledgeStore(TEST_ROOT / f"{uuid4().hex}.sqlite3")


def ingest(store: KnowledgeStore, text: str, *, url: str = "https://example.org/knowledge", version: str = "1"):
    return store.ingest_text(
        text=text,
        title="Verified educational note",
        publisher="Example Education",
        source_url=url,
        source_type=KnowledgeSourceType.OPEN_EDUCATION,
        usage_note="Author-created summary for testing.",
        verification_status=VerificationStatus.VERIFIED,
        topics=["financial-ratios"],
        version=version,
        published_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def test_schema_migration_and_fts_are_initialized() -> None:
    store = new_store()
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == KnowledgeStore.SCHEMA_VERSION
        assert connection.execute("SELECT COUNT(*) FROM kb_schema_migrations").fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='knowledge_chunks_fts'"
        ).fetchone()[0] == 1


@pytest.mark.parametrize(
    ("suffix", "payload", "expected"),
    [
        (".md", "# P/E\n\nP/E là tỷ lệ giá trên lợi nhuận.", "P/E là tỷ lệ"),
        (".txt", "ROE là lợi nhuận trên vốn chủ sở hữu.", "ROE là lợi nhuận"),
        (".html", "<html><script>ignore()</script><p>RSI đo động lượng.</p></html>", "RSI đo động lượng"),
    ],
)
def test_read_markdown_txt_and_html(suffix: str, payload: str, expected: str) -> None:
    path = TEST_ROOT / f"source-{uuid4().hex}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    extracted = KnowledgeStore.read_document(path)
    assert expected in extracted
    assert "ignore" not in extracted


def test_repeated_header_footer_lines_are_removed() -> None:
    path = TEST_ROOT / f"source-{uuid4().hex}.txt"
    path.write_text(
        "Báo cáo giáo dục\nNội dung trang một.\nBáo cáo giáo dục\nNội dung trang hai.\n"
        "Báo cáo giáo dục\nNội dung trang ba.",
        encoding="utf-8",
    )
    extracted = KnowledgeStore.read_document(path)
    assert "Báo cáo giáo dục" not in extracted
    assert "Nội dung trang ba" in extracted


def test_read_text_pdf_when_optional_parser_is_installed() -> None:
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    path = TEST_ROOT / f"source-{uuid4().hex}.pdf"
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 100 Td (RSI measures momentum.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as handle:
        writer.write(handle)
    assert "RSI measures momentum" in KnowledgeStore.read_document(path)
    manifest = path.with_suffix(".json")
    manifest.write_text(json.dumps({"documents": [{
        "path": path.name, "title": "PDF RSI note", "publisher": "Example Education",
        "source_url": "https://example.org/pdf-rsi", "source_type": "open_education",
        "usage_note": "Text PDF test", "verification_status": "VERIFIED",
        "topics": ["technical-analysis"], "version": "1",
    }]}), encoding="utf-8")
    store = new_store()
    assert store.ingest_manifest(manifest)
    assert store.search("RSI momentum").citations


def test_duplicate_content_and_url_versioning() -> None:
    store = new_store()
    first = ingest(store, "# ROE\n\nROE = lợi nhuận / vốn chủ sở hữu.")
    duplicate = ingest(store, "# ROE\n\nROE = lợi nhuận / vốn chủ sở hữu.", url="https://mirror.example.org/roe")
    assert duplicate.document_id == first.document_id
    assert store.count_documents() == 1
    second = ingest(store, "# ROE\n\nROE dùng vốn chủ sở hữu bình quân.", version="2")
    assert second.document_id != first.document_id
    assert store.count_documents() == 1


def test_verified_ingestion_requires_real_provenance_fields() -> None:
    store = new_store()
    with pytest.raises((KnowledgeIngestionError, ValueError)):
        store.ingest_text(
            text="knowledge", title="Missing publisher", publisher="", source_url="https://example.org/x",
            source_type=KnowledgeSourceType.OTHER, usage_note="test",
            verification_status=VerificationStatus.VERIFIED, topics=["test"], version="1",
        )


def test_chunk_provenance_and_diacriticless_fts_retrieval() -> None:
    store = new_store()
    document = ingest(store, "# ROE\n\nROE là lợi nhuận sau thuế chia vốn chủ sở hữu bình quân.")
    result = store.search("roe la gi")
    assert result.citations
    citation = result.citations[0]
    assert citation.document_id == document.document_id
    assert citation.chunk_id
    assert citation.heading == "ROE"
    assert str(citation.source_url) == "https://example.org/knowledge"
    assert citation.retrieved_at.tzinfo is not None


class FakeSemanticRetriever:
    def __init__(self, chunk_id: str) -> None:
        self.chunk_id = chunk_id

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        return [(self.chunk_id, 0.9)]


def test_hybrid_ranking_can_add_optional_semantic_signal() -> None:
    store = new_store()
    document = ingest(store, "# ROE\n\nROE đo hiệu quả dùng vốn chủ sở hữu.")
    with sqlite3.connect(store.database_path) as connection:
        chunk_id = connection.execute(
            "SELECT chunk_id FROM knowledge_chunks WHERE document_id=?", (document.document_id,)
        ).fetchone()[0]
    store.semantic_retriever = FakeSemanticRetriever(chunk_id)
    result = store.search("hiệu quả vốn")
    assert result.citations[0].retrieval_modes == ["bm25", "semantic"]
    assert result.retrieval_mode.endswith("semantic-rrf")


def test_exact_finance_term_and_lexical_only_fallback() -> None:
    store = new_store()
    ingest(store, "# P/E\n\nP/E = giá cổ phiếu chia EPS.")
    result = store.search("P/E là gì?")
    assert result.citations
    assert "exact" in result.citations[0].retrieval_modes
    assert result.retrieval_mode == "exact+bm25-rrf"


def test_manifest_rejects_verified_document_without_provenance() -> None:
    directory = TEST_ROOT / uuid4().hex
    directory.mkdir(parents=True)
    (directory / "note.md").write_text("# Note\n\nText", encoding="utf-8")
    (directory / "sources.json").write_text(json.dumps({"documents": [{
        "path": "note.md", "title": "Note", "publisher": "",
        "source_url": "https://example.org/note", "source_type": "other",
        "usage_note": "test", "verification_status": "VERIFIED", "topics": ["test"], "version": "1",
    }]}), encoding="utf-8")
    with pytest.raises(ValueError):
        new_store().ingest_manifest(directory / "sources.json")


def test_theory_and_current_rule_routing_do_not_request_market_data() -> None:
    parser = QueryParser(use_configured_provider=False)
    theory = parser.parse("P/E là gì?")
    current = parser.parse("Biên độ HOSE hiện tại là bao nhiêu?")
    assert theory.intent is QueryIntent.THEORY
    assert current.intent is QueryIntent.CURRENT_RULE
    assert IntentRouter().plan(theory).steps == [RouteStep.KNOWLEDGE]
    assert IntentRouter().plan(current).steps == [RouteStep.KNOWLEDGE]


def test_mixed_routing_combines_knowledge_and_real_data_steps() -> None:
    parsed = QueryParser(use_configured_provider=False).parse("ROE của FPT có tốt không?")
    assert parsed.intent is QueryIntent.MIXED
    assert IntentRouter().plan(parsed).steps == [RouteStep.KNOWLEDGE, RouteStep.DATA, RouteStep.FUNDAMENTAL]


def test_mixed_chat_keeps_knowledge_and_market_provenance_separate(market_data) -> None:
    chat_service = ChatService(analysis_service=AnalysisService(provider=InMemoryProvider(market_data)))
    response = chat_service.handle("ROE của FPT có tốt không?", ConversationState())
    assert response.metadata["routing_plan"] == ["KNOWLEDGE", "DATA", "FUNDAMENTAL"]
    assert response.metadata["mixed_components"] == ["knowledge", "fundamental"]
    assert response.metadata["knowledge_citations"]
    assert response.metadata["knowledge_does_not_control_recommendation"] is True
    assert response.source_details  # deterministic fixture represents the real-data channel
    assert "Áp dụng vào dữ liệu thật · FPT" in response.message


def test_current_rule_answer_has_citation_and_freshness_warning() -> None:
    service = KnowledgeService()
    response = service.answer("ATO khác ATC thế nào?", current_rule=True)
    assert response.metadata["retrieval_mode"] == "knowledge_local"
    assert response.metadata["knowledge_citations"]
    assert response.metadata["freshness_checked"] is True
    assert response.warnings


def test_empty_kb_refuses_instead_of_hallucinating() -> None:
    service = KnowledgeService(store=new_store(), bootstrap=False)
    response = service.answer("Một khái niệm hoàn toàn không có trong kho")
    assert response.message.startswith(KnowledgeService.EMPTY_MESSAGE)
    assert response.metadata["knowledge_result"]["citations"] == []


def test_theory_chat_uses_local_kb_and_has_no_research_execution() -> None:
    class FailIfCalledAnalysis:
        def __getattr__(self, name: str):
            raise AssertionError(f"Market/analysis service must not be touched for THEORY: {name}")

    class FailIfCalledWebResearch:
        def search(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("Web research must not be called when verified local KB evidence exists")

    service = ChatService(
        analysis_service=FailIfCalledAnalysis(),  # type: ignore[arg-type]
        web_research=FailIfCalledWebResearch(),  # type: ignore[arg-type]
        knowledge_service=KnowledgeService(),
    )
    response = service.handle("P/E là gì?", ConversationState())
    assert response.metadata["routing_plan"] == ["KNOWLEDGE"]
    assert response.metadata["retrieval_mode"] == "knowledge_local"
    assert "research_execution" not in response.metadata
    assert response.sources

