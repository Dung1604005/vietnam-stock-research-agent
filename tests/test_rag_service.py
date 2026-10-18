from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stock_multi_agent.models.research import ResearchDocumentType
from stock_multi_agent.services.rag_service import RAGService
from stock_multi_agent.services.research_store import ResearchStore


TEST_ROOT = Path(__file__).resolve().parents[1] / ".runtime" / "test-artifacts" / "rag"


class TopicEmbeddingProvider:
    model_id = "test-topic-v1"

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        vectors = []
        for text in texts:
            normalized = text.casefold()
            if any(term in normalized for term in ("revenue", "growth", "doanh thu", "tăng trưởng")):
                vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif any(term in normalized for term in ("debt", "risk", "nợ", "rủi ro")):
                vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            vectors.append(vector)
        return vectors


class GroundedProvider:
    def parse_structured(self, system, payload, output_model):
        assert "SOURCE" in system
        assert payload["sources"][0]["source_id"] == "S1"
        return output_model.model_validate({
            "claims": [{"claim_vi": "Doanh thu mảng công nghệ được báo cáo tăng trưởng.", "source_ids": ["S1"]}],
            "limitations": "Corpus thử nghiệm chỉ có một tài liệu.",
        })


def build_store(name: str) -> ResearchStore:
    store = ResearchStore(TEST_ROOT / ("period-" + name), embedding_provider=TopicEmbeddingProvider())
    store.ingest_text(
        text="FPT Q3 2026 reported revenue growth from overseas technology services.",
        title="FPT earnings Q3 2026", document_type=ResearchDocumentType.EARNINGS_RELEASE,
        source_url="https://example.com/fpt-growth", publisher="FPT IR",
        published_at=datetime(2026, 10, 1, tzinfo=timezone.utc), tickers=["FPT"],
    )
    store.ingest_text(
        text="ABC reported higher debt and refinancing risk.",
        title="ABC risk", document_type=ResearchDocumentType.OTHER,
        source_url="https://example.com/abc-risk", publisher="ABC IR",
        published_at=datetime(2026, 10, 1, tzinfo=timezone.utc), tickers=["ABC"],
    )
    return store


def test_vector_similarity_retrieves_ticker_filtered_chunk() -> None:
    store = build_store("vector.sqlite3")
    bundle = store.search_vector("tăng trưởng doanh thu", ticker="FPT")
    assert bundle.citations
    assert bundle.citations[0].title == "FPT earnings Q3 2026"
    assert bundle.citations[0].retrieval_score == 1.0
    assert bundle.citations[0].retrieval_modes == ["vector"]
    assert store.reindex_embeddings() == 2


def test_rag_synthesizes_only_citation_grounded_claims() -> None:
    store = build_store("synthesis.sqlite3")
    execution = RAGService(store, provider=GroundedProvider(), enabled=True).answer(
        "Tăng trưởng doanh thu FPT thế nào?", ticker="FPT",
    )
    assert execution.status == "GROUNDED"
    assert "Trả lời từ RAG" in execution.message
    assert "[1]" in execution.message
    assert execution.cited_chunk_ids == [execution.bundle.citations[0].chunk_id]


def test_rag_rejects_hallucinated_citation_ids() -> None:
    class HallucinatingProvider:
        def parse_structured(self, system, payload, output_model):
            del system, payload
            return output_model.model_validate({
                "claims": [{"claim_vi": "Một claim không có nguồn thật.", "source_ids": ["S999"]}],
                "limitations": "",
            })

    execution = RAGService(
        build_store("hallucination.sqlite3"), provider=HallucinatingProvider(), enabled=True,
    ).answer("Tăng trưởng doanh thu FPT", ticker="FPT")
    assert execution.status == "SYNTHESIS_FAILED"
    assert "citation hợp lệ" in execution.message


def test_rag_rejects_non_vietnamese_generated_claims() -> None:
    class WrongLanguageProvider:
        def parse_structured(self, system, payload, output_model):
            del system, payload
            if output_model.__name__ == "_VietnameseText":
                return output_model.model_validate({"text_vi": "这是中文。"})
            return output_model.model_validate({
                "claims": [{"claim_vi": "This is an English claim from the report.", "source_ids": ["S1"]}],
                "limitations": "",
            })

    execution = RAGService(
        build_store("wrong-language.sqlite3"), provider=WrongLanguageProvider(), enabled=True,
    ).answer("Tăng trưởng doanh thu FPT", ticker="FPT")
    assert execution.status == "SYNTHESIS_FAILED"
