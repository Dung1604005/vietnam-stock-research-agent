from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from stock_multi_agent.models.knowledge import (
    KnowledgeGraphContext,
    KnowledgeGraphNode,
    KnowledgeGraphPath,
    KnowledgeNodeType,
    KnowledgeRelationType,
    KnowledgeSourceType,
    VerificationStatus,
)
from stock_multi_agent.services.knowledge_graph import Neo4jKnowledgeGraph
from stock_multi_agent.services.knowledge_store import KnowledgeStore


TEST_ROOT = Path(__file__).resolve().parents[1] / ".runtime" / "test-artifacts" / "knowledge-graph"


class FakeResult(list):
    def consume(self):
        return None

    def single(self):
        return self[0] if self else None


class FakeSession:
    def __init__(self, queries: list[tuple[str, dict[str, object]]]) -> None:
        self.queries = queries

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def run(self, statement: str, parameters_: dict[str, object] | None = None, **parameters: object):
        parameters = {**(parameters_ or {}), **parameters}
        self.queries.append((statement, parameters))
        if "RETURN root, nodes(path)" in statement:
            return FakeResult([{
                "root": {
                    "id": "concept:roe", "type": "CONCEPT", "label": "roe",
                    "aliases": ["roe", "return on equity"], "normalized_label": "roe",
                    "namespace": "stock-kb",
                },
                "path_nodes": [
                    {
                        "id": "concept:roe", "type": "CONCEPT", "label": "roe",
                        "aliases": ["roe"], "normalized_label": "roe", "namespace": "stock-kb",
                    },
                    {
                        "id": "document:doc-1234567890123456", "type": "DOCUMENT", "label": "ROE note",
                        "aliases": ["roe note"], "normalized_label": "roe note", "namespace": "stock-kb",
                        "document_id": "doc-1234567890123456",
                    },
                ],
                "path_relationships": [{
                    "kind": "MENTIONS", "weight": 1.0,
                    "evidence_document_id": "doc-1234567890123456", "evidence_chunk_id": "chunk-1",
                }],
                "alias_score": 3,
            }])
        if "count(DISTINCT n)" in statement:
            return FakeResult([{"nodes": 8, "edges": 12, "documents": 1}])
        return FakeResult()


class FakeDriver:
    def __init__(self) -> None:
        self.queries: list[tuple[str, dict[str, object]]] = []

    def verify_connectivity(self) -> None:
        return None

    def session(self, **_kwargs: object) -> FakeSession:
        return FakeSession(self.queries)

    def close(self) -> None:
        return None


def test_neo4j_sync_and_two_hop_context_are_provenance_aware() -> None:
    driver = FakeDriver()
    graph = Neo4jKnowledgeGraph(driver=driver, enabled=True)
    status = graph.sync([{
        "document_id": "doc-1234567890123456", "title": "ROE note", "publisher": "Example",
        "version": "1", "topics": ["financial-ratios"],
        "chunks": [{"chunk_id": "chunk-1", "heading": "ROE", "text": "ROE là lợi nhuận trên vốn."}],
    }], rebuild=True)

    context = graph.expand("ROE là gì?", max_hops=2)

    assert status == {"available": True, "nodes": 8, "edges": 12, "documents": 1}
    assert context.related_document_ids == ["doc-1234567890123456"]
    assert context.paths[0].evidence_chunk_id == "chunk-1"
    assert any("MERGE (d:KnowledgeEntity" in query for query, _ in driver.queries)
    assert any("RELATES*1..2" in query for query, _ in driver.queries)


class GraphCandidateStub:
    def __init__(self) -> None:
        self.document_id = ""

    def sync(self, records, *, rebuild=False):  # type: ignore[no-untyped-def]
        del records, rebuild
        return {"available": True, "documents": 1}

    def expand(self, query: str, *, max_hops: int, limit: int) -> KnowledgeGraphContext:
        del max_hops, limit
        return KnowledgeGraphContext(
            query=query,
            matched_nodes=[KnowledgeGraphNode(
                node_id="topic:financial-ratios", node_type=KnowledgeNodeType.TOPIC,
                label="financial ratios", aliases=["financial ratios"],
            )],
            paths=[KnowledgeGraphPath(
                node_ids=["topic:financial-ratios", f"document:{self.document_id}"],
                labels=["financial ratios", "ROE note"], relations=[KnowledgeRelationType.HAS_TOPIC],
                score=1.0, evidence_document_id=self.document_id,
            )],
            related_document_ids=[self.document_id],
            retrieval_note="Neo4j graph candidate",
        )


def test_neo4j_document_candidates_are_fused_into_kb_retrieval() -> None:
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    graph = GraphCandidateStub()
    store = KnowledgeStore(TEST_ROOT / f"{uuid4().hex}.sqlite3", knowledge_graph=graph)  # type: ignore[arg-type]
    document = store.ingest_text(
        text="# Hiệu quả vốn\n\nLợi nhuận chia cho vốn chủ sở hữu bình quân.",
        title="ROE note", publisher="Example", source_url="https://example.org/roe-graph",
        source_type=KnowledgeSourceType.OPEN_EDUCATION,
        usage_note="Test", verification_status=VerificationStatus.VERIFIED,
        topics=["financial-ratios"], version="1",
        published_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    graph.document_id = document.document_id

    result = store.search("financial ratios", limit=2)

    assert result.citations
    assert result.citations[0].document_id == document.document_id
    assert "graph" in result.citations[0].retrieval_modes
    assert result.retrieval_mode.endswith("+neo4j-rrf")
