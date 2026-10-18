from __future__ import annotations

import os
from pathlib import Path

from stock_multi_agent.models.chat import AssistantResponse
from stock_multi_agent.models.knowledge import KnowledgeSearchResult
from stock_multi_agent.services.knowledge_store import KnowledgeIngestionError, KnowledgeStore


class KnowledgeService:
    """Deterministic knowledge retrieval and evidence formatting; not an LLM agent."""

    EMPTY_MESSAGE = "Knowledge Base chưa có đủ tài liệu đã xác minh cho nội dung này."

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        *,
        seed_manifest: Path | None = None,
        bootstrap: bool = True,
    ) -> None:
        self.store = store or KnowledgeStore()
        self.enabled = os.getenv("STOCK_AGENT_KNOWLEDGE_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
        self.seed_manifest = seed_manifest or Path(__file__).resolve().parents[3] / "knowledge" / "sources.json"
        self.bootstrap_warning: str | None = None
        self.graph_sync_status: dict[str, int | str | bool] = {"available": False, "documents": 0}
        if bootstrap and self.enabled:
            self.ensure_seeded()
            # Startup/readiness must not perform optional network connections.
            # Explicit sync remains available through the existing CLI/library.
            if os.getenv('STOCK_AGENT_KNOWLEDGE_GRAPH_SYNC_ON_STARTUP', 'false').lower() == 'true':
                self.graph_sync_status = self.store.sync_knowledge_graph()

    def ensure_seeded(self) -> None:
        if self.store.count_documents() or not self.seed_manifest.is_file():
            return
        try:
            self.store.ingest_manifest(self.seed_manifest)
        except (KnowledgeIngestionError, OSError, ValueError) as exc:
            self.bootstrap_warning = f"Knowledge seed ingestion failed: {exc}"

    def retrieve(self, query: str, *, limit: int = 3) -> KnowledgeSearchResult:
        if not self.enabled:
            return KnowledgeSearchResult(query=query, retrieval_note="Local Knowledge Base is disabled by configuration.")
        return self.store.search(query, limit=limit)

    def answer(self, query: str, *, current_rule: bool = False) -> AssistantResponse:
        result = self.retrieve(query, limit=2)
        if not result.citations:
            warnings = [result.retrieval_note]
            if self.bootstrap_warning:
                warnings.append(self.bootstrap_warning)
            return AssistantResponse(
                message=(
                    f"{self.EMPTY_MESSAGE}\n\n"
                    "Bạn có thể yêu cầu tìm nguồn chính thức trên Internet để bổ sung tài liệu có provenance."
                ),
                warnings=warnings,
                metadata={
                    "knowledge_result": result.model_dump(mode="json"),
                    "retrieval_mode": "knowledge_none",
                    "evidence_origin": "Local KB",
                },
            )

        sections: list[str] = ["### Kiến thức chứng khoán"]
        seen_quotes: set[str] = set()
        for index, citation in enumerate(result.citations, 1):
            quote = citation.quote.strip()
            if quote in seen_quotes:
                continue
            seen_quotes.add(quote)
            sections.append(f"{quote}\n\n*Bằng chứng: [{index}]*")

        graph_context = result.graph_context
        if graph_context and graph_context.paths:
            relation_lines: list[str] = []
            for path in graph_context.paths[:3]:
                parts = [path.labels[0]]
                for relation, label in zip(path.relations, path.labels[1:]):
                    parts.extend([f"--{relation.value}-->", label])
                relation_lines.append(f"- {' '.join(parts)}")
            sections.append("#### Quan hệ từ Knowledge Graph\n\n" + "\n".join(relation_lines))

        source_lines = []
        warnings: list[str] = []
        for index, citation in enumerate(result.citations, 1):
            published = citation.published_at.date().isoformat() if citation.published_at else "không được nguồn cung cấp"
            effective = citation.effective_at.date().isoformat() if citation.effective_at else "không được nguồn cung cấp"
            source_lines.append(
                f"[{index}] [{citation.title}]({citation.source_url}) — {citation.publisher}; "
                f"published: {published}; effective: {effective}; retrieved: {citation.retrieved_at.date().isoformat()}."
            )
            if citation.freshness_warning:
                warnings.append(citation.freshness_warning)
        sections.append("#### Nguồn tham khảo\n\n" + "\n\n".join(source_lines))
        if current_rule or result.requires_freshness_check:
            sections.append(
                "*Lưu ý freshness: đây là nội dung có thể thay đổi theo quy định. "
                "Hãy đối chiếu phiên bản hiện hành tại liên kết chính thức trước khi sử dụng.*"
            )
        if self.bootstrap_warning:
            warnings.append(self.bootstrap_warning)

        return AssistantResponse(
            message="\n\n".join(sections),
            sources=list(dict.fromkeys(str(item.source_url) for item in result.citations)),
            warnings=list(dict.fromkeys(warnings)),
            metadata={
                "knowledge_result": result.model_dump(mode="json"),
                "knowledge_citations": [item.model_dump(mode="json") for item in result.citations],
                "retrieval_mode": "knowledge_local",
                "knowledge_retrieval_mode": result.retrieval_mode,
                "evidence_origin": "Local KB",
                "freshness_checked": bool(current_rule or result.requires_freshness_check),
                "knowledge_graph": (
                    result.graph_context.model_dump(mode="json") if result.graph_context else None
                ),
                "knowledge_graph_sync": self.graph_sync_status,
            },
        )

