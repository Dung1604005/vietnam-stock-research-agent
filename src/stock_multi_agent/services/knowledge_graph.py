from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from typing import Any, Protocol

from stock_multi_agent.models.knowledge import (
    KnowledgeGraphContext,
    KnowledgeGraphNode,
    KnowledgeGraphPath,
    KnowledgeNodeType,
    KnowledgeRelationType,
)


class Neo4jDriverProtocol(Protocol):
    def verify_connectivity(self) -> None: ...
    def session(self, **kwargs: object): ...  # type: ignore[no-untyped-def]
    def close(self) -> None: ...


class Neo4jKnowledgeGraph:
    """Neo4j-backed graph projection of the provenance-first local KB."""

    _CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
        "pe": ("p/e", "pe", "price earnings", "price to earnings"),
        "pb": ("p/b", "pb", "price to book"),
        "roe": ("roe", "return on equity", "lợi nhuận trên vốn chủ sở hữu"),
        "roa": ("roa", "return on assets", "lợi nhuận trên tài sản"),
        "eps": ("eps", "earnings per share", "lợi nhuận mỗi cổ phiếu"),
        "rsi": ("rsi", "relative strength index", "chỉ số sức mạnh tương đối"),
        "macd": ("macd", "moving average convergence divergence"),
        "sma": ("sma", "simple moving average", "trung bình động đơn giản"),
        "ema": ("ema", "exponential moving average", "trung bình động hàm mũ"),
        "atr": ("atr", "average true range"),
        "bollinger-bands": ("bollinger", "bollinger bands", "dải bollinger"),
        "dcf": ("dcf", "discounted cash flow", "chiết khấu dòng tiền"),
        "ev-ebitda": ("ev ebitda", "enterprise value ebitda"),
        "capm": ("capm", "capital asset pricing model"),
        "sharpe-ratio": ("sharpe", "sharpe ratio"),
        "volatility": ("volatility", "biến động", "độ biến động"),
        "drawdown": ("drawdown", "mức sụt giảm", "sụt giảm tối đa"),
        "liquidity": ("liquidity", "thanh khoản"),
        "diversification": ("diversification", "đa dạng hóa", "đa dạng hoá"),
        "revenue": ("revenue", "doanh thu"),
        "profit": ("profit", "lợi nhuận"),
        "cash-flow": ("cash flow", "dòng tiền", "lưu chuyển tiền"),
        "balance-sheet": ("balance sheet", "bảng cân đối kế toán"),
        "income-statement": ("income statement", "kết quả kinh doanh"),
        "dividend": ("dividend", "cổ tức"),
        "stock-split": ("stock split", "chia tách cổ phiếu"),
        "rights-offering": ("rights offering", "quyền mua", "phát hành thêm"),
        "ato": ("ato", "at the open"),
        "atc": ("atc", "at the close"),
        "price-limit": ("price limit", "biên độ giá", "giá trần", "giá sàn"),
        "settlement": ("settlement", "thanh toán", "t+2"),
        "backtesting": ("backtest", "backtesting", "kiểm định quá khứ"),
        "data-leakage": ("data leakage", "rò rỉ dữ liệu", "look ahead"),
    }

    def __init__(self, driver: Neo4jDriverProtocol | None = None, *, enabled: bool | None = None) -> None:
        self.enabled = self._env_bool("STOCK_AGENT_KNOWLEDGE_GRAPH_ENABLED", True) if enabled is None else enabled
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
        self.user = os.getenv("NEO4J_USER", "neo4j").strip()
        self.password = os.getenv("NEO4J_PASSWORD", "").strip()
        self.database = os.getenv("NEO4J_DATABASE", "neo4j").strip() or "neo4j"
        self.last_error: str | None = None
        self.driver = driver
        if self.enabled and self.driver is None and self.password:
            try:
                from neo4j import GraphDatabase

                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    connection_timeout=2.0,
                    max_connection_pool_size=5,
                )
            except (ImportError, OSError, ValueError) as exc:
                self.last_error = f"Neo4j driver initialization failed: {exc}"

    @property
    def available(self) -> bool:
        if not self.enabled or self.driver is None:
            return False
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as exc:  # Neo4j exposes several transport/auth exception types.
            self.last_error = f"Neo4j is unavailable: {exc}"
            return False

    def initialize_schema(self) -> bool:
        if not self.available:
            return False
        statements = [
            "CREATE CONSTRAINT knowledge_entity_id IF NOT EXISTS FOR (n:KnowledgeEntity) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX knowledge_entity_type IF NOT EXISTS FOR (n:KnowledgeEntity) ON (n.type)",
            "CREATE INDEX knowledge_entity_label IF NOT EXISTS FOR (n:KnowledgeEntity) ON (n.normalized_label)",
        ]
        try:
            with self.driver.session(database=self.database) as session:  # type: ignore[union-attr]
                for statement in statements:
                    session.run(statement).consume()
            return True
        except Exception as exc:
            self.last_error = f"Neo4j schema initialization failed: {exc}"
            return False

    def sync(self, records: list[dict[str, object]], *, rebuild: bool = False) -> dict[str, int | str | bool]:
        if not self.initialize_schema():
            return {"available": False, "documents": 0, "error": self.last_error or "Neo4j unavailable"}
        try:
            with self.driver.session(database=self.database) as session:  # type: ignore[union-attr]
                if rebuild:
                    session.run("MATCH (n:KnowledgeEntity {namespace: $namespace}) DETACH DELETE n", namespace="stock-kb").consume()
                for record in records:
                    self._sync_record(session, record)
            stats = self.stats()
            return {"available": True, **stats}
        except Exception as exc:
            self.last_error = f"Neo4j sync failed: {exc}"
            return {"available": False, "documents": 0, "error": self.last_error}

    def expand(self, query: str, *, max_hops: int = 2, limit: int = 20) -> KnowledgeGraphContext:
        normalized = self.normalize(query)
        if not normalized:
            return KnowledgeGraphContext(query=query, retrieval_note="Knowledge Graph received no searchable terms.")
        if not self.available:
            return KnowledgeGraphContext(
                query=query,
                retrieval_note=self.last_error or "Neo4j Knowledge Graph is disabled or not configured.",
            )
        hop_count = 1 if max_hops <= 1 else 2
        cypher = f"""
        MATCH (root:KnowledgeEntity {{namespace: $namespace}})
        UNWIND root.aliases AS alias
        WITH root, alias
        WHERE size(alias) > 1 AND $query CONTAINS alias
        WITH root, max(size(alias)) AS alias_score
        ORDER BY alias_score DESC
        LIMIT 6
        MATCH path=(root)-[:RELATES*1..{hop_count}]-(neighbor:KnowledgeEntity {{namespace: $namespace}})
        WHERE all(rel IN relationships(path) WHERE rel.kind <> 'PUBLISHED_BY' OR length(path) = 1)
        RETURN root, nodes(path) AS path_nodes, relationships(path) AS path_relationships,
               alias_score
        ORDER BY alias_score DESC, length(path) ASC
        LIMIT $limit
        """
        try:
            with self.driver.session(database=self.database) as session:  # type: ignore[union-attr]
                rows = list(session.run(cypher, {
                    "namespace": "stock-kb",
                    "query": normalized,
                    "limit": max(1, min(limit, 50)),
                }))
        except Exception as exc:
            self.last_error = f"Neo4j traversal failed: {exc}"
            return KnowledgeGraphContext(query=query, retrieval_note=self.last_error)

        matched: dict[str, KnowledgeGraphNode] = {}
        paths: list[KnowledgeGraphPath] = []
        documents: dict[str, float] = {}
        expansions: dict[str, float] = {}
        for row in rows:
            root = self._properties(row["root"])
            root_model = self._node_model(root)
            matched[root_model.node_id] = root_model
            path_nodes = [self._properties(node) for node in row["path_nodes"]]
            path_relationships = [self._properties(rel) for rel in row["path_relationships"]]
            if len(path_nodes) < 2 or not path_relationships:
                continue
            weight = 1.0
            relations: list[KnowledgeRelationType] = []
            for relationship in path_relationships:
                weight *= float(relationship.get("weight", 0.5))
                relations.append(KnowledgeRelationType(str(relationship["kind"])))
            score = float(row.get("alias_score", 1.0)) * weight / len(relations)
            last_relationship = path_relationships[-1]
            path = KnowledgeGraphPath(
                node_ids=[str(node["id"]) for node in path_nodes],
                labels=[str(node["label"]) for node in path_nodes],
                relations=relations,
                score=max(0.0, score),
                evidence_document_id=self._optional_text(last_relationship.get("evidence_document_id")),
                evidence_chunk_id=self._optional_text(last_relationship.get("evidence_chunk_id")),
            )
            paths.append(path)
            endpoint = path_nodes[-1]
            endpoint_type = str(endpoint["type"])
            if endpoint_type == KnowledgeNodeType.DOCUMENT.value:
                document_id = str(endpoint.get("document_id") or str(endpoint["id"]).removeprefix("document:"))
                documents[document_id] = max(documents.get(document_id, 0.0), score)
            elif endpoint_type in {
                KnowledgeNodeType.CONCEPT.value, KnowledgeNodeType.TOPIC.value, KnowledgeNodeType.SECTION.value,
            } and str(endpoint["id"]) not in matched:
                label = str(endpoint["label"])
                expansions[label] = max(expansions.get(label, 0.0), score)

        paths.sort(key=lambda item: (-item.score, len(item.relations), item.labels))
        ordered_documents = [item[0] for item in sorted(documents.items(), key=lambda item: (-item[1], item[0]))]
        ordered_expansions = [item[0] for item in sorted(expansions.items(), key=lambda item: (-item[1], item[0]))]
        return KnowledgeGraphContext(
            query=query,
            matched_nodes=list(matched.values()),
            paths=paths[:limit],
            related_document_ids=ordered_documents[:limit],
            expansion_terms=ordered_expansions[:12],
            retrieval_note=(
                f"Neo4j matched {len(matched)} node(s), traversed up to {hop_count} hop(s), "
                f"and found {len(ordered_documents)} related document(s)."
                if matched else "No Neo4j entity matched this query."
            ),
        )

    def stats(self) -> dict[str, int]:
        if not self.available:
            return {"nodes": 0, "edges": 0, "documents": 0}
        query = """
        MATCH (n:KnowledgeEntity {namespace: $namespace})
        OPTIONAL MATCH (n)-[r:RELATES]->()
        RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS edges,
               count(DISTINCT CASE WHEN n.type='DOCUMENT' THEN n END) AS documents
        """
        with self.driver.session(database=self.database) as session:  # type: ignore[union-attr]
            row = session.run(query, namespace="stock-kb").single()
        return {
            "nodes": int(row["nodes"]), "edges": int(row["edges"]), "documents": int(row["documents"]),
        }

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()

    def _sync_record(self, session: Any, record: dict[str, object]) -> None:
        document_id = str(record["document_id"])
        document_node = self._node_payload(
            f"document:{document_id}", KnowledgeNodeType.DOCUMENT, str(record["title"]),
            [str(record["title"])], {"document_id": document_id, "version": str(record["version"])},
        )
        publisher = str(record["publisher"])
        publisher_node = self._node_payload(
            f"publisher:{self._digest(self.normalize(publisher))}", KnowledgeNodeType.PUBLISHER,
            publisher, [publisher], {},
        )
        topics = [
            self._node_payload(
                f"topic:{self._safe_id(str(topic))}", KnowledgeNodeType.TOPIC,
                str(topic).replace("-", " "), [str(topic), str(topic).replace("-", " ")], {},
            )
            for topic in record.get("topics", [])  # type: ignore[union-attr]
        ]
        chunks = list(record.get("chunks", []))  # type: ignore[arg-type]
        sections: list[dict[str, object]] = []
        concept_chunks: dict[str, set[str]] = {}
        concept_nodes: dict[str, dict[str, object]] = {}
        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            heading = str(chunk.get("heading", "")).strip()
            text = str(chunk["text"])
            if heading:
                sections.append({
                    **self._node_payload(
                        f"section:{self._digest(f'{document_id}:{self.normalize(heading)}')}",
                        KnowledgeNodeType.SECTION, heading, [heading],
                        {"document_id": document_id, "chunk_id": chunk_id},
                    ),
                    "chunk_id": chunk_id,
                })
            for canonical, aliases in self._concepts_in_text(self.normalize(f"{heading} {text}")):
                node_id = f"concept:{self._safe_id(canonical)}"
                concept_nodes[node_id] = self._node_payload(
                    node_id, KnowledgeNodeType.CONCEPT, aliases[0] if aliases else canonical,
                    [canonical, *aliases], {},
                )
                concept_chunks.setdefault(node_id, set()).add(chunk_id)
        concepts = [
            {**node, "chunk_ids": sorted(concept_chunks[node_id])}
            for node_id, node in concept_nodes.items()
        ]
        session.run(
            """MERGE (d:KnowledgeEntity {id: $document.id})
               SET d += $document, d.namespace=$namespace
               WITH d MATCH (d)-[old:RELATES]->() DELETE old""",
            document=document_node, namespace="stock-kb",
        ).consume()
        session.run(
            """MATCH (d:KnowledgeEntity {id: $document_id})
               MERGE (p:KnowledgeEntity {id: $publisher.id}) SET p += $publisher, p.namespace=$namespace
               MERGE (d)-[r:RELATES {kind:'PUBLISHED_BY', evidence_document_id:$evidence_document_id}]->(p)
               SET r.weight=0.65""",
            document_id=document_node["id"], publisher=publisher_node,
            evidence_document_id=document_id, namespace="stock-kb",
        ).consume()
        session.run(
            """MATCH (d:KnowledgeEntity {id: $document_id})
               UNWIND $topics AS topic
               MERGE (t:KnowledgeEntity {id: topic.id}) SET t += topic, t.namespace=$namespace
               MERGE (d)-[r:RELATES {kind:'HAS_TOPIC', evidence_document_id:$evidence_document_id}]->(t)
               SET r.weight=1.0""",
            document_id=document_node["id"], topics=topics,
            evidence_document_id=document_id, namespace="stock-kb",
        ).consume()
        session.run(
            """MATCH (d:KnowledgeEntity {id: $document_id})
               UNWIND $sections AS section
               MERGE (s:KnowledgeEntity {id: section.id}) SET s += section, s.namespace=$namespace
               MERGE (d)-[r:RELATES {kind:'HAS_SECTION', evidence_document_id:$evidence_document_id,
                                     evidence_chunk_id:section.chunk_id}]->(s)
               SET r.weight=0.9""",
            document_id=document_node["id"], sections=sections,
            evidence_document_id=document_id, namespace="stock-kb",
        ).consume()
        session.run(
            """MATCH (d:KnowledgeEntity {id: $document_id})
               UNWIND $concepts AS concept
               MERGE (c:KnowledgeEntity {id: concept.id}) SET c += concept, c.namespace=$namespace
               MERGE (d)-[r:RELATES {kind:'MENTIONS', evidence_document_id:$evidence_document_id}]->(c)
               SET r.weight=1.0, r.evidence_chunk_id=head(concept.chunk_ids)
               WITH c, concept
               UNWIND $topic_ids AS topic_id
               MATCH (t:KnowledgeEntity {id:topic_id})
               MERGE (c)-[about:RELATES {kind:'ABOUT_TOPIC', evidence_document_id:$evidence_document_id}]->(t)
               SET about.weight=0.65, about.evidence_chunk_id=head(concept.chunk_ids)""",
            document_id=document_node["id"], concepts=concepts,
            topic_ids=[topic["id"] for topic in topics],
            evidence_document_id=document_id, namespace="stock-kb",
        ).consume()

    def _node_payload(
        self,
        node_id: str,
        node_type: KnowledgeNodeType,
        label: str,
        aliases: list[str],
        metadata: dict[str, str | int | float | bool | None],
    ) -> dict[str, object]:
        normalized_aliases = sorted({
            normalized for alias in [label, *aliases]
            if (normalized := self.normalize(alias))
        })
        return {
            "id": node_id, "type": node_type.value, "label": label,
            "normalized_label": self.normalize(label), "aliases": normalized_aliases,
            **metadata,
        }

    def _concepts_in_text(self, normalized_text: str) -> list[tuple[str, tuple[str, ...]]]:
        return [
            (canonical, aliases)
            for canonical, aliases in self._CONCEPT_ALIASES.items()
            if any(self._phrase_present(normalized_text, self.normalize(alias)) for alias in aliases)
        ]

    @staticmethod
    def normalize(value: str) -> str:
        decomposed = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
        unaccented = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
        normalized = re.sub(r"[^a-z0-9+]+", " ", unaccented).strip()
        normalized = re.sub(r"\bp\s+e\b", "pe", normalized)
        normalized = re.sub(r"\bp\s+b\b", "pb", normalized)
        return normalized

    @staticmethod
    def _phrase_present(text: str, phrase: str) -> bool:
        return bool(phrase and re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text))

    @staticmethod
    def _properties(value: Any) -> dict[str, object]:
        return dict(value)

    @staticmethod
    def _node_model(properties: dict[str, object]) -> KnowledgeGraphNode:
        metadata = {
            key: value for key, value in properties.items()
            if key not in {"id", "type", "label", "aliases", "normalized_label", "namespace"}
            and isinstance(value, (str, int, float, bool, type(None)))
        }
        return KnowledgeGraphNode(
            node_id=str(properties["id"]), node_type=str(properties["type"]),
            label=str(properties["label"]), aliases=list(properties.get("aliases", [])), metadata=metadata,
        )

    @staticmethod
    def _optional_text(value: object) -> str | None:
        return str(value) if value not in (None, "") else None

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _safe_id(value: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
        return cleaned[:80] or Neo4jKnowledgeGraph._digest(value)

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        return default if raw is None else raw.strip().lower() not in {"0", "false", "no", "off"}
