from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from stock_multi_agent.models.research import Citation, DocumentChunk, EvidenceBundle, ResearchDocument, ResearchDocumentType


class ResearchIngestionError(ValueError):
    pass


class ResearchStore:
    """Local, citation-first document store for verified user-supplied documents.

    It never downloads a URL itself. Source selection, licensing and retrieval are
    explicit responsibilities, so a citation can never be fabricated from a URL.
    """

    def __init__(self, database_path: Path | None = None) -> None:
        self.database_path = database_path or self._runtime_root() / "research" / "research.sqlite3"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def _runtime_root() -> Path:
        default_root = Path(__file__).resolve().parents[3] / ".runtime"
        runtime_root = Path(os.getenv("STOCK_AGENT_RUNTIME_DIR", str(default_root))).resolve()
        if os.name == "nt" and runtime_root.drive.upper() != "D:":
            raise ResearchIngestionError(f"Workspace policy requires research storage on drive D:, got {runtime_root}")
        return runtime_root

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    publisher TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    tickers_json TEXT NOT NULL,
                    language TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES research_documents(document_id),
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS research_chunks_fts USING fts5(chunk_id UNINDEXED, text);
                """
            )

    @staticmethod
    def read_text_file(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix not in {".txt", ".md", ".html", ".htm"}:
            raise ResearchIngestionError("Only .txt, .md, .html and .htm are supported without an optional document parser.")
        text = path.read_text(encoding="utf-8")
        if suffix in {".html", ".htm"}:
            text = ResearchStore.normalize_html(text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def normalize_html(html: str) -> str:
        text = re.sub(r"<(?:script|style)[^>]*>.*?</(?:script|style)>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _chunks(document_id: str, text: str, chunk_chars: int = 1200, overlap_chars: int = 180) -> list[DocumentChunk]:
        if not text.strip():
            raise ResearchIngestionError("Document has no extractable text.")
        chunks: list[DocumentChunk] = []
        start = 0
        ordinal = 0
        while start < len(text):
            end = min(len(text), start + chunk_chars)
            if end < len(text):
                boundary = text.rfind(" ", start + chunk_chars // 2, end)
                if boundary > start:
                    end = boundary
            segment = text[start:end].strip()
            if segment:
                chunk_id = hashlib.sha256(f"{document_id}:{ordinal}:{start}:{end}".encode()).hexdigest()
                chunks.append(DocumentChunk(chunk_id=chunk_id, document_id=document_id, ordinal=ordinal, text=segment, char_start=start, char_end=end))
                ordinal += 1
            if end >= len(text):
                break
            start = max(end - overlap_chars, start + 1)
        return chunks

    def ingest_text(
        self,
        *,
        text: str,
        title: str,
        document_type: ResearchDocumentType,
        source_url: str,
        publisher: str,
        published_at: datetime,
        tickers: list[str] | None = None,
        language: str = "vi",
    ) -> ResearchDocument:
        if published_at.tzinfo is None or published_at.utcoffset() is None:
            raise ResearchIngestionError("published_at must include a timezone")
        canonical_text = re.sub(r"\s+", " ", text).strip()
        content_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
        document_id = hashlib.sha256(f"{source_url}:{content_hash}".encode()).hexdigest()
        document = ResearchDocument(
            document_id=document_id,
            title=title,
            document_type=document_type,
            source_url=source_url,
            publisher=publisher,
            published_at=published_at,
            retrieved_at=datetime.now(timezone.utc),
            content_sha256=content_hash,
            tickers=[ticker.upper() for ticker in (tickers or [])],
            language=language,
        )
        chunks = self._chunks(document_id, canonical_text)
        with self._connect() as connection:
            existing = connection.execute("SELECT document_id FROM research_documents WHERE document_id = ?", (document_id,)).fetchone()
            if existing:
                return document
            connection.execute(
                """INSERT INTO research_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (document.document_id, document.title, document.document_type.value, str(document.source_url), document.publisher,
                 document.published_at.isoformat(), document.retrieved_at.isoformat(), document.content_sha256,
                 json.dumps(document.tickers), document.language),
            )
            connection.executemany(
                "INSERT INTO research_chunks VALUES (?, ?, ?, ?, ?, ?)",
                [(chunk.chunk_id, chunk.document_id, chunk.ordinal, chunk.text, chunk.char_start, chunk.char_end) for chunk in chunks],
            )
            connection.executemany(
                "INSERT INTO research_chunks_fts (chunk_id, text) VALUES (?, ?)",
                [(chunk.chunk_id, chunk.text) for chunk in chunks],
            )
        return document

    def search(self, query: str, ticker: str | None = None, limit: int = 5) -> EvidenceBundle:
        cleaned = re.sub(r"\s+", " ", query).strip()
        if not cleaned:
            raise ResearchIngestionError("Research query cannot be empty.")
        terms = " OR ".join(self._search_terms(cleaned))
        if not terms:
            return EvidenceBundle(query=query, citations=[], retrieval_note="No searchable terms in the request.")
        ticker_filter = ticker.upper() if ticker else None
        sql = """
            SELECT c.chunk_id, c.document_id, c.text, d.title, d.source_url, d.publisher,
                   d.published_at, d.retrieved_at, d.content_sha256, d.tickers_json
            FROM research_chunks_fts f
            JOIN research_chunks c ON c.chunk_id = f.chunk_id
            JOIN research_documents d ON d.document_id = c.document_id
            WHERE research_chunks_fts MATCH ?
            ORDER BY bm25(research_chunks_fts)
            LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(sql, (terms, max(1, min(limit * 4, 30)))).fetchall()
        citations: list[Citation] = []
        seen_documents: set[str] = set()
        for row in rows:
            if ticker_filter and ticker_filter not in json.loads(row["tickers_json"]):
                continue
            if row["document_id"] in seen_documents:
                continue
            citations.append(Citation(
                document_id=row["document_id"], chunk_id=row["chunk_id"], title=row["title"],
                source_url=row["source_url"], publisher=row["publisher"],
                published_at=datetime.fromisoformat(row["published_at"]),
                retrieved_at=datetime.fromisoformat(row["retrieved_at"]), content_sha256=row["content_sha256"],
                quote=row["text"][:1400],
            ))
            seen_documents.add(row["document_id"])
            if len(citations) >= limit:
                break
        note = "Verified local documents matched the query." if citations else "No verified local document matched this query."
        return EvidenceBundle(query=query, citations=citations, retrieval_note=note)

    @staticmethod
    def _search_terms(query: str) -> list[str]:
        """Add a few deterministic research aliases, without needing an LLM."""
        terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
        normalized = "".join(
            char for char in unicodedata.normalize("NFD", query.casefold())
            if unicodedata.category(char) != "Mn"
        )
        if re.search(r"\bbao\s+cao\b", normalized):
            terms.extend(["annual", "report"])
        if re.search(r"\bbao\s+cao\s+tai\s+chinh\b", normalized):
            terms.extend(["financial", "statement"])
        if re.search(r"\bcong\s+bo\b", normalized):
            terms.append("disclosure")
        if re.search(r"\bnghi\s+quyet\b", normalized):
            terms.append("resolution")
        return list(dict.fromkeys(terms))
