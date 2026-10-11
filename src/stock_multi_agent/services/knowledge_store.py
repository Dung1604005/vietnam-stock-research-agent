from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, Field, HttpUrl, field_validator

from stock_multi_agent.models.knowledge import (
    KnowledgeChunk,
    KnowledgeCitation,
    KnowledgeDocument,
    KnowledgeSearchResult,
    KnowledgeSourceType,
    VerificationStatus,
)


class KnowledgeIngestionError(ValueError):
    pass


class SemanticKnowledgeRetriever(Protocol):
    """Optional local semantic retriever. Lexical retrieval never depends on it."""

    def search(self, query: str, limit: int) -> list[tuple[str, float]]: ...


class KnowledgeManifestEntry(BaseModel):
    path: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=500)
    publisher: str = Field(min_length=1, max_length=200)
    source_url: HttpUrl
    source_type: KnowledgeSourceType
    language: str = "vi"
    published_at: datetime | None = None
    effective_at: datetime | None = None
    usage_note: str = Field(min_length=1, max_length=1000)
    verification_status: VerificationStatus = VerificationStatus.VERIFIED
    topics: list[str] = Field(min_length=1)
    version: str = Field(min_length=1, max_length=80)
    time_sensitive: bool = False

    @field_validator("published_at", "effective_at")
    @classmethod
    def optional_times_are_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Manifest timestamps must include a timezone")
        return value


class _HTMLTextExtractor(HTMLParser):
    _BLOCK_TAGS = {
        "article", "aside", "blockquote", "br", "div", "footer", "h1", "h2", "h3",
        "h4", "h5", "h6", "header", "li", "main", "p", "section", "table", "td", "th", "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[no-untyped-def]
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


class KnowledgeStore:
    """Versioned, provenance-first local KB backed by SQLite FTS5."""

    SCHEMA_VERSION = 1
    _TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "source"}
    _STOP_WORDS = {
        "a", "an", "and", "cach", "cho", "co", "cua", "duoc", "gi", "how", "is", "la",
        "mot", "nao", "nhu", "of", "tai", "the", "thi", "tinh", "to", "va", "what",
        "bao", "biet", "dieu", "hien", "khac", "khong", "lam", "nghia", "nhieu", "noi",
        "cap", "do", "quy", "tac",
        "cong", "thuc",
    }
    _TERM_ALIASES = {
        "pe": ("p/e", "pe", "price earnings", "price-to-earnings"),
        "pb": ("p/b", "pb", "price to book", "price-to-book"),
        "roe": ("roe", "return on equity"),
        "roa": ("roa", "return on assets"),
        "eps": ("eps", "earnings per share"),
        "rsi": ("rsi", "relative strength index"),
        "macd": ("macd", "moving average convergence divergence"),
        "atr": ("atr", "average true range"),
        "ato": ("ato", "at the open"),
        "atc": ("atc", "at the close"),
        "dcf": ("dcf", "discounted cash flow"),
        "capm": ("capm", "capital asset pricing model"),
        "quyen": ("quyen mua", "rights offering"),
        "tuc": ("co tuc", "dividend"),
        "phat": ("phat hanh them", "secondary offering"),
        "ev": ("ev ebitda", "enterprise value ebitda"),
        "sharpe": ("sharpe ratio", "reward to variability"),
    }
    _TOPIC_HINTS = {
        "market-basics": ("co phieu", "von hoa", "thanh khoan", "vn index", "khop lenh", "thoa thuan"),
        "technical-analysis": ("rsi", "macd", "sma", "ema", "bollinger", "atr", "ho tro", "khang cu", "xu huong"),
        "financial-ratios": ("pe", "pb", "roe", "roa", "eps", "bien loi nhuan", "debt to equity", "current ratio"),
        "valuation": ("dinh gia", "dcf", "ev ebitda", "margin of safety"),
        "risk": ("rui ro", "volatility", "drawdown", "da dang hoa", "position sizing", "thanh khoan"),
        "corporate-actions": ("co tuc", "chia tach", "phat hanh", "quyen mua", "khong huong quyen"),
        "market-rules": ("ato", "atc", "lo", "bien do", "gia tran", "gia san", "thanh toan", "quy dinh"),
        "financial-statements": ("bang can doi", "ket qua kinh doanh", "luu chuyen tien", "doanh thu", "loi nhuan"),
        "advanced": ("beta", "alpha", "sharpe", "capm", "backtest", "look ahead", "survivorship", "data leakage"),
    }

    def __init__(
        self,
        database_path: Path | None = None,
        semantic_retriever: SemanticKnowledgeRetriever | None = None,
    ) -> None:
        configured = os.getenv("STOCK_AGENT_KNOWLEDGE_DATABASE", "").strip()
        self.database_path = database_path or (Path(configured).resolve() if configured else self._runtime_root() / "knowledge" / "knowledge.sqlite3")
        if os.name == "nt" and self.database_path.resolve().drive.upper() != "D:":
            raise KnowledgeIngestionError(
                f"Workspace policy requires knowledge storage on drive D:, got {self.database_path.resolve()}"
            )
        self.semantic_retriever = semantic_retriever
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def _runtime_root() -> Path:
        default_root = Path(__file__).resolve().parents[3] / ".runtime"
        runtime_root = Path(os.getenv("STOCK_AGENT_RUNTIME_DIR", str(default_root))).resolve()
        if os.name == "nt" and runtime_root.drive.upper() != "D:":
            raise KnowledgeIngestionError(f"Workspace policy requires knowledge storage on drive D:, got {runtime_root}")
        return runtime_root

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS kb_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    publisher TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    canonical_url TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    language TEXT NOT NULL,
                    published_at TEXT,
                    effective_at TEXT,
                    retrieved_at TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    usage_note TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    topics_json TEXT NOT NULL,
                    version TEXT NOT NULL,
                    time_sensitive INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(canonical_url, version)
                );
                CREATE INDEX IF NOT EXISTS idx_kb_documents_hash ON knowledge_documents(content_sha256);
                CREATE INDEX IF NOT EXISTS idx_kb_documents_url ON knowledge_documents(canonical_url, active);
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES knowledge_documents(document_id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    heading TEXT NOT NULL,
                    text TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    UNIQUE(document_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_kb_chunks_document ON knowledge_chunks(document_id, ordinal);
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    heading,
                    text,
                    normalized_text,
                    tokenize='unicode61 remove_diacritics 2'
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO kb_schema_migrations(version, applied_at) VALUES (?, ?)",
                (self.SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
            )
            connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")

    @staticmethod
    def canonicalize_url(value: str) -> str:
        parsed = urlsplit(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise KnowledgeIngestionError("source_url must be an absolute HTTP(S) URL")
        query = urlencode([
            (key, item) for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in KnowledgeStore._TRACKING_PARAMS
        ])
        path = re.sub(r"/{2,}", "/", parsed.path or "/")
        if path != "/":
            path = path.rstrip("/")
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, query, ""))

    @staticmethod
    def normalize_unicode(value: str) -> str:
        value = unicodedata.normalize("NFC", value).replace("\u00a0", " ")
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n[ \t]+", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()

    @staticmethod
    def normalize_for_search(value: str) -> str:
        decomposed = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
        unaccented = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
        normalized = re.sub(r"[^a-z0-9]+", " ", unaccented).strip()
        normalized = re.sub(r"\bp\s+e\b", "pe", normalized)
        normalized = re.sub(r"\bp\s+b\b", "pb", normalized)
        return normalized

    @classmethod
    def read_document(cls, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            return cls._remove_repeated_boilerplate(cls.normalize_unicode(path.read_text(encoding="utf-8")))
        if suffix in {".html", ".htm"}:
            parser = _HTMLTextExtractor()
            parser.feed(path.read_text(encoding="utf-8"))
            return cls._remove_repeated_boilerplate(cls.normalize_unicode(parser.text()))
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise KnowledgeIngestionError(
                    "PDF ingestion requires the optional 'knowledge' dependency: pip install -e .[knowledge]"
                ) from exc
            reader = PdfReader(str(path))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            return cls._remove_repeated_boilerplate(cls.normalize_unicode(text))
        raise KnowledgeIngestionError("Supported knowledge formats are .md, .txt, .html, .htm and text-based .pdf")

    @staticmethod
    def _remove_repeated_boilerplate(text: str) -> str:
        """Drop short lines repeated across pages, a conservative header/footer heuristic."""
        lines = text.splitlines()
        counts: dict[str, int] = {}
        for line in lines:
            key = re.sub(r"\s+", " ", line).strip()
            if key and len(key) <= 180 and not key.startswith("#"):
                counts[key] = counts.get(key, 0) + 1
        kept = [
            line for line in lines
            if not (
                (key := re.sub(r"\s+", " ", line).strip())
                and len(key) <= 180
                and not key.startswith("#")
                and counts.get(key, 0) >= 3
            )
        ]
        return KnowledgeStore.normalize_unicode("\n".join(kept))

    @classmethod
    def _chunks(cls, document_id: str, text: str, max_chars: int = 2200) -> list[KnowledgeChunk]:
        if not text.strip():
            raise KnowledgeIngestionError("Document has no extractable text.")
        sections: list[tuple[str, str]] = []
        heading = ""
        body: list[str] = []
        for line in text.splitlines():
            match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
            if match:
                if body:
                    sections.append((heading, "\n".join(body).strip()))
                heading, body = match.group(1).strip(), []
            else:
                body.append(line)
        if body:
            sections.append((heading, "\n".join(body).strip()))
        if not sections:
            sections = [("", text)]

        chunks: list[KnowledgeChunk] = []
        cursor = 0
        ordinal = 0
        for section_heading, section_text in sections:
            paragraphs = [item.strip() for item in re.split(r"\n\s*\n", section_text) if item.strip()]
            batch: list[str] = []
            size = 0
            for paragraph in paragraphs or [section_text]:
                if batch and size + len(paragraph) + 2 > max_chars:
                    ordinal, cursor = cls._append_chunk(
                        chunks, document_id, ordinal, cursor, text, section_heading, "\n\n".join(batch)
                    )
                    batch, size = [], 0
                batch.append(paragraph)
                size += len(paragraph) + 2
            if batch:
                ordinal, cursor = cls._append_chunk(
                    chunks, document_id, ordinal, cursor, text, section_heading, "\n\n".join(batch)
                )
        return chunks

    @staticmethod
    def _append_chunk(
        chunks: list[KnowledgeChunk], document_id: str, ordinal: int, cursor: int,
        full_text: str, heading: str, body: str,
    ) -> tuple[int, int]:
        segment = f"{heading}\n\n{body}".strip() if heading else body.strip()
        start = full_text.find(body[: min(80, len(body))], cursor)
        start = cursor if start < 0 else start
        end = min(len(full_text), start + len(body))
        chunk_id = hashlib.sha256(f"{document_id}:{ordinal}:{heading}:{body}".encode("utf-8")).hexdigest()
        chunks.append(KnowledgeChunk(
            chunk_id=chunk_id, document_id=document_id, ordinal=ordinal,
            heading=heading, text=segment, char_start=start, char_end=max(start + 1, end),
        ))
        return ordinal + 1, end

    def ingest_text(
        self,
        *,
        text: str,
        title: str,
        publisher: str,
        source_url: str,
        source_type: KnowledgeSourceType,
        usage_note: str,
        verification_status: VerificationStatus,
        topics: list[str],
        version: str,
        language: str = "vi",
        published_at: datetime | None = None,
        effective_at: datetime | None = None,
        time_sensitive: bool = False,
        retrieved_at: datetime | None = None,
    ) -> KnowledgeDocument:
        if verification_status is VerificationStatus.VERIFIED and (not publisher.strip() or not source_url.strip()):
            raise KnowledgeIngestionError("VERIFIED documents require publisher and source_url provenance.")
        canonical_text = self.normalize_unicode(text)
        if not canonical_text:
            raise KnowledgeIngestionError("Document has no content.")
        canonical_url = self.canonicalize_url(source_url)
        content_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
        document_id = hashlib.sha256(f"{canonical_url}:{version}:{content_hash}".encode("utf-8")).hexdigest()
        document = KnowledgeDocument(
            document_id=document_id, title=title.strip(), publisher=publisher.strip(),
            source_url=source_url, canonical_url=canonical_url, source_type=source_type,
            language=language, published_at=published_at, effective_at=effective_at,
            retrieved_at=retrieved_at or datetime.now(timezone.utc), content_sha256=content_hash,
            usage_note=usage_note.strip(), verification_status=verification_status,
            topics=list(dict.fromkeys(topic.strip().lower() for topic in topics if topic.strip())),
            version=version.strip(), time_sensitive=time_sensitive,
        )
        chunks = self._chunks(document.document_id, canonical_text)
        with self._connect() as connection:
            duplicate = connection.execute(
                "SELECT document_id FROM knowledge_documents WHERE content_sha256 = ? AND active = 1",
                (content_hash,),
            ).fetchone()
            if duplicate:
                return self.get_document(duplicate["document_id"], connection=connection)
            existing = connection.execute(
                "SELECT document_id, content_sha256 FROM knowledge_documents WHERE canonical_url = ? AND version = ?",
                (canonical_url, version),
            ).fetchone()
            if existing and existing["content_sha256"] != content_hash:
                raise KnowledgeIngestionError("The same canonical URL/version already has different content; increment version.")
            connection.execute("UPDATE knowledge_documents SET active = 0 WHERE canonical_url = ?", (canonical_url,))
            connection.execute(
                """INSERT INTO knowledge_documents(
                    document_id,title,publisher,source_url,canonical_url,source_type,language,published_at,
                    effective_at,retrieved_at,content_sha256,usage_note,verification_status,topics_json,
                    version,time_sensitive,active
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
                (
                    document.document_id, document.title, document.publisher, str(document.source_url),
                    str(document.canonical_url), document.source_type.value, document.language,
                    document.published_at.isoformat() if document.published_at else None,
                    document.effective_at.isoformat() if document.effective_at else None,
                    document.retrieved_at.isoformat(), document.content_sha256, document.usage_note,
                    document.verification_status.value, json.dumps(document.topics, ensure_ascii=False),
                    document.version, int(document.time_sensitive),
                ),
            )
            connection.executemany(
                "INSERT INTO knowledge_chunks VALUES (?,?,?,?,?,?,?,?)",
                [(
                    chunk.chunk_id, chunk.document_id, chunk.ordinal, chunk.heading, chunk.text,
                    self.normalize_for_search(f"{chunk.heading} {chunk.text}"), chunk.char_start, chunk.char_end,
                ) for chunk in chunks],
            )
            connection.executemany(
                "INSERT INTO knowledge_chunks_fts(chunk_id,heading,text,normalized_text) VALUES (?,?,?,?)",
                [(
                    chunk.chunk_id, chunk.heading, chunk.text,
                    self.normalize_for_search(f"{chunk.heading} {chunk.text}"),
                ) for chunk in chunks],
            )
        return document

    def ingest_manifest(self, manifest_path: Path) -> list[KnowledgeDocument]:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = raw["documents"] if isinstance(raw, dict) else raw
        if not isinstance(records, list):
            raise KnowledgeIngestionError("Manifest must be a list or contain a 'documents' list.")
        documents: list[KnowledgeDocument] = []
        for raw_entry in records:
            entry = KnowledgeManifestEntry.model_validate(raw_entry)
            source_path = (manifest_path.parent / entry.path).resolve()
            if not source_path.is_relative_to(manifest_path.parent.resolve()):
                raise KnowledgeIngestionError("Manifest document path must stay inside the manifest directory.")
            if not source_path.is_file():
                raise KnowledgeIngestionError(f"Knowledge source file does not exist: {source_path}")
            payload = entry.model_dump(exclude={"path"})
            payload["source_url"] = str(entry.source_url)
            documents.append(self.ingest_text(text=self.read_document(source_path), **payload))
        return documents

    def get_document(self, document_id: str, connection: sqlite3.Connection | None = None) -> KnowledgeDocument:
        owns_connection = connection is None
        active_connection = connection or self._connect()
        try:
            row = active_connection.execute(
                "SELECT * FROM knowledge_documents WHERE document_id = ?", (document_id,),
            ).fetchone()
            if not row:
                raise KeyError(document_id)
            return self._document_from_row(row)
        finally:
            if owns_connection:
                active_connection.close()

    def count_documents(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM knowledge_documents WHERE active = 1").fetchone()[0])

    def search(self, query: str, limit: int = 5, topics: list[str] | None = None) -> KnowledgeSearchResult:
        cleaned = re.sub(r"\s+", " ", query).strip()
        if not cleaned:
            raise KnowledgeIngestionError("Knowledge query cannot be empty.")
        normalized = self.normalize_for_search(cleaned)
        inferred_topics = set(topics or self._infer_topics(normalized))
        tokens = [token for token in normalized.split() if token not in self._STOP_WORDS and len(token) > 1]
        terms = list(dict.fromkeys(tokens))
        if not terms:
            return KnowledgeSearchResult(query=query, retrieval_note="No searchable knowledge terms were found.")

        candidates: dict[str, dict[str, object]] = {}
        mode_ranks: dict[str, dict[str, int]] = {"exact": {}, "bm25": {}, "semantic": {}}
        with self._connect() as connection:
            fts_query = " OR ".join(f'"{term}"' for term in terms[:20] if re.fullmatch(r"[a-z0-9]+", term))
            if fts_query:
                rows = connection.execute(
                    """SELECT c.*, d.*, bm25(knowledge_chunks_fts, 0.0, 2.0, 1.0, 0.6) AS lexical_score
                       FROM knowledge_chunks_fts f
                       JOIN knowledge_chunks c ON c.chunk_id=f.chunk_id
                       JOIN knowledge_documents d ON d.document_id=c.document_id
                       WHERE knowledge_chunks_fts MATCH ? AND d.active=1
                       ORDER BY lexical_score LIMIT ?""",
                    (fts_query, max(30, limit * 8)),
                ).fetchall()
                for rank, row in enumerate(rows, 1):
                    candidates[row["chunk_id"]] = dict(row)
                    mode_ranks["bm25"][row["chunk_id"]] = rank

            exact_phrases = self._exact_phrases(normalized)
            if exact_phrases:
                conditions = " OR ".join("c.normalized_text LIKE ?" for _ in exact_phrases)
                rows = connection.execute(
                    f"""SELECT c.*, d.* FROM knowledge_chunks c
                        JOIN knowledge_documents d ON d.document_id=c.document_id
                        WHERE d.active=1 AND ({conditions}) LIMIT ?""",
                    (*[f"%{term}%" for term in exact_phrases], max(30, limit * 8)),
                ).fetchall()
                ranked = sorted(
                    (
                        row for row in rows
                        if any(re.search(rf"\b{re.escape(term)}\b", row["normalized_text"]) for term in exact_phrases)
                    ),
                    key=lambda row: -sum(
                        bool(re.search(rf"\b{re.escape(term)}\b", row["normalized_text"]))
                        for term in exact_phrases
                    ),
                )
                for rank, row in enumerate(ranked, 1):
                    candidates.setdefault(row["chunk_id"], dict(row))
                    mode_ranks["exact"][row["chunk_id"]] = rank

            if self.semantic_retriever is not None:
                for rank, (chunk_id, semantic_score) in enumerate(self.semantic_retriever.search(cleaned, max(20, limit * 4)), 1):
                    row = connection.execute(
                        """SELECT c.*, d.* FROM knowledge_chunks c JOIN knowledge_documents d
                           ON d.document_id=c.document_id WHERE c.chunk_id=? AND d.active=1""",
                        (chunk_id,),
                    ).fetchone()
                    if row:
                        candidates.setdefault(chunk_id, dict(row))
                        candidates[chunk_id]["semantic_score"] = max(0.0, min(1.0, float(semantic_score)))
                        mode_ranks["semantic"][chunk_id] = rank

        ranked_citations: list[KnowledgeCitation] = []
        for chunk_id, row in candidates.items():
            row_topics = set(json.loads(str(row["topics_json"])))
            normalized_text = str(row["normalized_text"])
            matched_terms = sum(
                bool(re.search(rf"\b{re.escape(term)}\b", normalized_text)) for term in terms
            )
            coverage = matched_terms / len(terms)
            has_strong_signal = chunk_id in mode_ranks["exact"] or chunk_id in mode_ranks["semantic"]
            if coverage < 0.34 and not has_strong_signal:
                continue
            score = sum(1.0 / (60 + ranks[chunk_id]) for ranks in mode_ranks.values() if chunk_id in ranks) * 100
            score += 0.50 * coverage
            score += 0.80 * len(inferred_topics.intersection(row_topics))
            score += self._source_weight(str(row["source_type"]))
            if row["verification_status"] == VerificationStatus.VERIFIED.value:
                score += 0.30
            modes = [name for name, ranks in mode_ranks.items() if chunk_id in ranks]
            warning = self._freshness_warning(row)
            ranked_citations.append(self._citation_from_row(row, score, modes, warning))

        ranked_citations.sort(key=lambda item: (-item.score, item.document_id, item.chunk_id))
        selected = ranked_citations[: max(1, min(limit, 10))]
        semantic_used = bool(mode_ranks["semantic"])
        note = "Verified local KB evidence matched the query." if selected else "No verified local KB evidence matched this query."
        return KnowledgeSearchResult(
            query=query,
            citations=selected,
            retrieval_note=note,
            retrieval_mode="exact+bm25+semantic-rrf" if semantic_used else "exact+bm25-rrf",
            requires_freshness_check=any(item.freshness_warning for item in selected),
        )

    @classmethod
    def _expanded_terms(cls, tokens: list[str]) -> list[str]:
        expanded: list[str] = []
        for token in tokens:
            if token in cls._TERM_ALIASES:
                for alias in cls._TERM_ALIASES[token]:
                    expanded.extend(cls.normalize_for_search(alias).split())
        return expanded

    @classmethod
    def _exact_phrases(cls, normalized: str) -> list[str]:
        result: list[str] = []
        compact = normalized.replace(" ", "")
        for canonical, aliases in cls._TERM_ALIASES.items():
            if canonical in normalized.split() or canonical in compact:
                result.extend(cls.normalize_for_search(alias) for alias in aliases)
        return list(dict.fromkeys(item for item in result if item))

    @classmethod
    def _infer_topics(cls, normalized: str) -> list[str]:
        return [
            topic for topic, hints in cls._TOPIC_HINTS.items()
            if any(re.search(rf"\b{re.escape(hint)}\b", normalized) for hint in hints)
        ]

    @staticmethod
    def _source_weight(source_type: str) -> float:
        return {
            KnowledgeSourceType.OFFICIAL_REGULATION.value: 0.50,
            KnowledgeSourceType.EXCHANGE_GUIDANCE.value: 0.45,
            KnowledgeSourceType.REGULATOR_EDUCATION.value: 0.40,
            KnowledgeSourceType.ACCOUNTING_STANDARD.value: 0.40,
            KnowledgeSourceType.OPEN_EDUCATION.value: 0.30,
            KnowledgeSourceType.PROFESSIONAL_EDUCATION.value: 0.30,
        }.get(source_type, 0.10)

    @staticmethod
    def _freshness_warning(row: sqlite3.Row | dict[str, object]) -> str | None:
        if not bool(row["time_sensitive"]):
            return None
        effective = row["effective_at"] or row["published_at"]
        basis = f"effective/published {str(effective)[:10]}" if effective else "no effective date supplied"
        return (
            "Quy định có thể thay đổi; hãy kiểm tra bản hiện hành tại nguồn chính thức "
            f"({basis}, retrieved {str(row['retrieved_at'])[:10]})."
        )

    @staticmethod
    def _citation_from_row(
        row: sqlite3.Row | dict[str, object], score: float, modes: list[str], warning: str | None,
    ) -> KnowledgeCitation:
        return KnowledgeCitation(
            document_id=str(row["document_id"]), chunk_id=str(row["chunk_id"]),
            title=str(row["title"]), heading=str(row["heading"]), publisher=str(row["publisher"]),
            source_url=str(row["source_url"]), source_type=str(row["source_type"]),
            published_at=datetime.fromisoformat(str(row["published_at"])) if row["published_at"] else None,
            effective_at=datetime.fromisoformat(str(row["effective_at"])) if row["effective_at"] else None,
            retrieved_at=datetime.fromisoformat(str(row["retrieved_at"])), content_sha256=str(row["content_sha256"]),
            verification_status=str(row["verification_status"]), topics=json.loads(str(row["topics_json"])),
            version=str(row["version"]), quote=str(row["text"])[:2400], score=max(0.0, score),
            retrieval_modes=modes, freshness_warning=warning,
        )

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> KnowledgeDocument:
        return KnowledgeDocument(
            document_id=row["document_id"], title=row["title"], publisher=row["publisher"],
            source_url=row["source_url"], canonical_url=row["canonical_url"], source_type=row["source_type"],
            language=row["language"],
            published_at=datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
            effective_at=datetime.fromisoformat(row["effective_at"]) if row["effective_at"] else None,
            retrieved_at=datetime.fromisoformat(row["retrieved_at"]), content_sha256=row["content_sha256"],
            usage_note=row["usage_note"], verification_status=row["verification_status"],
            topics=json.loads(row["topics_json"]), version=row["version"], time_sensitive=bool(row["time_sensitive"]),
        )

