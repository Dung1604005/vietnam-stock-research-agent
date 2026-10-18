from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from collections.abc import Iterator
from difflib import SequenceMatcher

import numpy as np

from stock_multi_agent.models.research import Citation, DocumentChunk, EvidenceBundle, ResearchDocument, ResearchDocumentType, ReportingPeriod
from stock_multi_agent.services.embeddings import EmbeddingProvider, LocalHashEmbeddingProvider, configured_embedding_provider
from stock_multi_agent.services.source_registry import canonical_url, SourceRegistry
from stock_multi_agent.services.request_telemetry import observed
from stock_multi_agent.services.reporting_period import infer_reporting_period
from stock_multi_agent.services.research_freshness import ResearchScope, period_sql, freshness_note


class ResearchIngestionError(ValueError):
    pass


class ResearchStore:
    """Local, citation-first document store for verified user-supplied documents.

    It never downloads a URL itself. Source selection, licensing and retrieval are
    explicit responsibilities, so a citation can never be fabricated from a URL.
    """

    _STOP_WORDS = frozenset({"tim", "cho", "toi", "cua", "va", "the", "nao", "cac", "nhung", "mot", "nhieu",
        "co", "khong", "la", "gi", "day", "nay", "trong", "voi", "ve", "tu", "den", "duoc", "hay", "hoac",
        "toi", "muon", "can", "giup", "cach", "lam", "bi", "se", "da", "dang", "tai", "sao", "khi", "neu",
        "thoi", "ngay", "mai", "hom", "qua", "cuoi", "tuan", "chi", "duong", "noi", "cong", "thuc",
        "the", "and", "of", "for", "how", "is", "a", "to", "what", "with", "from", "it", "on"})

    def __init__(
        self,
        database_path: Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.database_path = database_path or self._runtime_root() / "research" / "research.sqlite3"
        rag_enabled = os.getenv("STOCK_AGENT_RAG_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
        self.embedding_provider = embedding_provider or (configured_embedding_provider() if rag_enabled else None)
        self.last_embedding_error: str | None = None
        self.last_graph_error: str | None = None
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def _runtime_root() -> Path:
        default_root = Path(__file__).resolve().parents[3] / ".runtime"
        runtime_root = Path(os.getenv("STOCK_AGENT_RUNTIME_DIR", str(default_root))).resolve()
        if os.name == "nt" and runtime_root.drive.upper() != "D:":
            raise ResearchIngestionError(f"Workspace policy requires research storage on drive D:, got {runtime_root}")
        return runtime_root

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

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
                CREATE TABLE IF NOT EXISTS research_chunk_embeddings (
                    chunk_id TEXT NOT NULL REFERENCES research_chunks(chunk_id),
                    model_id TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    embedded_at TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, model_id)
                );
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(research_documents)")}
            for name, definition in {"quality_tier": "TEXT NOT NULL DEFAULT 'T4'", "usage_note": "TEXT NOT NULL DEFAULT 'Legacy user-supplied document'",
                    "version": "INTEGER NOT NULL DEFAULT 1", "supersedes_id": "TEXT", "period_end": "TEXT",
                    "reporting_period_json": "TEXT NOT NULL DEFAULT '{}'"}.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE research_documents ADD COLUMN {name} {definition}")
            connection.execute("CREATE INDEX IF NOT EXISTS research_source_idx ON research_documents(source_url)")
            connection.execute("CREATE INDEX IF NOT EXISTS research_period_idx ON research_documents(document_type,period_end)")
            # Backfill metadata only. Document content, identity and embeddings remain untouched.
            legacy = connection.execute("SELECT document_id,title,document_type,period_end FROM research_documents WHERE reporting_period_json='{}'").fetchall()
            for row in legacy:
                pieces = connection.execute("SELECT text FROM research_chunks WHERE document_id=? ORDER BY ordinal LIMIT 2", (row['document_id'],)).fetchall()
                detected = infer_reporting_period(row['title'], '\n'.join(piece[0] for piece in pieces), row['document_type'],
                    declared_end=date.fromisoformat(row['period_end']) if row['period_end'] else None)
                connection.execute("UPDATE research_documents SET reporting_period_json=?,period_end=? WHERE document_id=?",
                    (detected.model_dump_json(), detected.end.isoformat() if detected.end else row['period_end'], row['document_id']))

    @staticmethod
    def read_text_file(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise ResearchIngestionError("Install the knowledge extra on D: to ingest PDF") from exc
            text = "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
            if not text.strip():
                raise ResearchIngestionError("Scanned PDF has no text layer; OCR is required")
            return text
        if suffix not in {".txt", ".md", ".html", ".htm"}:
            raise ResearchIngestionError("Only .txt, .md, .html and .htm are supported without an optional document parser.")
        text = path.read_text(encoding="utf-8")
        if suffix in {".html", ".htm"}:
            text = ResearchStore.normalize_html(text)
        return text.strip()

    @staticmethod
    def normalize_html(html: str) -> str:
        text = re.sub(r"<(?:script|style)[^>]*>.*?</(?:script|style)>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"</(?:p|div|h[1-6]|tr|section)>|<br\s*/?>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        import html
        return html.unescape(text).strip()

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
                boundary = text.rfind("\n\n", start + chunk_chars // 3, end)
                if boundary < 0:
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
        quality_tier: str | None = None,
        usage_note: str | None = None,
        period_end: date | None = None,
    ) -> ResearchDocument:
        if published_at.tzinfo is None or published_at.utcoffset() is None:
            raise ResearchIngestionError("published_at must include a timezone")
        source_url = canonical_url(source_url)
        policy = SourceRegistry().resolve(source_url)
        canonical_text = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
        content_hash = hashlib.sha256(re.sub(r"\s+", " ", canonical_text).encode("utf-8")).hexdigest()
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
            quality_tier=quality_tier or (policy.quality_tier if policy else "T4"),
            usage_note=usage_note or (policy.usage_note if policy else "Explicit user-supplied document"),
            period_end=period_end,
            reporting_period=infer_reporting_period(title, canonical_text, document_type, declared_end=period_end),
        )
        document.period_end = document.reporting_period.end
        chunks = self._chunks(document_id, canonical_text)
        with self._connect() as connection:
            previous = connection.execute("SELECT * FROM research_documents WHERE source_url = ? ORDER BY version DESC LIMIT 1", (source_url,)).fetchone()
            existing = connection.execute("SELECT * FROM research_documents WHERE document_id = ? OR (content_sha256 = ? AND tickers_json = ?)",
                (document_id, content_hash, json.dumps(document.tickers))).fetchone()
            reverting = existing is not None and previous is not None and previous["content_sha256"] != content_hash
            if existing and not reverting:
                old_period = ReportingPeriod.model_validate_json(existing["reporting_period_json"])
                if existing["source_url"] == source_url and old_period.status == "UNKNOWN" and document.reporting_period.status == "DETECTED":
                    connection.execute("UPDATE research_documents SET reporting_period_json=?,period_end=? WHERE document_id=?",
                        (document.reporting_period.model_dump_json(), document.period_end.isoformat(), existing["document_id"]))
                    existing = connection.execute("SELECT * FROM research_documents WHERE document_id=?", (existing["document_id"],)).fetchone()
                return self._document(existing)
            if reverting:
                document_id = hashlib.sha256(f"{source_url}:{content_hash}:{previous['version'] + 1}".encode()).hexdigest()
                document.document_id = document_id
                chunks = self._chunks(document_id, canonical_text)
            # Cross-URL near duplicates are merged only when all numeric tokens
            # agree. A changed filing at the same URL is always a new revision.
            candidates = connection.execute("SELECT d.*, (SELECT group_concat(text, ' ') FROM research_chunks c WHERE c.document_id=d.document_id) AS body FROM research_documents d WHERE tickers_json=? AND source_url<>?",
                (json.dumps(document.tickers), source_url)).fetchall()
            words = set(re.findall(r"\w+", canonical_text.casefold()))
            numbers = re.findall(r"\d+(?:[.,]\d+)*", canonical_text)
            for candidate in candidates:
                previous_text = candidate["body"] or ""
                other_words = set(re.findall(r"\w+", previous_text.casefold()))
                similarity = len(words & other_words) / max(1, len(words | other_words))
                ordered_similarity = SequenceMatcher(None, canonical_text.casefold(), previous_text.casefold(), autojunk=False).ratio() if similarity >= .985 else 0
                if previous is None and len(words) >= 30 and ordered_similarity >= .985 and numbers == re.findall(r"\d+(?:[.,]\d+)*", previous_text):
                    data = {key: candidate[key] for key in candidate.keys() if key != "body"}
                    return self._document(data)
            if previous:
                document.version = previous["version"] + 1
                document.supersedes_id = previous["document_id"]
            connection.execute(
                """INSERT INTO research_documents (document_id,title,document_type,source_url,publisher,published_at,retrieved_at,content_sha256,tickers_json,language,
                    quality_tier,usage_note,version,supersedes_id,period_end,reporting_period_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (document.document_id, document.title, document.document_type.value, str(document.source_url), document.publisher,
                 document.published_at.isoformat(), document.retrieved_at.isoformat(), document.content_sha256,
                 json.dumps(document.tickers), document.language, document.quality_tier, document.usage_note,
                 document.version, document.supersedes_id, document.period_end.isoformat() if document.period_end else None, document.reporting_period.model_dump_json()),
            )
            connection.executemany(
                "INSERT INTO research_chunks VALUES (?, ?, ?, ?, ?, ?)",
                [(chunk.chunk_id, chunk.document_id, chunk.ordinal, chunk.text, chunk.char_start, chunk.char_end) for chunk in chunks],
            )
            connection.executemany(
                "INSERT INTO research_chunks_fts (chunk_id, text) VALUES (?, ?)",
                [(chunk.chunk_id, chunk.text) for chunk in chunks],
            )
        self._embed_chunks(chunks)
        from stock_multi_agent.services.graph_v2 import InvestmentGraph
        try:
            InvestmentGraph(self.database_path).ingest_document(document, chunks)
            self.last_graph_error = None
        except (OSError, ValueError, sqlite3.Error) as exc:
            # A projection failure must not invalidate a committed source document.
            self.last_graph_error = type(exc).__name__
        return document

    def graph_citations(self, query: str, *, as_of=None, limit=5) -> list[Citation]:
        """Opt-in relational candidates; ordinary RAG freshness gates still apply."""
        from stock_multi_agent.services.graph_v2 import InvestmentGraph
        from datetime import datetime, time, timezone
        cutoff = datetime.combine(as_of, time.max, timezone.utc) if as_of else None
        paths = InvestmentGraph(self.database_path).query(query, as_of=cutoff, limit=40)
        evidence_ids = list(dict.fromkeys(item['evidence']['evidence_id'] for item in paths))
        result = []
        with self._connect() as db:
            for evidence_id in evidence_ids:
                row = db.execute('SELECT d.*,c.chunk_id,c.text FROM research_documents d JOIN research_chunks c ON d.document_id=c.document_id WHERE c.chunk_id=?', (evidence_id,)).fetchone()
                if not row:
                    continue
                # Exclude superseded versions visible by the requested date.
                newer = db.execute('SELECT 1 FROM research_documents WHERE supersedes_id=? AND date(published_at)<=?',
                    (row['document_id'], (as_of or date.today()).isoformat())).fetchone()
                if newer:
                    continue
                document = self._document(row)
                result.append(Citation(**document.model_dump(), chunk_id=row['chunk_id'], quote=row['text'][:1400],
                    retrieval_modes=['graph_v2'], retrieval_score=0.01))
                if len(result) >= limit:
                    break
        return result

    @staticmethod
    def _document(row) -> ResearchDocument:
        data = dict(row)
        data["tickers"] = json.loads(data.pop("tickers_json"))
        data["reporting_period"] = json.loads(data.pop("reporting_period_json", "{}"))
        return ResearchDocument.model_validate(data)

    def period_overview(self, ticker: str, limit=3) -> list[Citation]:
        """Metadata for missing-data notices only; these are not current analysis evidence."""
        with self._connect() as db:
            rows = db.execute("""SELECT d.*,c.chunk_id,c.text FROM research_documents d
                JOIN research_chunks c ON c.document_id=d.document_id AND c.ordinal=0
                WHERE EXISTS (SELECT 1 FROM json_each(d.tickers_json) WHERE value=?)
                AND date(d.published_at)<=date('now')
                AND NOT EXISTS (SELECT 1 FROM research_documents newer WHERE newer.supersedes_id=d.document_id AND date(newer.published_at)<=date('now'))
                ORDER BY d.period_end DESC,d.published_at DESC LIMIT ?""", (ticker.upper(), limit)).fetchall()
        return [Citation(document_id=row['document_id'], chunk_id=row['chunk_id'], title=row['title'],
            source_url=row['source_url'], publisher=row['publisher'], published_at=row['published_at'], retrieved_at=row['retrieved_at'],
            content_sha256=row['content_sha256'], quote=row['text'][:1400], period_end=row['period_end'],
            reporting_period=ReportingPeriod.model_validate_json(row['reporting_period_json']),
            document_type=row['document_type'], tickers=json.loads(row['tickers_json'])) for row in rows]

    @observed("research.hybrid_retrieval", "tool")
    def search(self, query: str, ticker: str | None = None, limit: int = 5, *, as_of: date | None = None,
               document_types: list[ResearchDocumentType] | None = None, published_after: date | None = None,
               scope: ResearchScope | None = None) -> EvidenceBundle:
        """Hybrid local retrieval using vector similarity plus SQLite FTS5."""
        limit = max(1, min(20, limit))
        filters = dict(as_of=as_of, document_types=document_types, published_after=published_after, scope=scope)
        lexical = self._search_lexical(query, ticker=ticker, limit=max(limit * 4, 20), **filters)
        vector = self.search_vector(query, ticker=ticker, limit=max(limit * 4, 20), **filters)
        combined: dict[str, Citation] = {}
        scores: dict[str, float] = {}
        for rank, citation in enumerate(vector.citations, 1):
            if isinstance(self.embedding_provider, LocalHashEmbeddingProvider) and not self._has_content_anchor(query, citation, ticker):
                continue
            combined[citation.chunk_id] = citation
            if citation.retrieval_score < self._env_float("STOCK_AGENT_RAG_HYBRID_MIN_SIMILARITY", 0.12, 0, 1):
                continue
            scores[citation.chunk_id] = 1 / (60 + rank)
        for rank, citation in enumerate(lexical.citations, 1):
            if not self._has_content_anchor(query, citation, ticker):
                continue
            if citation.chunk_id in scores:
                previous = combined[citation.chunk_id]
                modes = list(dict.fromkeys([*previous.retrieval_modes, "fts5"]))
                combined[citation.chunk_id] = previous.model_copy(update={"retrieval_modes": modes})
                scores[citation.chunk_id] += 1 / (60 + rank)
            else:
                combined[citation.chunk_id] = citation
                scores[citation.chunk_id] = 1 / (60 + rank)

        ranked = [item for key, item in combined.items() if key in scores]
        today = as_of or datetime.now(timezone.utc).date()
        fresh_query = bool(re.search(r"\b(?:mới|gần đây|latest|recent|hiện tại)\b", query, re.IGNORECASE))
        def rank_score(item):
            tier_boost = {"T1": 1.12, "T2": 1.06, "T3": 1, "T4": 0.94}[item.quality_tier]
            age = max(0, (today - item.published_at.date()).days)
            freshness = 1 + 0.1 / (1 + age / 90) if fresh_query else 1
            return scores[item.chunk_id] * tier_boost * freshness
        # Within a current query, known report periods precede equally relevant older periods.
        ranked.sort(key=lambda item: (item.period_end or date.min, rank_score(item)) if scope and scope.mode == "CURRENT" else (date.min, rank_score(item)), reverse=True)
        selected: list[Citation] = []
        seen_documents: set[str] = set()
        while ranked and len(selected) < limit:
            def diverse_score(item):
                terms = set(self._search_terms(item.quote))
                redundancy = max((len(terms & set(self._search_terms(other.quote))) /
                    max(1, len(terms | set(self._search_terms(other.quote)))) for other in selected), default=0)
                relevance = rank_score(item) - 0.006 * redundancy
                return (item.period_end or date.min, relevance) if scope and scope.mode == "CURRENT" else (date.min, relevance)
            citation = max(ranked, key=diverse_score)
            ranked = [item for item in ranked if item.document_id != citation.document_id]
            score = max(0.0, min(1.0, rank_score(citation) * 30))
            selected.append(citation.model_copy(update={"retrieval_score": score}))
            seen_documents.add(citation.document_id)
            if len(selected) >= limit:
                break
        modes = sorted({mode for item in selected for mode in item.retrieval_modes})
        note = (
            f"Verified local documents matched with hybrid retrieval ({'+'.join(modes)})."
            if selected else "No verified local document matched this query."
        )
        if self.last_embedding_error:
            note += f" Vector retrieval warning: {self.last_embedding_error}"
        bundle = EvidenceBundle(query=query, citations=selected, retrieval_note=note,
            freshness_status=scope.mode if scope else "NOT_CHECKED",
            freshness_note=freshness_note(selected, historical=bool(scope and scope.mode == "HISTORY")) if scope else "")
        if scope and scope.mode == "CURRENT" and not selected:
            archived = self.search(query, ticker=ticker, limit=3, as_of=as_of, document_types=document_types, published_after=published_after)
            bundle.historical_citations = archived.citations or (self.period_overview(ticker) if ticker else [])
            bundle.freshness_status = "INSUFFICIENT_CURRENT_DATA"
            bundle.freshness_note = freshness_note(bundle.historical_citations, insufficient=True)
        return bundle

    @classmethod
    def _content_terms(cls, value):
        plain = "".join(char for char in unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
                        if unicodedata.category(char) != "Mn")
        plain = re.sub(r"\bp\s*[/ ]\s*e\b", "pe", plain)
        plain = re.sub(r"\bp\s*[/ ]\s*b\b", "pb", plain)
        for phrase, aliases in LocalHashEmbeddingProvider._ALIASES.items():
            if any(re.search(rf"\b{re.escape(alias)}\b", plain) for alias in (phrase, *aliases)):
                plain += " " + phrase + " " + " ".join(aliases)
        return {word for word in re.findall(r"[a-z0-9]+", plain) if word not in cls._STOP_WORDS and (len(word) >= 3 or word in {"pe", "pb"})}

    @classmethod
    def _has_content_anchor(cls, query, citation, ticker):
        terms = cls._content_terms(query)
        if ticker:
            terms.discard(ticker.casefold())
        if not terms:
            return bool(ticker)
        anchors = terms & cls._content_terms(citation.quote + " " + citation.title)
        acronyms = {"rsi", "macd", "pe", "pb", "roe", "roa", "eps", "capm", "dcf", "sharpe", "drawdown", "volatility", "survivorship"}
        if anchors & acronyms:
            return True
        if len(terms) == 1:
            return any(len(term) >= 5 for term in anchors)
        return len(anchors) >= 2

    def _search_lexical(self, query: str, ticker: str | None = None, limit: int = 5, *, as_of: date | None = None,
                        document_types=None, published_after: date | None = None, scope=None) -> EvidenceBundle:
        cleaned = re.sub(r"\s+", " ", query).strip()
        if not cleaned:
            raise ResearchIngestionError("Research query cannot be empty.")
        terms = " OR ".join(self._search_terms(cleaned))
        if not terms:
            return EvidenceBundle(query=query, citations=[], retrieval_note="No searchable terms in the request.")
        condition, parameters = self._filters(ticker, as_of, document_types, published_after, scope)
        sql = f"""
            SELECT c.chunk_id, c.document_id, c.text, d.*
            FROM research_chunks_fts f
            JOIN research_chunks c ON c.chunk_id = f.chunk_id
            JOIN research_documents d ON d.document_id = c.document_id
            WHERE research_chunks_fts MATCH ? AND {condition}
            ORDER BY bm25(research_chunks_fts)
            LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(sql, (terms, *parameters, max(1, min(limit * 4, 200)))).fetchall()
        citations: list[Citation] = []
        seen_documents: set[str] = set()
        for rank, row in enumerate(rows, 1):
            if row["document_id"] in seen_documents:
                continue
            citations.append(Citation(
                document_id=row["document_id"], chunk_id=row["chunk_id"], title=row["title"],
                source_url=row["source_url"], publisher=row["publisher"],
                published_at=datetime.fromisoformat(row["published_at"]),
                retrieved_at=datetime.fromisoformat(row["retrieved_at"]), content_sha256=row["content_sha256"],
                quote=row["text"][:1400], retrieval_score=1.0 / rank, retrieval_modes=["fts5"],
                quality_tier=row["quality_tier"], version=row["version"], period_end=row["period_end"],
                reporting_period=ReportingPeriod.model_validate_json(row["reporting_period_json"]), document_type=row["document_type"], tickers=json.loads(row["tickers_json"]),
            ))
            seen_documents.add(row["document_id"])
            if len(citations) >= limit:
                break
        note = "Verified local documents matched the query." if citations else "No verified local document matched this query."
        return EvidenceBundle(query=query, citations=citations, retrieval_note=note)

    def search_vector(self, query: str, ticker: str | None = None, limit: int = 5, *, as_of: date | None = None,
                      document_types=None, published_after: date | None = None, scope=None) -> EvidenceBundle:
        if self.embedding_provider is None:
            return EvidenceBundle(query=query, citations=[], retrieval_note="Vector retrieval is disabled.")
        cleaned = re.sub(r"\s+", " ", query).strip()
        if not cleaned:
            raise ResearchIngestionError("Research query cannot be empty.")
        try:
            self._ensure_embeddings()
            query_embed = getattr(self.embedding_provider, "embed_queries", self.embedding_provider.embed)
            query_vector = np.asarray(query_embed([cleaned])[0], dtype=np.float32)
            norm = float(np.linalg.norm(query_vector))
            if not norm or not np.isfinite(query_vector).all():
                raise ValueError("Invalid query vector")
            query_vector = query_vector / norm
            model_id = self.embedding_provider.model_id
            condition, parameters = self._filters(ticker, as_of, document_types, published_after, scope)
            with self._connect() as connection:
                rows = connection.execute(
                    f"""SELECT c.chunk_id, c.document_id, c.text, d.*,
                              e.dimensions, e.vector
                       FROM research_chunk_embeddings e
                       JOIN research_chunks c ON c.chunk_id=e.chunk_id
                       JOIN research_documents d ON d.document_id=c.document_id
                       WHERE e.model_id=? AND {condition}""",
                    (model_id, *parameters),
                ).fetchall()
        except Exception as exc:
            self.last_embedding_error = f"{type(exc).__name__}: {exc}"
            return EvidenceBundle(query=query, citations=[], retrieval_note="Vector retrieval failed; FTS fallback remains available.")

        ticker_filter = ticker.upper() if ticker else None
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            if ticker_filter and ticker_filter not in json.loads(row["tickers_json"]):
                continue
            vector = np.frombuffer(row["vector"], dtype="<f4")
            if len(vector) != int(row["dimensions"]) or len(vector) != len(query_vector):
                continue
            similarity = float(np.dot(query_vector, vector))
            scored.append((max(0.0, min(1.0, similarity)), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        minimum = self._env_float("STOCK_AGENT_RAG_MIN_SIMILARITY", 0.05, 0.0, 1.0)
        citations: list[Citation] = []
        seen_documents: set[str] = set()
        for similarity, row in scored:
            if similarity < minimum or row["document_id"] in seen_documents:
                continue
            citations.append(Citation(
                document_id=row["document_id"], chunk_id=row["chunk_id"], title=row["title"],
                source_url=row["source_url"], publisher=row["publisher"],
                published_at=datetime.fromisoformat(row["published_at"]),
                retrieved_at=datetime.fromisoformat(row["retrieved_at"]), content_sha256=row["content_sha256"],
                quote=row["text"][:1400], retrieval_score=similarity, retrieval_modes=["vector"],
                quality_tier=row["quality_tier"], version=row["version"], period_end=row["period_end"],
                reporting_period=ReportingPeriod.model_validate_json(row["reporting_period_json"]), document_type=row["document_type"], tickers=json.loads(row["tickers_json"]),
            ))
            seen_documents.add(row["document_id"])
            if len(citations) >= limit:
                break
        self.last_embedding_error = None
        note = "Verified local documents matched by vector similarity." if citations else "No vector match passed the similarity threshold."
        return EvidenceBundle(query=query, citations=citations, retrieval_note=note)

    @staticmethod
    def _filters(ticker, as_of, document_types, published_after, scope=None):
        conditions, values = [], []
        if ticker:
            conditions.append("EXISTS (SELECT 1 FROM json_each(d.tickers_json) WHERE value=?)")
            values.append(ticker.upper())
        cutoff = (as_of or datetime.now(timezone.utc).date()).isoformat()
        conditions.append("date(d.published_at) <= ?")
        values.append(cutoff)
        if published_after:
            conditions.append("date(d.published_at) >= ?")
            values.append(published_after.isoformat())
        if document_types:
            conditions.append("d.document_type IN (" + ",".join("?" for _ in document_types) + ")")
            values.extend(ResearchDocumentType(item).value for item in document_types)
        conditions.append("NOT EXISTS (SELECT 1 FROM research_documents newer WHERE newer.supersedes_id=d.document_id AND date(newer.published_at)<=?)")
        values.append(cutoff)
        if scope:
            predicates, arguments = period_sql(scope, as_of or datetime.now(timezone.utc).date())
            conditions.extend(predicates)
            values.extend(arguments)
        return " AND ".join(conditions), values

    def _ensure_embeddings(self) -> None:
        if self.embedding_provider is None:
            return
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT c.chunk_id, c.document_id, c.ordinal, c.text, c.char_start, c.char_end
                   FROM research_chunks c
                   LEFT JOIN research_chunk_embeddings e
                     ON e.chunk_id=c.chunk_id AND e.model_id=?
                   WHERE e.chunk_id IS NULL ORDER BY c.document_id, c.ordinal""",
                (self.embedding_provider.model_id,),
            ).fetchall()
        chunks = [DocumentChunk(**dict(row)) for row in rows]
        self._embed_chunks(chunks)

    def reindex_embeddings(self, *, force: bool = False) -> int:
        if self.embedding_provider is None:
            return 0
        if force:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM research_chunk_embeddings WHERE model_id=?",
                    (self.embedding_provider.model_id,),
                )
        self._ensure_embeddings()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM research_chunk_embeddings WHERE model_id=?",
                (self.embedding_provider.model_id,),
            ).fetchone()
        return int(row["count"]) if row else 0

    def _embed_chunks(self, chunks: list[DocumentChunk]) -> None:
        if self.embedding_provider is None or not chunks:
            return
        try:
            vectors = self.embedding_provider.embed([chunk.text for chunk in chunks])
            if len(vectors) != len(chunks):
                raise RuntimeError("Embedding provider returned the wrong number of vectors")
            now = datetime.now(timezone.utc).isoformat()
            rows = []
            for chunk, vector in zip(chunks, vectors):
                normalized = np.asarray(vector, dtype="<f4")
                norm = float(np.linalg.norm(normalized))
                if not norm or normalized.ndim != 1 or not np.isfinite(normalized).all():
                    continue
                normalized /= norm
                rows.append((
                    chunk.chunk_id, self.embedding_provider.model_id, len(normalized), normalized.tobytes(), now,
                ))
            with self._connect() as connection:
                connection.executemany(
                    """INSERT OR REPLACE INTO research_chunk_embeddings
                       (chunk_id,model_id,dimensions,vector,embedded_at) VALUES (?,?,?,?,?)""",
                    rows,
                )
            self.last_embedding_error = None
        except Exception as exc:
            self.last_embedding_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
        try:
            value = float(os.getenv(name, str(default)))
        except ValueError:
            value = default
        return max(minimum, min(maximum, value))

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
        def normalized(term):
            return "".join(ch for ch in unicodedata.normalize("NFD", term.casefold().replace("đ", "d")) if unicodedata.category(ch) != "Mn")
        return [term for term in dict.fromkeys(terms) if normalized(term) not in ResearchStore._STOP_WORDS]
