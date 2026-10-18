"""Small persistent queue sharing ResearchStore's SQLite lifecycle."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pydantic import HttpUrl

from stock_multi_agent.models.research_crawler import CandidateStatus, DocumentCandidate, StorageMode, utcnow
from stock_multi_agent.services.source_registry import canonical_url


class CandidateRepository:
    def __init__(self, store):
        self.store = store
        with store._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS research_candidates (
                    candidate_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, source_url TEXT NOT NULL,
                    status TEXT NOT NULL, payload TEXT NOT NULL, updated_at TEXT NOT NULL,
                    UNIQUE(source_id, source_url));
                CREATE INDEX IF NOT EXISTS crawl_status_idx ON research_candidates(source_id,status);
                CREATE TABLE IF NOT EXISTS research_sources_state (
                    source_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS research_crawl_runs (
                    run_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS research_crawl_failures (
                    candidate_id TEXT PRIMARY KEY, error_type TEXT NOT NULL, attempts INTEGER NOT NULL,
                    next_retry TEXT, permanent INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS research_source_locks (
                    source_id TEXT PRIMARY KEY, owner TEXT NOT NULL, expires_at TEXT NOT NULL);
            """)

    def upsert(self, candidate: DocumentCandidate) -> DocumentCandidate:
        candidate = candidate.model_copy(deep=True)
        url = canonical_url(str(candidate.source_url))
        candidate.source_url = HttpUrl(url)
        candidate.candidate_id = hashlib.sha256(f"{candidate.source_id}:{url}".encode()).hexdigest()
        with self.store._connect() as db:
            db.execute("INSERT OR IGNORE INTO research_candidates VALUES (?,?,?,?,?,?)",
                       (candidate.candidate_id, candidate.source_id, url, candidate.status.value,
                        candidate.model_dump_json(), utcnow().isoformat()))
            row = db.execute("SELECT payload FROM research_candidates WHERE candidate_id=?", (candidate.candidate_id,)).fetchone()
            existing = DocumentCandidate.model_validate_json(row[0])
            newer_listing = (candidate.published_at is not None and existing.published_at is not None
                and candidate.published_at > existing.published_at and not existing.date_evidence.startswith("User reviewed:"))
            if newer_listing and existing.status != CandidateStatus.REJECTED:
                existing.title = candidate.title or existing.title
                existing.published_at, existing.date_evidence = candidate.published_at, candidate.date_evidence
                existing.reporting_period, existing.period_end = candidate.reporting_period, candidate.period_end
                existing.status, existing.reason = CandidateStatus.DISCOVERED, "LISTING_REVISION"
                existing.etag, existing.last_modified, existing.last_checked_at = None, None, None
                db.execute("UPDATE research_candidates SET status=?,payload=?,updated_at=? WHERE candidate_id=?",
                           (existing.status.value, existing.model_dump_json(), utcnow().isoformat(), candidate.candidate_id))
            if not existing.approved_full_text and existing.status not in {CandidateStatus.REJECTED, CandidateStatus.INDEXED, CandidateStatus.UNCHANGED}:
                if candidate.title and len(candidate.title) > len(existing.title):
                    existing.title = candidate.title
                if not existing.published_at and candidate.published_at:
                    existing.published_at, existing.date_evidence = candidate.published_at, candidate.date_evidence
                if not existing.tickers:
                    existing.tickers = candidate.tickers
                if existing.reporting_period.status == "UNKNOWN" and candidate.reporting_period.status != "UNKNOWN":
                    existing.reporting_period, existing.period_end = candidate.reporting_period, candidate.period_end
                db.execute("UPDATE research_candidates SET payload=? WHERE candidate_id=?", (existing.model_dump_json(), candidate.candidate_id))
        return existing

    def save(self, candidate: DocumentCandidate):
        with self.store._connect() as db:
            db.execute("UPDATE research_candidates SET status=?,payload=?,updated_at=? WHERE candidate_id=?",
                       (candidate.status.value, candidate.model_dump_json(), utcnow().isoformat(), candidate.candidate_id))

    def get(self, candidate_id: str) -> DocumentCandidate:
        with self.store._connect() as db:
            row = db.execute("SELECT payload FROM research_candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
        if row is None:
            raise ValueError("Candidate not found")
        return DocumentCandidate.model_validate_json(row[0])

    def list(self, *, source_id=None, status=None, limit=100):
        clauses, params = [], []
        if source_id:
            clauses.append("source_id=?")
            params.append(source_id)
        if status:
            clauses.append("status=?")
            params.append(str(getattr(status, "value", status)))
        with self.store._connect() as db:
            rows = db.execute("SELECT payload FROM research_candidates" + (" WHERE " + " AND ".join(clauses) if clauses else "")
                              + " ORDER BY updated_at DESC,candidate_id LIMIT ?", (*params, limit)).fetchall()
        return [DocumentCandidate.model_validate_json(row[0]) for row in rows]

    def pending(self, source, *, limit=100):
        now = utcnow()
        # Do not LIMIT before eligibility checks: old indexed items must not starve fresh discoveries.
        values = self.list(source_id=source.source_id, limit=100_000)
        def due(item):
            if item.status == CandidateStatus.QUARANTINED and item.reason == "PERMISSION_REQUIRED":
                return source.storage_mode == StorageMode.FULL_TEXT
            if item.status == CandidateStatus.RETRYABLE_ERROR:
                return item.next_attempt_at is None or item.next_attempt_at <= now
            if item.status in {CandidateStatus.DISCOVERED, CandidateStatus.FETCHED}:
                return True
            if item.status in {CandidateStatus.INDEXED, CandidateStatus.UNCHANGED}:
                return item.last_checked_at is None or now - item.last_checked_at >= timedelta(hours=source.refresh_hours)
            return False
        eligible = [item for item in values if due(item)]
        eligible.sort(key=lambda item: (bool(item.document_id), item.discovered_at, item.candidate_id))
        return eligible[:limit]

    def review(self, candidate_id: str, action: str, *, note="", published_at=None, title=None, tickers=None, document_type=None):
        item = self.get(candidate_id)
        if action == "approve":
            if not note.strip():
                raise ValueError("Approval requires a usage/permission note")
            item.approved_full_text, item.approval_note = True, note.strip()
        elif action == "reject":
            item.status, item.reason = CandidateStatus.REJECTED, "USER_REJECTED"
            self.save(item)
            return item
        elif action != "retry":
            raise ValueError("Unknown review action")
        if published_at is not None:
            item.published_at = published_at
            item.date_evidence = "User reviewed: " + note
        if title:
            item.title = title
        if tickers is not None:
            item.tickers = [value.upper() for value in tickers]
        if document_type is not None:
            item.document_type = document_type
        # Validate assignments, including timezone supplied by a reviewer.
        item.status, item.reason, item.next_attempt_at = CandidateStatus.DISCOVERED, "REVIEWED", None
        item = DocumentCandidate.model_validate(item.model_dump())
        self.save(item)
        return item

    def state(self, source_id):
        with self.store._connect() as db:
            row = db.execute("SELECT payload FROM research_sources_state WHERE source_id=?", (source_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def save_state(self, source_id, state):
        with self.store._connect() as db:
            db.execute("INSERT OR REPLACE INTO research_sources_state VALUES (?,?)", (source_id, json.dumps(state)))

    def acquire(self, source_id, owner, seconds=660):
        now = utcnow()
        with self.store._connect() as db:
            db.execute("DELETE FROM research_source_locks WHERE expires_at <= ?", (now.isoformat(),))
            result = db.execute("INSERT OR IGNORE INTO research_source_locks VALUES (?,?,?)",
                                (source_id, owner, (now + timedelta(seconds=seconds)).isoformat()))
        return result.rowcount == 1

    def release(self, source_id, owner):
        with self.store._connect() as db:
            db.execute("DELETE FROM research_source_locks WHERE source_id=? AND owner=?", (source_id, owner))

    def record_run(self, report):
        with self.store._connect() as db:
            db.execute("INSERT OR REPLACE INTO research_crawl_runs VALUES (?,?)", (report.run_id, report.model_dump_json()))

    def record_failure(self, candidate):
        with self.store._connect() as db:
            db.execute("INSERT OR REPLACE INTO research_crawl_failures VALUES (?,?,?,?,?)",
                       (candidate.candidate_id, candidate.reason, candidate.attempts,
                        candidate.next_attempt_at.isoformat() if candidate.next_attempt_at else None,
                        int(candidate.status == CandidateStatus.PERMANENT_ERROR)))

    def health(self):
        with self.store._connect() as db:
            states = {row[0]: json.loads(row[1]) for row in db.execute("SELECT * FROM research_sources_state")}
            counts = {row[0]: row[1] for row in db.execute("SELECT status,count(*) FROM research_candidates GROUP BY status")}
        summary = {}
        for source_id, state in states.items():
            item = {key: state.get(key) for key in ("last_run_at", "last_success_at", "last_queue_run_at", "next_due_at", "consecutive_errors")}
            item["pending_listing_pages"] = len(state.get("frontier", []))
            item["due"] = not state.get("next_due_at") or datetime.fromisoformat(state["next_due_at"]) <= utcnow()
            item["stale"] = not state.get("last_success_at") or bool(state.get("consecutive_errors")) or item["due"]
            summary[source_id] = item
        return {"sources": summary, "candidate_counts": counts}
