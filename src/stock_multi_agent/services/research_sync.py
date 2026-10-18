"""One bounded sync run, reusable by CLI, scheduler and the review UI."""
from __future__ import annotations

import random
import time
from datetime import date, datetime, timedelta
from uuid import uuid4

from stock_multi_agent.models.research_crawler import CandidateStatus, CrawlRun, DocumentCandidate, StorageMode, utcnow
from stock_multi_agent.services.content_extractor import HtmlContentExtractor
from stock_multi_agent.services.content_fetcher import ContentFetchError
from stock_multi_agent.services.document_metadata import enrich_candidate, metadata_problem
from stock_multi_agent.services.ingestion_policy import ingestion_decision
from stock_multi_agent.services.research_crawl_store import CandidateRepository
from stock_multi_agent.services.research_crawl_transport import CrawlBudgetExceeded, CrawlPolicyError, CrawlTransport
from stock_multi_agent.services.research_discovery import ResearchDiscovery, is_navigation
from stock_multi_agent.services.research_ingestion import IngestionEntry, ResearchIngestionService
from stock_multi_agent.services.research_store import ResearchStore
from stock_multi_agent.services.source_registry import SourceRegistry


class ResearchSyncService:
    def __init__(self, store=None, registry=None, *, transport_factory=None):
        self.store = store or ResearchStore()
        self.registry = registry or SourceRegistry()
        self.repository = CandidateRepository(self.store)
        self.transport_factory = transport_factory or (lambda source, deadline: CrawlTransport(source, deadline=deadline))
        self.ingestion = ResearchIngestionService(self.store, registry=self.registry)

    def enqueue_chat(self, discoveries, *, ticker=None):
        """Persist search metadata only. Never persist a search snippet as document text."""
        result = []
        for item in discoveries:
            source = self.registry.resolve(str(item.source_url))
            if source is None or not source.accepts(str(item.source_url)) or source.storage_mode == StorageMode.BLOCKED:
                continue
            candidate = self.repository.upsert(DocumentCandidate(source_id=source.source_id, source_url=item.source_url,
                title=item.title, publisher=source.publisher, tickers=source.ticker_map or ([ticker] if ticker else []),
                storage_mode=source.storage_mode, reason="CHAT_DISCOVERY"))
            result.append({"candidate_id": candidate.candidate_id, "status": candidate.status.value, "source_url": str(candidate.source_url)})
        return result

    def sync(self, *, source_id=None, due=True, force_discovery=False, seconds=600, queue_only=False, candidate_id=None):
        if not 1 <= seconds <= 3600:
            raise ValueError("Sync deadline must be between 1 and 3600 seconds")
        report = CrawlRun(run_id=uuid4().hex)
        deadline = time.monotonic() + seconds
        owner = report.run_id
        if not self.repository.acquire("__research_sync__", owner, seconds + 60):
            report.status = "ALREADY_RUNNING"
            report.ended_at = utcnow()
            return report
        try:
            with self.store._connect() as db:
                before = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in
                    ("research_candidates", "research_documents", "research_chunks", "research_chunk_embeddings")}
            sources = [source for source in self.registry.sources if source_id is None or source.source_id == source_id]
            if source_id and not sources:
                raise ValueError("Unknown source_id")
            for source in sources:
                if time.monotonic() >= deadline:
                    report.status = "PAUSED_BUDGET"
                    break
                if not source.fetch_allowed or source.storage_mode == StorageMode.BLOCKED:
                    continue
                state = self.repository.state(source.source_id)
                pending = self.repository.pending(source, limit=source.max_pages_per_run)
                if candidate_id:
                    pending = [item for item in pending if item.candidate_id == candidate_id]
                next_due = datetime.fromisoformat(state["next_due_at"]) if state.get("next_due_at") else None
                discovery_due = source.discovery_allowed and not queue_only and (force_discovery or not due or not next_due or next_due <= utcnow())
                if not discovery_due and not pending:
                    continue
                if not self.repository.acquire(source.source_id, owner, seconds + 60):
                    continue
                transport = self.transport_factory(source, deadline)
                source_result = {"source_id": source.source_id, "status": "OK", "discovered": 0}
                with self.store._connect() as db:
                    source_before = db.execute("SELECT count(*) FROM research_candidates WHERE source_id=?", (source.source_id,)).fetchone()[0]
                state["last_run_at"] = utcnow().isoformat()
                try:
                    if discovery_due:
                        try:
                            source_result["discovered"] = ResearchDiscovery().discover(source, transport, self.repository, state,
                                force=force_discovery, max_fetches=max(1, source.max_pages_per_run // 2))
                        except CrawlBudgetExceeded:
                            raise
                        except Exception as exc:
                            source_result["status"] = "DISCOVERY_ERROR"
                            report.errors.append({"source_id": source.source_id, "stage": "discovery", "error_type": type(exc).__name__,
                                "reason": f"HTTP_{exc.status_code}" if isinstance(exc, ContentFetchError) and exc.status_code else str(exc) if isinstance(exc, CrawlPolicyError) else "DISCOVERY_FAILED"})
                        if state.get("discovery_errors"):
                            source_result["listing_errors"] = state["discovery_errors"][:20]
                            if not state.get("last_discovered_links") and source_result["status"] == "OK":
                                source_result["status"] = "DISCOVERY_ERROR"
                    pending = self.repository.pending(source, limit=source.max_pages_per_run)
                    if candidate_id:
                        pending = [item for item in pending if item.candidate_id == candidate_id]
                    for candidate in pending:
                        if time.monotonic() >= deadline:
                            raise CrawlBudgetExceeded("DEADLINE")
                        status = self._process(candidate, source, transport)
                        report.counts[status] = report.counts.get(status, 0) + 1
                        if status in {"RETRYABLE_ERROR", "PERMANENT_ERROR"}:
                            source_result["status"] = "FETCH_ERRORS"
                    if discovery_due and source_result["status"] == "OK" and source_result["discovered"] == 0:
                        has_documents = any(not is_navigation(source, str(item.source_url)) for item in self.repository.list(source_id=source.source_id, limit=100_000))
                        if not has_documents:
                            source_result["status"] = "NO_DOCUMENT_LINKS"
                    if not discovery_due:
                        state["last_queue_run_at"] = utcnow().isoformat()
                    elif source_result["status"] == "OK":
                        state["last_success_at"] = utcnow().isoformat()
                        state["consecutive_errors"] = 0
                        state["next_due_at"] = (utcnow() + timedelta(hours=source.refresh_hours)).isoformat()
                    elif discovery_due:
                        state["consecutive_errors"] = state.get("consecutive_errors", 0) + 1
                        state["next_due_at"] = (utcnow() + timedelta(hours=min(24, 2 ** min(5, state["consecutive_errors"])))).isoformat()
                except CrawlBudgetExceeded:
                    source_result["status"] = "PAUSED_BUDGET"
                    state["next_due_at"] = utcnow().isoformat()
                finally:
                    source_result["discovered"] = state.get("last_discovered_links", 0) if discovery_due else 0
                    self.repository.save_state(source.source_id, state)
                    self.repository.release(source.source_id, owner)
                    report.request_count += transport.requests
                    report.bytes_downloaded += transport.bytes_downloaded
                    with self.store._connect() as db:
                        source_result["new_candidates"] = db.execute("SELECT count(*) FROM research_candidates WHERE source_id=?", (source.source_id,)).fetchone()[0] - source_before
                    report.sources.append(source_result)
            if report.status == "RUNNING":
                report.status = "COMPLETED_WITH_ISSUES" if report.errors else "COMPLETED"
                if any(item["status"] not in {"OK", "PAUSED_BUDGET"} for item in report.sources):
                    report.status = "COMPLETED_WITH_ISSUES"
                if any(source["status"] == "PAUSED_BUDGET" for source in report.sources):
                    report.status = "PAUSED_BUDGET"
        finally:
            report.ended_at = utcnow()
            self.repository.release("__research_sync__", owner)
            self.repository.record_run(report)
            if 'before' in locals():
                with self.store._connect() as db:
                    for table, count in before.items():
                        report.counts["new_" + table.removeprefix("research_")] = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] - count
                self.repository.record_run(report)
            directory = self.store.database_path.parent / "reports"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{report.started_at:%Y%m%dT%H%M%S}-{report.run_id}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
            # Rotate only this service's own report pattern; leave user files alone.
            for path in sorted(directory.glob("????????T??????-????????????????????????????????.json"), reverse=True)[100:]:
                path.unlink()
        return report

    def _process(self, candidate, source, transport):
        if is_navigation(source, str(candidate.source_url)):
            candidate.status, candidate.reason = CandidateStatus.REJECTED, "LISTING_NOT_DOCUMENT"
            self.repository.save(candidate)
            return candidate.status.value
        decision = ingestion_decision(source, candidate)
        candidate.storage_mode = decision.storage_mode
        if not decision.allowed and decision.storage_mode != StorageMode.EXCERPT_ONLY:
            candidate.status = CandidateStatus.QUARANTINED
            candidate.reason = decision.reason
            self.repository.save(candidate)
            return candidate.status.value
        try:
            fetched = transport.fetch(str(candidate.source_url), etag=candidate.etag if candidate.document_id else None,
                                      last_modified=candidate.last_modified if candidate.document_id else None)
            if not source.accepts(str(fetched.final_url)):
                raise CrawlPolicyError("REDIRECT_OUTSIDE_SOURCE")
            candidate.last_checked_at = utcnow()
            if fetched.status_code == 304:
                if not candidate.document_id:
                    raise ValueError("Unexpected 304 without an indexed document")
                candidate.status, candidate.reason = CandidateStatus.UNCHANGED, "HTTP_NOT_MODIFIED"
            else:
                candidate.status = CandidateStatus.FETCHED
                self.repository.save(candidate)
                extracted = HtmlContentExtractor().extract(fetched)
                candidate = enrich_candidate(candidate, extracted, fetched, source)
                candidate.etag, candidate.last_modified = fetched.etag, fetched.last_modified
                # Excerpts are persisted only under the explicit source excerpt/full-text permission.
                candidate.excerpt = extracted.text[:600]
                problem = metadata_problem(candidate, extracted.text)
                if decision.storage_mode != StorageMode.FULL_TEXT:
                    problem = "PERMISSION_REQUIRED"
                if problem:
                    candidate.status, candidate.reason = CandidateStatus.QUARANTINED, problem
                else:
                    entry = IngestionEntry(source_url=candidate.source_url, title=candidate.title, publisher=candidate.publisher,
                        published_at=candidate.published_at, document_type=candidate.document_type, tickers=candidate.tickers,
                        approved_full_text=True, usage_note=candidate.approval_note or source.usage_note)
                    result = self.ingestion.ingest_extracted(entry, extracted,
                        period_end=date.fromisoformat(candidate.period_end) if candidate.period_end else None)
                    candidate.status = CandidateStatus(result["status"])
                    candidate.document_id, candidate.reason = result["document_id"], "VALIDATED"
            candidate.attempts = 0
            candidate.next_attempt_at = None
        except CrawlBudgetExceeded:
            raise
        except CrawlPolicyError as exc:
            candidate.status, candidate.reason = CandidateStatus.QUARANTINED, str(exc)
        except ContentFetchError as exc:
            candidate.attempts += 1
            retryable = exc.status_code is None or exc.status_code in {408, 429} or exc.status_code >= 500
            candidate.status = CandidateStatus.RETRYABLE_ERROR if retryable and candidate.attempts < 5 else CandidateStatus.PERMANENT_ERROR
            candidate.reason = f"HTTP_{exc.status_code}" if exc.status_code else "NETWORK_ERROR"
            candidate.next_attempt_at = utcnow() + timedelta(seconds=min(86400, 60 * 2 ** candidate.attempts) + random.uniform(0, 30)) if candidate.status == CandidateStatus.RETRYABLE_ERROR else None
            self.repository.record_failure(candidate)
        except Exception as exc:
            candidate.status = CandidateStatus.QUARANTINED
            candidate.reason = "OCR_REQUIRED" if "OCR" in str(exc) or "text layer" in str(exc) else "EXTRACTION_OR_VALIDATION_ERROR"
        self.repository.save(candidate)
        return candidate.status.value
