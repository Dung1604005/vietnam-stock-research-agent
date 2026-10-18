"""Request-scoped counters; persist only allowlisted metadata, never prompts/secrets."""
from __future__ import annotations

import functools
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from threading import RLock

from stock_multi_agent.models.research_contracts import RequestBudget


class BudgetExceeded(RuntimeError):
    pass


_current: ContextVar[RequestMetrics | None] = ContextVar("research_request", default=None)


class RequestMetrics:
    def __init__(self, budget: RequestBudget | None = None):
        self.request_id = str(uuid4())
        self.budget = budget or RequestBudget()
        self.started = time.perf_counter()
        self.llm_calls = 0
        self.tool_calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.source_count = 0
        self.api_calls = 0
        self.retries = 0
        self.output_token_upper_bound = 0
        self._lock = RLock()
        self.nodes: list[dict] = []

    def consume(self, kind: str):
        with self._lock:
            self._consume(kind)

    def _consume(self, kind: str):
        if time.perf_counter() - self.started >= self.budget.deadline_seconds:
            raise BudgetExceeded("Request deadline exceeded")
        if kind == "llm":
            if self.llm_calls >= self.budget.max_llm_calls:
                raise BudgetExceeded("Request LLM call budget exhausted")
            self.llm_calls += 1
        elif kind == "tool":
            if self.tool_calls >= self.budget.max_tool_calls:
                raise BudgetExceeded("Request tool budget exhausted")
            self.tool_calls += 1

    def snapshot(self) -> dict:
        return {"request_id": self.request_id, "latency_ms": round((time.perf_counter() - self.started) * 1000),
                "llm_calls": self.llm_calls, "tool_calls": self.tool_calls, "cache_hits": self.cache_hits,
                "failures": self.failures, "source_count": self.source_count, "token_usage": None,
                "output_token_upper_bound": self.output_token_upper_bound,
                "api_calls": self.api_calls, "retries": self.retries,
                "nodes": list(self.nodes), "budget": self.budget.model_dump(mode="json")}

    def persist(self):
        if os.getenv("STOCK_AGENT_REQUEST_TELEMETRY_ENABLED", "true").lower() in {"false", "0", "off"}:
            return
        root = Path(os.getenv("STOCK_AGENT_RUNTIME_DIR", str(Path(__file__).resolve().parents[3] / ".runtime")))
        if os.name == "nt" and root.resolve().drive.upper() != "D:":
            return
        try:
            destination = root / "telemetry" / "requests.sqlite3"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(destination) as connection:
                connection.execute("CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, created_at TEXT, metrics_json TEXT)")
                connection.execute("INSERT INTO requests VALUES (?,?,?)", (
                    self.request_id, datetime.now(timezone.utc).isoformat(), json.dumps(self.snapshot())))
        except (OSError, sqlite3.Error):
            logging.getLogger(__name__).warning("request_telemetry_write_failed")


def current_metrics() -> RequestMetrics | None:
    return _current.get()


@contextmanager
def request_scope(budget: RequestBudget | None = None):
    metrics = RequestMetrics(budget)
    token = _current.set(metrics)
    try:
        yield metrics
    finally:
        _current.reset(token)
        metrics.persist()


def observed(name: str, kind: str | None = None):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            from stock_multi_agent.services.progress_events import current_progress_sink
            metrics = current_metrics()
            sink = current_progress_sink()
            started_event = sink.emit(name, 'STARTED', request_id=metrics.request_id if metrics else '') if sink else None
            started = time.perf_counter()
            status = "ok"
            terminal_status = 'SUCCEEDED'
            try:
                if kind and metrics:
                    metrics.consume(kind)
                result = function(*args, **kwargs)
                if isinstance(result, dict):
                    reports = result.get('agent_reports', [])
                    statuses = [item.get('status') if isinstance(item, dict) else getattr(item, 'status', None) for item in reports]
                    if any(item in {'PARTIAL', 'UNAVAILABLE'} for item in statuses):
                        terminal_status = 'PARTIAL'
                return result
            except Exception:
                if metrics:
                    with metrics._lock:
                        metrics.failures += 1
                status = "error"
                raise
            finally:
                elapsed = round((time.perf_counter() - started) * 1000)
                if metrics:
                    with metrics._lock:
                        metrics.nodes.append({"node": name, "status": status, "latency_ms": elapsed})
                if sink:
                    sink.emit(name, terminal_status if status == 'ok' else 'FAILED',
                        request_id=metrics.request_id if metrics else '',
                        attempt=started_event.attempt, latency_ms=elapsed)
        return wrapped
    return decorate
