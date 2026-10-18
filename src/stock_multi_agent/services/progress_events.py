"""Thread-safe real task lifecycle events. Workers never call Streamlit."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from threading import RLock
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ProgressEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    event_id: str
    request_id: str
    sequence: int
    node: str
    attempt: int
    status: Literal['STARTED', 'SUCCEEDED', 'PARTIAL', 'FAILED', 'SKIPPED', 'FALLBACK']
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detail: str = ''
    latency_ms: int | None = None


class ProgressSink:
    def __init__(self, request_id: str | None = None):
        self.request_id = request_id or str(uuid4())
        self._events: list[ProgressEvent] = []
        self._attempts: dict[str, int] = {}
        self._cursor = 0
        self._lock = RLock()

    def emit(self, node, status, *, request_id='', detail='', latency_ms=None, attempt=None):
        with self._lock:
            if status == 'STARTED':
                self._attempts[node] = self._attempts.get(node, 0) + 1
            sequence = len(self._events) + 1
            attempt = attempt or self._attempts.get(node, 1)
            identifier = request_id or self.request_id
            event = ProgressEvent(event_id=f'{identifier}:{node}:{attempt}:{sequence}',
                request_id=identifier, sequence=sequence, node=node, attempt=attempt,
                status=status, detail=detail[:240], latency_ms=latency_ms)
            self._events.append(event)
            return event

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [event.model_dump(mode='json') for event in self._events]

    def drain(self) -> list[dict]:
        with self._lock:
            result = [event.model_dump(mode='json') for event in self._events[self._cursor:]]
            self._cursor = len(self._events)
            return result


_sink: ContextVar[ProgressSink | None] = ContextVar('progress_event_sink', default=None)


def current_progress_sink() -> ProgressSink | None:
    return _sink.get()


@contextmanager
def progress_scope(sink: ProgressSink | None = None):
    selected = sink or ProgressSink()
    token = _sink.set(selected)
    try:
        yield selected
    finally:
        _sink.reset(token)
