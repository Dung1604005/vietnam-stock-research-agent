from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SemanticTelemetryEvent:
    parser_source: str
    parser_latency_ms: int
    llm_attempted: bool
    llm_latency_ms: int | None
    fallback_reason: str | None
    ambiguity_count: int
    needs_clarification: bool
    prompt_version: str
    example_bank_version: str
    token_usage: int | None = None
    cost_usd: float | None = None
    occurred_at: datetime | None = None


class SemanticTelemetryStore(Protocol):
    def record(self, event: SemanticTelemetryEvent) -> None: ...


class NullSemanticTelemetryStore:
    """Explicit opt-out used by deterministic offline evaluation."""

    def record(self, event: SemanticTelemetryEvent) -> None:
        del event


class SQLiteSemanticTelemetryStore:
    """Privacy-minimal, durable parser telemetry stored inside workspace runtime."""

    def __init__(self, database_path: Path | None = None) -> None:
        self.database_path = database_path or self._runtime_root() / "telemetry" / "semantic.sqlite3"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def _runtime_root() -> Path:
        default_root = Path(__file__).resolve().parents[3] / ".runtime"
        runtime_root = Path(os.getenv("STOCK_AGENT_RUNTIME_DIR", str(default_root))).resolve()
        if os.name == "nt" and runtime_root.drive.upper() != "D:":
            raise RuntimeError(f"Workspace policy requires telemetry on drive D:, got {runtime_root}")
        return runtime_root

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_parse_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    parser_source TEXT NOT NULL,
                    parser_latency_ms INTEGER NOT NULL,
                    llm_attempted INTEGER NOT NULL,
                    llm_latency_ms INTEGER,
                    fallback_reason TEXT,
                    ambiguity_count INTEGER NOT NULL,
                    needs_clarification INTEGER NOT NULL,
                    prompt_version TEXT NOT NULL,
                    example_bank_version TEXT NOT NULL,
                    token_usage INTEGER,
                    cost_usd REAL
                )
                """
            )

    def record(self, event: SemanticTelemetryEvent) -> None:
        occurred_at = event.occurred_at or datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO semantic_parse_events (
                    occurred_at, parser_source, parser_latency_ms, llm_attempted,
                    llm_latency_ms, fallback_reason, ambiguity_count,
                    needs_clarification, prompt_version, example_bank_version,
                    token_usage, cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    occurred_at.isoformat(), event.parser_source, event.parser_latency_ms,
                    int(event.llm_attempted), event.llm_latency_ms, event.fallback_reason,
                    event.ambiguity_count, int(event.needs_clarification),
                    event.prompt_version, event.example_bank_version,
                    event.token_usage, event.cost_usd,
                ),
            )

    def summary(self) -> dict[str, object]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT parser_source, fallback_reason, parser_latency_ms, llm_latency_ms, token_usage, cost_usd "
                "FROM semantic_parse_events ORDER BY id"
            ).fetchall()
        latencies = sorted(row[2] for row in rows)
        percentile_index = max(0, round((len(latencies) - 1) * 0.95)) if latencies else 0
        sources: dict[str, int] = {}
        fallback_reasons: dict[str, int] = {}
        for source, fallback_reason, *_ in rows:
            sources[source] = sources.get(source, 0) + 1
            if fallback_reason:
                fallback_reasons[fallback_reason] = fallback_reasons.get(fallback_reason, 0) + 1
        known_tokens = [row[4] for row in rows if row[4] is not None]
        known_costs = [row[5] for row in rows if row[5] is not None]
        return {
            "events": len(rows),
            "parser_sources": sources,
            "fallback_reasons": fallback_reasons,
            "latency_ms": {
                "mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
                "p95": latencies[percentile_index] if latencies else None,
            },
            "token_usage": {"known_events": len(known_tokens), "total": sum(known_tokens) if known_tokens else None},
            "cost_usd": {"known_events": len(known_costs), "total": round(sum(known_costs), 6) if known_costs else None},
        }


def configured_semantic_telemetry_store() -> SemanticTelemetryStore | None:
    if os.getenv("STOCK_AGENT_SEMANTIC_TELEMETRY_ENABLED", "true").strip().lower() in {"0", "false", "off"}:
        return None
    return SQLiteSemanticTelemetryStore()
