from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from stock_multi_agent.models.analysis import MarketData


class MarketDataCache(Protocol):
    def get(self, key: str) -> MarketData | None: ...
    def set(self, key: str, data: MarketData, ttl_seconds: int) -> None: ...


class SQLiteMarketDataCache:
    """Small process-safe cache. The default path lives under project `.runtime`."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS market_cache ("
                "cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                "created_at TEXT NOT NULL, expires_at TEXT NOT NULL)"
            )

    def get(self, key: str) -> MarketData | None:
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload, expires_at FROM market_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            try:
                expires_at = datetime.fromisoformat(row[1])
                if expires_at <= now:
                    connection.execute("DELETE FROM market_cache WHERE cache_key = ?", (key,))
                    return None
                return MarketData.model_validate_json(row[0])
            except Exception:
                connection.execute("DELETE FROM market_cache WHERE cache_key = ?", (key,))
                return None

    def set(self, key: str, data: MarketData, ttl_seconds: int) -> None:
        created_at = datetime.now(timezone.utc)
        expires_at = created_at + timedelta(seconds=max(0, ttl_seconds))
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO market_cache(cache_key, payload, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (key, data.model_dump_json(), created_at.isoformat(), expires_at.isoformat()),
            )


class RawSnapshotStore:
    """Content-addressed immutable raw snapshots for reproducible provider debugging."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        provider: str,
        data_kind: str,
        ticker: str,
        payload: Any,
        fetched_at: datetime,
        request_parameters: dict[str, Any],
    ) -> tuple[str, str]:
        envelope = {
            "provider": provider,
            "data_kind": data_kind,
            "ticker": ticker,
            "fetched_at": fetched_at.isoformat(),
            "request_parameters": request_parameters,
            "payload": payload,
        }
        encoded = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        relative = Path(provider.lower()) / fetched_at.date().isoformat() / f"{digest}.json.gz"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with gzip.open(destination, "wb") as handle:
                handle.write(encoded)
        return digest, relative.as_posix()
