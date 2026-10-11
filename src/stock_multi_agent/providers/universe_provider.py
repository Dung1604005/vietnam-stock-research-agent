from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from stock_multi_agent.models.data import DataKind, Instrument, InstrumentType, SourceReference
from stock_multi_agent.models.screening import UniverseDefinition
from stock_multi_agent.providers.data_store import RawSnapshotStore
from stock_multi_agent.providers.market_provider import MarketDataError, VnstockMarketDataProvider


@dataclass(frozen=True)
class InstrumentUniverseSnapshot:
    """A retrieval-time capture of a current listing endpoint."""

    instruments: list[Instrument]
    source_reference: SourceReference


class InstrumentUniverseProvider(Protocol):
    def fetch_current(self) -> InstrumentUniverseSnapshot: ...


class CurrentUniverseCache:
    """A short-lived convenience cache; immutable raw snapshots remain the evidence record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, ttl_seconds: int) -> InstrumentUniverseSnapshot | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            source = SourceReference.model_validate(payload["source_reference"])
            if datetime.now(timezone.utc) - source.fetched_at > timedelta(seconds=ttl_seconds):
                return None
            instruments = [Instrument.model_validate(item) for item in payload["instruments"]]
            return InstrumentUniverseSnapshot(instruments=instruments, source_reference=source)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def set(self, snapshot: InstrumentUniverseSnapshot) -> None:
        payload = {
            "source_reference": snapshot.source_reference.model_dump(mode="json"),
            "instruments": [instrument.model_dump(mode="json") for instrument in snapshot.instruments],
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)


class VnstockInstrumentUniverseProvider:
    """Current Vietnam equity listings from VCI through vnstock, with snapshot provenance."""

    SCHEMA_VERSION = "vnstock-vci-instrument-universe-v1"
    SOURCE_URL = "https://trading.vietcap.com.vn/api/price/symbols/getAll"
    ACTIVE_EXCHANGES = frozenset({"HSX", "HOSE", "HNX", "UPCOM"})

    def __init__(
        self,
        cache: CurrentUniverseCache | None = None,
        snapshot_store: RawSnapshotStore | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        runtime_root = VnstockMarketDataProvider._configure_runtime_storage()
        self.cache = cache or CurrentUniverseCache(runtime_root / "cache" / "current_instrument_universe_v2.json")
        self.snapshot_store = snapshot_store or RawSnapshotStore(runtime_root / "snapshots")
        self.cache_ttl_seconds = cache_ttl_seconds if cache_ttl_seconds is not None else self._env_int(
            "STOCK_AGENT_UNIVERSE_CACHE_TTL_SECONDS", 86400,
        )

    def fetch_current(self) -> InstrumentUniverseSnapshot:
        cached = self.cache.get(self.cache_ttl_seconds)
        if cached is not None:
            return cached
        try:
            from vnstock.api.listing import Listing

            fetched_at = datetime.now(timezone.utc)
            frame = Listing(source="VCI", show_log=False).symbols_by_exchange(lang="vi")
        except Exception as exc:
            raise MarketDataError(f"Unable to fetch the current VCI instrument universe: {exc}") from exc
        instruments = self._normalize(frame)
        if not instruments:
            raise MarketDataError("VCI current instrument universe returned no listed equities.")
        digest, snapshot_id = self.snapshot_store.save(
            provider="vnstock-VCI",
            data_kind=DataKind.INSTRUMENT_UNIVERSE.value,
            ticker="ALL",
            payload=self._frame_payload(frame),
            fetched_at=fetched_at,
            request_parameters={"endpoint": "symbols_by_exchange", "lang": "vi", "asset_type": "STOCK"},
        )
        source = SourceReference(
            provider="vnstock",
            source_id="VCI:Listing.symbols_by_exchange:current",
            data_kind=DataKind.INSTRUMENT_UNIVERSE,
            fetched_at=fetched_at,
            source_url=self.SOURCE_URL,
            request_parameters={"lang": "vi", "asset_type": "STOCK"},
            schema_version=self.SCHEMA_VERSION,
            content_sha256=digest,
            snapshot_id=snapshot_id,
        )
        snapshot = InstrumentUniverseSnapshot(instruments=instruments, source_reference=source)
        self.cache.set(snapshot)
        return snapshot

    @staticmethod
    def _normalize(frame: Any) -> list[Instrument]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return []
        data = frame.copy()
        data.columns = [str(column).strip().lower() for column in data.columns]
        if not {"symbol", "exchange", "type"}.issubset(data.columns):
            return []
        data = data[data["type"].astype(str).str.upper().eq("STOCK")]
        result: list[Instrument] = []
        seen: set[str] = set()
        for row in data.to_dict(orient="records"):
            ticker = str(row.get("symbol", "")).strip().upper()
            if not re.fullmatch(r"[A-Z][A-Z0-9]{1,9}", ticker) or ticker in seen:
                continue
            seen.add(ticker)
            name = str(row.get("organ_name", "")).strip() or None
            exchange = str(row.get("exchange", "")).strip().upper() or None
            # VCI includes a DELISTED board in its catalogue. It is valuable
            # evidence for a future historical source, but is not part of a
            # current investable listing universe.
            if exchange not in VnstockInstrumentUniverseProvider.ACTIVE_EXCHANGES:
                continue
            result.append(Instrument(
                ticker=ticker, name=name, exchange=exchange,
                instrument_type=InstrumentType.EQUITY, currency="VND",
                active=True,
            ))
        return sorted(result, key=lambda item: item.ticker)

    @staticmethod
    def _frame_payload(frame: Any) -> Any:
        if isinstance(frame, pd.DataFrame):
            return json.loads(frame.to_json(orient="split", date_format="iso", default_handler=str))
        return frame

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return max(0, int(os.getenv(name, str(default))))
        except ValueError:
            return default


def universe_definition_from_snapshot(key: str, snapshot: InstrumentUniverseSnapshot) -> UniverseDefinition:
    """Turn a current capture into a disclosed universe without fabricating historical membership."""

    return UniverseDefinition(
        key=key,
        tickers=[item.ticker for item in snapshot.instruments],
        source="VCI current listing catalogue via vnstock",
        membership_as_of=None,
        captured_at=snapshot.source_reference.fetched_at,
        source_reference=snapshot.source_reference,
        point_in_time_supported=False,
    )
