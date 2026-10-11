from __future__ import annotations

import json
from pathlib import Path

from stock_multi_agent.models.screening import UniverseDefinition
from stock_multi_agent.providers.universe_provider import (
    InstrumentUniverseProvider,
    VnstockInstrumentUniverseProvider,
    universe_definition_from_snapshot,
)


class UniverseService:
    """Configured watchlists plus explicit retrieval-time VCI listing universes."""

    CURRENT_UNIVERSE_KEYS = ("market_current", "hose_current", "hnx_current", "upcom_current")

    def __init__(
        self,
        groups: dict[str, list[str]] | None = None,
        path: Path | None = None,
        current_provider: InstrumentUniverseProvider | None = None,
    ) -> None:
        if groups is None:
            source_path = path or Path(__file__).resolve().parents[3] / "config" / "watchlist.json"
            with source_path.open(encoding="utf-8") as handle:
                groups = json.load(handle)
            source = f"config:{source_path.name}"
        else:
            source = "injected:configured-watchlist"
        self._universes = {
            key: UniverseDefinition(
                key=key,
                tickers=list(dict.fromkeys(ticker.upper() for ticker in tickers)),
                source=source,
                # A JSON watchlist has no historical membership evidence.
                membership_as_of=None,
                point_in_time_supported=False,
            )
            for key, tickers in groups.items()
            if tickers
        }
        if "default" not in self._universes:
            raise ValueError("Configured universe must contain a non-empty default watchlist.")
        self._current_provider = current_provider

    def select(self, key: str | None) -> UniverseDefinition:
        if key in self.CURRENT_UNIVERSE_KEYS:
            return self._current_universe(key)
        return self._universes.get(key or "", self._universes["default"])

    def groups(self) -> dict[str, list[str]]:
        return {key: list(value.tickers) for key, value in self._universes.items()}

    def _current_universe(self, key: str) -> UniverseDefinition:
        existing = self._universes.get(key)
        if existing is not None:
            return existing
        provider = self._current_provider or VnstockInstrumentUniverseProvider()
        snapshot = provider.fetch_current()
        definition = universe_definition_from_snapshot(key, snapshot)
        if key != "market_current":
            exchange = key.removesuffix("_current").upper()
            aliases = {"HOSE": {"HOSE", "HSX"}}
            allowed_exchanges = aliases.get(exchange, {exchange})
            allowed = [item.ticker for item in snapshot.instruments if item.exchange in allowed_exchanges]
            if not allowed:
                raise ValueError(f"Current VCI catalogue contains no listed equities for {exchange}.")
            definition = definition.model_copy(update={"tickers": allowed})
        self._universes[key] = definition
        return definition
