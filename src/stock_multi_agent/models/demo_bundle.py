"""Full-history local replay, distinct from last-bar analysis snapshots."""
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from stock_multi_agent.models.analysis import MarketData
from stock_multi_agent.models.data import DataKind


class DemoBundle(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal['demo-replay-v1'] = 'demo-replay-v1'
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rights_note: str = Field(min_length=12)
    redistribution_allowed: bool = False
    origin: Literal['SNAPSHOT_REPLAY'] = 'SNAPSHOT_REPLAY'
    datasets: dict[str, MarketData]
    recorded_forecasts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=lambda: [
        'Historical replay; original source dates are not updated to today.',
        'Rights declaration is supplied by the capturer, not a legal permission verifier.',
        'LLM generation and outbound retrieval are disabled in OFFLINE mode.'])

    @field_validator('captured_at')
    @classmethod
    def aware(cls, value):
        if value.utcoffset() is None:
            raise ValueError('Replay capture timestamp must be timezone-aware')
        return value

    @field_validator('rights_note')
    @classmethod
    def meaningful_rights_note(cls, value):
        if len(value.strip()) < 12:
            raise ValueError('Replay rights note must describe permitted usage, not whitespace')
        return value.strip()

    @model_validator(mode='after')
    def coherent(self):
        if not self.datasets:
            raise ValueError('Replay requires at least one real captured dataset')
        for ticker, data in self.datasets.items():
            if ticker != data.ticker or not data.source_refs:
                raise ValueError('Replay ticker and source provenance must be present and consistent')
            if data.as_of != data.bars[-1].date:
                raise ValueError('Replay price as-of must match the final recorded OHLCV session')
            price_sources = [source for source in data.source_refs if source.data_kind == DataKind.OHLCV]
            if not price_sources or not any(source.as_of == data.as_of for source in price_sources):
                raise ValueError('Replay needs OHLCV provenance with the original price as-of date')
            if data.fetched_at > self.captured_at:
                raise ValueError('Capture cannot precede original retrieval')
            if any(source.fetched_at > self.captured_at for source in data.source_refs):
                raise ValueError('Capture cannot precede a source retrieval')
        if set(self.recorded_forecasts) - set(self.datasets):
            raise ValueError('Forecast ticker missing from replay data')
        return self
