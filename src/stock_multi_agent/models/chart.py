from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from stock_multi_agent.models.data import SourceReference


class ChartType(str, Enum):
    PRICE = "PRICE"
    CANDLESTICK = "CANDLESTICK"
    PRICE_MA = "PRICE_MA"
    PRICE_VOLUME = "PRICE_VOLUME"
    VOLUME = "VOLUME"
    RSI = "RSI"
    MACD = "MACD"


class ChartRequest(BaseModel):
    ticker: str
    chart_type: ChartType = ChartType.PRICE
    period: str = "6M"
    indicators: list[str] = Field(default_factory=list)


class ChartArtifact(BaseModel):
    request: ChartRequest
    title: str
    records: list[dict[str, Any]]
    data_timestamp: date
    data_fetched_at: datetime | None = None
    source: str
    source_details: list[SourceReference] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
