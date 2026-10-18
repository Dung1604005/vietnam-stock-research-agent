"""Browser navigation actions, deliberately not trading instructions."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InteractionAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    label: str = Field(min_length=1, max_length=240)
    url: str
    ticker: str | None = None
    kind: Literal['STOCK', 'QUOTE', 'CHART', 'REPORT', 'SOURCE', 'BROKER']
    description: str | None = None
    published_at: str | None = None
    period_label: str | None = None
