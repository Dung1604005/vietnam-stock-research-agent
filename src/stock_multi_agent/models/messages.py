from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class AgentName(str, Enum):
    MANAGER = "manager"
    DATA = "data"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    RISK = "risk"
    USER = "user"


class MessageType(str, Enum):
    TASK = "TASK"
    RESULT = "RESULT"
    CRITIQUE = "CRITIQUE"
    REQUEST_REANALYSIS = "REQUEST_REANALYSIS"
    WARNING = "WARNING"
    FINAL_DECISION = "FINAL_DECISION"


class AgentMessage(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    sender: AgentName
    receiver: AgentName
    message_type: MessageType
    task: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

