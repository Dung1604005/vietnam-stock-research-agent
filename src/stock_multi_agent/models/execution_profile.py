"""Immutable request execution policy. Selection never modifies process environment."""
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionMode(str, Enum):
    STANDARD = 'STANDARD'
    QUICK = 'QUICK'
    DEEP = 'DEEP'
    OFFLINE = 'OFFLINE'


class ExecutionProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    mode: ExecutionMode = ExecutionMode.STANDARD
    max_llm_calls: int | None = Field(default=None, ge=0, le=10)
    max_tool_calls: int | None = Field(default=None, ge=0, le=300)
    deadline_seconds: float | None = Field(default=None, gt=0, le=600)
    external_allowed: bool = True
    description: str = 'Preserve existing intent-specific budgets.'

    @model_validator(mode='after')
    def offline_is_default_deny(self):
        if self.mode == ExecutionMode.OFFLINE and (self.external_allowed or self.max_llm_calls != 0):
            raise ValueError('OFFLINE must deny outbound access and LLM calls')
        return self


class DataOrigin(str, Enum):
    LIVE_FETCH = 'LIVE_FETCH'
    CACHE = 'CACHE'
    SNAPSHOT_REPLAY = 'SNAPSHOT_REPLAY'
    MODEL_ESTIMATE = 'MODEL_ESTIMATE'
