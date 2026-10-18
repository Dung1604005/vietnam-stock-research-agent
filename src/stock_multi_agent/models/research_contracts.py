"""Versioned contracts shared by planning, retrieval and answer auditing."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["research-v1"] = "research-v1"


class RequestBudget(Contract):
    max_llm_calls: int = Field(default=2, ge=0, le=10)
    max_tool_calls: int = Field(default=40, ge=0, le=300)
    max_output_tokens: int = Field(default=4000, ge=1)
    deadline_seconds: float = Field(default=120, gt=0, le=600)


class Evidence(Contract):
    evidence_id: str = Field(min_length=1)
    source_url: HttpUrl
    publisher: str = Field(min_length=1)
    published_at: datetime | None = None
    fetched_at: datetime
    as_of: date | None = None
    quality_tier: Literal["T1", "T2", "T3", "T4"] = "T4"
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    quote: str = Field(min_length=1)

    @field_validator("published_at", "fetched_at")
    @classmethod
    def aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("Evidence timestamps must include timezone")
        return value


class ToolCallRequest(Contract):
    tool: str = Field(pattern=r"^[a-z][a-z0-9_.]{0,79}$")
    arguments: dict[str, Any] = Field(default_factory=dict)
    required_evidence_ids: list[str] = Field(default_factory=list)


class AgentFinding(Contract):
    agent: str
    claim: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    data_quality: Literal["GOOD", "LIMITED", "UNAVAILABLE"]
    decision: str | None = None
    quantitative: bool = False

    @model_validator(mode="after")
    def grounded_numbers(self) -> AgentFinding:
        if self.quantitative and not self.evidence_ids:
            raise ValueError("Quantitative findings require evidence IDs")
        return self


class Conflict(Contract):
    kind: Literal["DATA", "TIME", "METRIC", "INTERPRETATION", "POLICY"]
    agents: list[str] = Field(min_length=2)
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class FinalAnswer(Contract):
    status: Literal["GROUNDED", "PARTIAL", "INSUFFICIENT_EVIDENCE"]
    text: str
    findings: list[AgentFinding] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def citations_exist(self) -> FinalAnswer:
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate evidence IDs")
        if any(set(item.evidence_ids).difference(ids) for item in self.findings):
            raise ValueError("Finding refers to unknown evidence")
        if self.status == "GROUNDED" and not self.evidence:
            raise ValueError("Grounded answers require evidence")
        return self
