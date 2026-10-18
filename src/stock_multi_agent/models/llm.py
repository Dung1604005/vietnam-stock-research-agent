from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from stock_multi_agent.models.messages import AgentName


class GroundedClaim(BaseModel):
    """A proposed claim with checkable anchors, not a semantic truth certificate."""

    claim_vi: str = Field(min_length=1, max_length=600)
    source_id: str = Field(min_length=1, max_length=200)
    exact_quote: str = Field(min_length=5, max_length=1000)
    metric: str | None = Field(default=None, max_length=100)
    period_end: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    unit: str | None = Field(default=None, max_length=80)
    ticker: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9]{1,9}$")
    value: float | None = Field(default=None, allow_inf_nan=False)


class AgentFollowUpProposal(BaseModel):
    """Allowlisted work request; never an arbitrary tool invocation."""

    task_kind: Literal['FETCH_FINANCIAL_HISTORY', 'VERIFY_FINANCIAL_FACT', 'FIND_RECENT_REPORT', 'CHECK_RISK_EVENT']
    reason: str = Field(min_length=5, max_length=400)
    source_ids: list[str] = Field(default_factory=list, max_length=5)
    metric: str | None = Field(default=None, max_length=100)
    period_end: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class AgentLLMReview(BaseModel):
    """A bounded LLM opinion kept separate from deterministic agent output."""

    agent: AgentName
    task: str = Field(min_length=1, max_length=200)
    status: Literal["GROUNDED", "UNAVAILABLE"]
    assessment: str = Field(min_length=1, max_length=1400)
    supporting_observations: list[str] = Field(default_factory=list, max_length=4)
    counterarguments: list[str] = Field(default_factory=list, max_length=4)
    missing_information: list[str] = Field(default_factory=list, max_length=4)
    cited_source_ids: list[str] = Field(default_factory=list, max_length=10)
    model_role: str = "evidence_interpreter"
    controls_numeric_output: bool = False
    claims: list[GroundedClaim] = Field(default_factory=list, max_length=6)
    grounding_status: Literal['CLAIM_ANCHORS_VERIFIED', 'LEGACY_REVIEW_LEVEL_ONLY', 'REJECTED', 'NOT_AUDITED'] = 'NOT_AUDITED'
    grounding_audit: dict = Field(default_factory=dict)
    follow_up_proposals: list[AgentFollowUpProposal] = Field(default_factory=list, max_length=2)

