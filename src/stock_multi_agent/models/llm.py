from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from stock_multi_agent.models.messages import AgentName


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

