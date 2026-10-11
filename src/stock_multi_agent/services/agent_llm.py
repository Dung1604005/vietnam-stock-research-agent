from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from stock_multi_agent.models.llm import AgentLLMReview
from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.providers.llm_provider import SemanticLLMProvider, configured_semantic_provider


class _StructuredReview(BaseModel):
    assessment: str = Field(min_length=1, max_length=1000)
    supporting_observations: list[str] = Field(default_factory=list, max_length=4)
    counterarguments: list[str] = Field(default_factory=list, max_length=4)
    missing_information: list[str] = Field(default_factory=list, max_length=4)
    cited_source_ids: list[str] = Field(default_factory=list, max_length=10)


class AgentLLMAdvisor:
    """Shared structured-reasoning capability for specialist agents.

    The advisor may interpret, challenge, and identify missing information. It
    never mutates measured values, scores, signals, vetoes, or recommendations.
    """

    SYSTEM_PROMPT = """You are one specialist inside a Vietnamese equity-research multi-agent system.
Use only the supplied structured facts. Do not invent prices, dates, ratios,
news, company facts, model accuracy, or sources. Distinguish observations from
uncertainty. Challenge weak assumptions. Cite only IDs in allowed_source_ids.
Never calculate or change a numeric score, signal, risk veto, forecast, or final
recommendation. Reply in concise Vietnamese and strictly match the JSON schema."""

    def __init__(self, provider: SemanticLLMProvider | None = None, enabled: bool | None = None) -> None:
        self.provider = provider if provider is not None else configured_semantic_provider()
        configured = os.getenv("STOCK_AGENT_LLM_SPECIALISTS_ENABLED", "false").strip().lower()
        self.enabled = enabled if enabled is not None else configured in {"1", "true", "yes", "on"}
        roles = os.getenv(
            "STOCK_AGENT_LLM_SPECIALIST_ROLES",
            "data,technical,fundamental,risk,manager",
        )
        self.roles = {item.strip().lower() for item in roles.split(",") if item.strip()}

    def review(
        self,
        agent: AgentName,
        task: str,
        facts: dict[str, Any],
        allowed_source_ids: list[str],
    ) -> AgentLLMReview | None:
        if not self.enabled or self.provider is None or agent.value not in self.roles:
            return None
        payload = {
            "specialist_role": agent.value,
            "task": task,
            "facts": facts,
            "allowed_source_ids": list(dict.fromkeys(allowed_source_ids)),
        }
        try:
            raw = self.provider.parse_structured(self.SYSTEM_PROMPT, payload, _StructuredReview)
        except Exception as exc:
            return AgentLLMReview(
                agent=agent,
                task=task,
                status="UNAVAILABLE",
                assessment=f"LLM specialist unavailable ({type(exc).__name__}); deterministic result preserved.",
            )
        allowed = set(allowed_source_ids)
        return AgentLLMReview(
            agent=agent,
            task=task,
            status="GROUNDED",
            assessment=raw.assessment,
            supporting_observations=raw.supporting_observations,
            counterarguments=raw.counterarguments,
            missing_information=raw.missing_information,
            cited_source_ids=[item for item in raw.cited_source_ids if item in allowed],
        )

