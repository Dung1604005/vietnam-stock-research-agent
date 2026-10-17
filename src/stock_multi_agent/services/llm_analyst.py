from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from stock_multi_agent.models.chat import AssistantResponse, ParsedQuery, QueryIntent
from stock_multi_agent.providers.llm_provider import SemanticLLMProvider, configured_semantic_provider


class AnalystNarrative(BaseModel):
    """LLM output that is deliberately separate from deterministic decisions."""

    summary: str = Field(min_length=1, max_length=900)
    bull_case: list[str] = Field(default_factory=list, max_length=3)
    bear_case: list[str] = Field(default_factory=list, max_length=3)
    data_limitations: list[str] = Field(default_factory=list, max_length=3)
    cited_source_ids: list[str] = Field(default_factory=list, max_length=8)


class LLMAnalystService:
    """Evidence-bounded LLM commentary, not an autonomous trading decision maker."""

    SYSTEM_PROMPT = """You are a cautious Vietnamese equity-research analyst.
Use ONLY the structured facts supplied. Do not invent prices, dates, ratios,
news, sources, or model performance. Do not override the deterministic
recommendation or numerical forecast. Give a concise Vietnamese scenario-based
commentary, distinguish upside from downside, and cite only source IDs present
in allowed_source_ids. This is educational research, never personalized trading
advice. Return JSON strictly matching the requested schema."""

    ANALYSIS_INTENTS = {
        QueryIntent.STOCK_ANALYSIS,
        QueryIntent.TECHNICAL_ANALYSIS,
        QueryIntent.FUNDAMENTAL_ANALYSIS,
        QueryIntent.RISK_ANALYSIS,
    }

    def __init__(self, provider: SemanticLLMProvider | None = None, enabled: bool | None = None) -> None:
        self.provider = provider if provider is not None else configured_semantic_provider()
        configured = os.getenv("STOCK_AGENT_LLM_ANALYST_ENABLED", "false").strip().lower()
        self.enabled = enabled if enabled is not None else configured in {"1", "true", "yes", "on"}

    def enrich(self, response: AssistantResponse, query: ParsedQuery) -> None:
        """Append validated commentary while preserving deterministic output on failure."""
        if not self.enabled or self.provider is None or query.intent not in self.ANALYSIS_INTENTS:
            return

        source_ids = [item.source_id for item in response.source_details]
        facts: dict[str, Any] = {
            "ticker": response.tickers[0] if response.tickers else None,
            "intent": query.intent.value,
            "requested_horizon": query.investment_horizon,
            "deterministic_analysis": {
                key: value
                for key, value in response.metadata.items()
                if key in {
                    "recommendation", "technical_score", "technical_signal",
                    "fundamental_score", "fundamental_signal", "risk_score", "risk_level",
                    "veto", "price_forecast", "conflicts",
                }
            },
            "warnings": response.warnings[:8],
            "allowed_source_ids": source_ids,
        }
        try:
            narrative = self.provider.parse_structured(self.SYSTEM_PROMPT, facts, AnalystNarrative)
        except Exception as exc:
            response.metadata["llm_analyst"] = {"status": "unavailable", "reason": type(exc).__name__}
            return

        allowed = set(source_ids)
        cited = [source_id for source_id in narrative.cited_source_ids if source_id in allowed]
        narrative = narrative.model_copy(update={"cited_source_ids": cited})
        response.message += self._render(narrative)
        response.metadata["llm_analyst"] = {
            "status": "grounded_commentary",
            **narrative.model_dump(mode="json"),
            "does_not_override_deterministic_decision": True,
        }

    @staticmethod
    def _render(narrative: AnalystNarrative) -> str:
        parts = ["\n\n---\n\n### Nhận định bổ sung từ Local LLM", narrative.summary]
        if narrative.bull_case:
            parts.append("\n**Kịch bản tích cực:**\n" + "\n".join(f"- {item}" for item in narrative.bull_case))
        if narrative.bear_case:
            parts.append("\n**Kịch bản thận trọng:**\n" + "\n".join(f"- {item}" for item in narrative.bear_case))
        if narrative.data_limitations:
            parts.append("\n**Giới hạn dữ liệu:**\n" + "\n".join(f"- {item}" for item in narrative.data_limitations))
        if narrative.cited_source_ids:
            parts.append("\n**Nguồn được LLM tham chiếu:** " + ", ".join(f"`{item}`" for item in narrative.cited_source_ids))
        parts.append("\nNhận định này không thay thế tín hiệu, dự báo số hoặc chính sách BUY/HOLD/WATCH deterministic.")
        return "\n".join(parts)
