from __future__ import annotations

from stock_multi_agent.models.chat import ConversationState, ParsedQuery
from stock_multi_agent.nlu.semantic_parser import HybridSemanticParser
from stock_multi_agent.nlu.telemetry import SemanticTelemetryStore
from stock_multi_agent.providers.llm_provider import SemanticLLMProvider


class QueryParser:
    """Backward-compatible facade over the hybrid semantic NLU pipeline."""

    def __init__(
        self,
        llm_provider: SemanticLLMProvider | None = None,
        use_configured_provider: bool = True,
        telemetry_store: SemanticTelemetryStore | None = None,
    ) -> None:
        self.semantic = HybridSemanticParser(llm_provider, use_configured_provider, telemetry_store)

    def parse(self, raw_query: str, context: ConversationState | None = None) -> ParsedQuery:
        return self.semantic.parse(raw_query, context)
