from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from stock_multi_agent.models.chat import ParsedQuery, QueryIntent


class EntityExtraction(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    sector: str | None = None
    duration: str | None = None
    temporal_direction: str | None = None
    indicators: list[str] = Field(default_factory=list)
    requested_information: list[str] = Field(default_factory=list)
    requested_result_count: int | None = Field(default=None, ge=1, le=100)
    requested_actions: list[str] = Field(default_factory=list)


class ContextResolution(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    references_previous_context: bool = False
    resolved_references: dict[str, Any] = Field(default_factory=dict)
    ambiguities: list[str] = Field(default_factory=list)


class SemanticQuery(BaseModel):
    original_query: str
    normalized_query: str
    intent: QueryIntent
    secondary_intents: list[QueryIntent] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    sector: str | None = None
    market: str = "VN"
    time_horizon: str | None = None
    chart_period: str | None = None
    indicators: list[str] = Field(default_factory=list)
    requested_result_count: int | None = Field(default=None, ge=1, le=100)
    requested_actions: list[str] = Field(default_factory=list)
    comparison_targets: list[str] = Field(default_factory=list)
    requested_information: list[str] = Field(default_factory=list)
    references_previous_context: bool = False
    resolved_references: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguities: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None
    interpretation: str = ""
    parser_source: str = "fallback"
    intent_scores: dict[str, float] = Field(default_factory=dict)

    def to_parsed_query(self) -> ParsedQuery:
        return ParsedQuery(
            raw_query=self.original_query,
            normalized_query=self.normalized_query,
            intent=self.intent,
            secondary_intents=self.secondary_intents,
            tickers=self.tickers,
            market=self.market,
            sector=self.sector,
            investment_horizon=self.time_horizon or "3 months",
            chart_period=self.chart_period,
            requested_indicators=self.indicators,
            requested_result_count=self.requested_result_count,
            requested_actions=self.requested_actions,
            comparison_targets=self.comparison_targets,
            requested_information=self.requested_information,
            follow_up=self.references_previous_context,
            references_previous_context=self.references_previous_context,
            resolved_references=self.resolved_references,
            confidence=self.confidence,
            ambiguities=self.ambiguities,
            needs_clarification=self.needs_clarification,
            clarification_question=self.clarification_question,
            interpretation=self.interpretation,
            parser_source=self.parser_source,
            intent_scores=self.intent_scores,
            needs_current_data=self.intent not in {
                QueryIntent.GENERAL_CHAT,
                QueryIntent.HELP,
                QueryIntent.EXPLANATION,
                QueryIntent.THEORY,
                QueryIntent.CURRENT_RULE,
                QueryIntent.UNKNOWN,
            },
        )
