from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from stock_multi_agent.models.chart import ChartArtifact
from stock_multi_agent.models.data import SourceReference
from stock_multi_agent.models.state import TraceEvent


class QueryIntent(str, Enum):
    GENERAL_CHAT = "GENERAL_CHAT"
    STOCK_ANALYSIS = "STOCK_ANALYSIS"
    TECHNICAL_ANALYSIS = "TECHNICAL_ANALYSIS"
    FUNDAMENTAL_ANALYSIS = "FUNDAMENTAL_ANALYSIS"
    RISK_ANALYSIS = "RISK_ANALYSIS"
    MARKET_OVERVIEW = "MARKET_OVERVIEW"
    STOCK_DISCOVERY = "STOCK_DISCOVERY"
    STOCK_RECOMMENDATION = "STOCK_RECOMMENDATION"
    STOCK_COMPARISON = "STOCK_COMPARISON"
    STOCK_CHART = "STOCK_CHART"
    INDICATOR_QUERY = "INDICATOR_QUERY"
    RESEARCH = "RESEARCH"
    THEORY = "THEORY"
    CURRENT_RULE = "CURRENT_RULE"
    MIXED = "MIXED"
    FOLLOW_UP = "FOLLOW_UP"
    EXPLANATION = "EXPLANATION"
    HELP = "HELP"
    UNKNOWN = "UNKNOWN"


class ResponseType(str, Enum):
    TEXT = "TEXT"
    ANALYSIS = "ANALYSIS"
    MARKET_OVERVIEW = "MARKET_OVERVIEW"
    STOCK_LIST = "STOCK_LIST"
    COMPARISON = "COMPARISON"
    CHART = "CHART"


class ParsedQuery(BaseModel):
    raw_query: str
    normalized_query: str = ""
    intent: QueryIntent
    secondary_intents: list[QueryIntent] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    market: str = "VN"
    sector: str | None = None
    investment_horizon: str = "3 months"
    chart_period: str | None = None
    requested_indicators: list[str] = Field(default_factory=list)
    requested_result_count: int | None = Field(default=None, ge=1, le=100)
    requested_actions: list[str] = Field(default_factory=list)
    comparison_targets: list[str] = Field(default_factory=list)
    requested_information: list[str] = Field(default_factory=list)
    risk_preference: str | None = None
    needs_current_data: bool = True
    follow_up: bool = False
    references_previous_context: bool = False
    resolved_references: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguities: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None
    interpretation: str = ""
    parser_source: str = "fallback"
    intent_scores: dict[str, float] = Field(default_factory=dict)
    parser_latency_ms: int = Field(default=0, ge=0)
    llm_attempted: bool = False
    llm_latency_ms: int | None = Field(default=None, ge=0)
    fallback_reason: str | None = None
    prompt_version: str = "semantic-system-v1"
    example_bank_version: str = "intent-examples-v1"


class ResponseTable(BaseModel):
    title: str
    columns: list[str]
    rows: list[dict[str, Any]]
    note: str | None = None


class AssistantResponse(BaseModel):
    message: str
    response_type: ResponseType = ResponseType.TEXT
    tickers: list[str] = Field(default_factory=list)
    charts: list[ChartArtifact] = Field(default_factory=list)
    tables: list[ResponseTable] = Field(default_factory=list)
    agent_trace: list[TraceEvent] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    source_details: list[SourceReference] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    data_timestamp: date | None = None
    data_fetched_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationMessage(BaseModel):
    role: ChatRole
    content: str
    response: AssistantResponse | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = "New conversation"
    messages: list[ConversationMessage] = Field(default_factory=list)
    active_tickers: list[str] = Field(default_factory=list)
    last_intent: QueryIntent | None = None
    investment_horizon: str = "3 months"
    last_analysis: dict[str, Any] = Field(default_factory=dict)
    last_chart_period: str | None = None
    last_stock_list: list[str] = Field(default_factory=list)
    entity_history: list[list[str]] = Field(default_factory=list)
    last_semantic_query: dict[str, Any] = Field(default_factory=dict)
    user_preferences: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    max_messages: int = Field(default=24, ge=4, le=100)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def add_message(self, message: ConversationMessage) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            removed = self.messages[: len(self.messages) - self.max_messages]
            self.messages = self.messages[-self.max_messages :]
            snippets = [f"{item.role.value}: {item.content[:120]}" for item in removed]
            self.summary = (self.summary + " | " + " | ".join(snippets))[-1200:].strip(" |")
        self.updated_at = datetime.now(timezone.utc)

    def recent_messages(self, limit: int = 8) -> list[ConversationMessage]:
        return self.messages[-limit:]
