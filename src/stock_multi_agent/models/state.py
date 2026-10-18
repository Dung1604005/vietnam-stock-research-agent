from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from stock_multi_agent.models.analysis import (
    FinalDecision,
    FundamentalAnalysis,
    MarketData,
    RiskAnalysis,
    TechnicalAnalysis,
)
from stock_multi_agent.models.agent_contracts import AgentExecutionReport
from stock_multi_agent.models.conflicts import ConflictRecord, ReanalysisAction
from stock_multi_agent.models.data import PortfolioPosition
from stock_multi_agent.models.forecast import PriceForecast
from stock_multi_agent.models.messages import AgentMessage, AgentName
from stock_multi_agent.models.llm import AgentLLMReview
from stock_multi_agent.models.research import Citation
from stock_multi_agent.models.research_contracts import RequestBudget


class TraceEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: AgentName
    phase: str
    action: str
    detail: str
    tool: str | None = None


class AgentState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    ticker: str
    request_budget: RequestBudget = Field(default_factory=RequestBudget)
    investment_horizon: str
    user_goal: str
    required_data: list[str] = Field(default_factory=lambda: ["ohlcv", "financials"])
    requested_indicators: list[str] = Field(default_factory=list)
    market_data: MarketData | None = None
    benchmark_data: MarketData | None = None
    price_forecast: PriceForecast | None = None
    research_evidence: list[Citation] = Field(default_factory=list)
    research_freshness_note: str = ""
    financial_fact_proposals: list[dict] = Field(default_factory=list)
    evidence_follow_up_outcomes: Annotated[list[dict], operator.add] = Field(default_factory=list)
    portfolio_positions: list[PortfolioPosition] = Field(default_factory=list)
    technical_result: TechnicalAnalysis | None = None
    fundamental_result: FundamentalAnalysis | None = None
    risk_result: RiskAnalysis | None = None
    messages: Annotated[list[AgentMessage], operator.add] = Field(default_factory=list)
    llm_reviews: Annotated[list[AgentLLMReview], operator.add] = Field(default_factory=list)
    agent_reports: Annotated[list[AgentExecutionReport], operator.add] = Field(default_factory=list)
    trace: Annotated[list[TraceEvent], operator.add] = Field(default_factory=list)
    current_plan: list[str] = Field(default_factory=list)
    completed_tasks: Annotated[list[str], operator.add] = Field(default_factory=list)
    iteration: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=2, ge=0, le=10)
    allow_llm_planning: bool = True
    force_data_refresh: bool = False
    conflict_detected: bool = False
    reanalysis_requested: bool = False
    active_conflicts: list[ConflictRecord] = Field(default_factory=list)
    reanalysis_action: ReanalysisAction = ReanalysisAction.NONE
    reanalysis_history: Annotated[list[ReanalysisAction], operator.add] = Field(default_factory=list)
    reanalysis_evidence_fingerprint: str | None = None
    final_decision: FinalDecision | None = None
