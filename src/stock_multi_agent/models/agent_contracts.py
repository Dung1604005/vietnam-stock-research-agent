from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from stock_multi_agent.models.analysis import FundamentalAnalysis, MarketData, RiskAnalysis, TechnicalAnalysis
from stock_multi_agent.models.data import PortfolioPosition
from stock_multi_agent.models.forecast import PriceForecast
from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.models.research import Citation

if TYPE_CHECKING:
    from stock_multi_agent.models.state import AgentState


class AgentSpec(BaseModel):
    """Auditable contract describing one agent's bounded authority."""

    role: AgentName
    goal: str = Field(min_length=1)
    required_context: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    output_type: str = Field(min_length=1)
    can_use_llm: bool = False
    decision_constraints: list[str] = Field(default_factory=list)


class AgentExecutionReport(BaseModel):
    """Common execution envelope emitted alongside each specialist result."""

    agent: AgentName
    stage: str = Field(min_length=1)
    status: Literal["COMPLETED", "PARTIAL", "UNAVAILABLE"]
    summary: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    output_key: str = Field(min_length=1)
    tools_used: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    evidence_count: int = Field(default=0, ge=0)


class AgentEvaluationSummary(BaseModel):
    """Small, machine-readable quality gate for one pipeline execution."""

    expected_agents: list[AgentName] = Field(default_factory=list)
    executed_agents: list[AgentName] = Field(default_factory=list)
    missing_agents: list[AgentName] = Field(default_factory=list)
    missing_outputs: list[str] = Field(default_factory=list)
    low_confidence_agents: list[AgentName] = Field(default_factory=list)
    contract_coverage: float = Field(ge=0.0, le=1.0)
    output_coverage: float = Field(ge=0.0, le=1.0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool


class DataAgentContext(BaseModel):
    ticker: str
    investment_horizon: str
    required_data: list[str]

    @classmethod
    def from_state(cls, state: AgentState) -> "DataAgentContext":
        return cls(
            ticker=state.ticker,
            investment_horizon=state.investment_horizon,
            required_data=list(state.required_data),
        )


class ResearchAgentContext(BaseModel):
    ticker: str
    investment_horizon: str
    user_goal: str

    @classmethod
    def from_state(cls, state: AgentState) -> "ResearchAgentContext":
        return cls(
            ticker=state.ticker,
            investment_horizon=state.investment_horizon,
            user_goal=state.user_goal,
        )


class TechnicalAgentContext(BaseModel):
    ticker: str
    investment_horizon: str
    requested_indicators: list[str]
    market_data: MarketData
    benchmark_data: MarketData | None = None
    price_forecast: PriceForecast | None = None

    @classmethod
    def from_state(cls, state: AgentState) -> "TechnicalAgentContext":
        if state.market_data is None:
            raise ValueError("Technical Agent requires market data")
        return cls(
            ticker=state.ticker,
            investment_horizon=state.investment_horizon,
            requested_indicators=list(state.requested_indicators),
            market_data=state.market_data,
            benchmark_data=state.benchmark_data,
            price_forecast=state.price_forecast,
        )


class FundamentalAgentContext(BaseModel):
    ticker: str
    investment_horizon: str
    market_data: MarketData
    research_evidence: list[Citation] = Field(default_factory=list)

    @classmethod
    def from_state(cls, state: AgentState) -> "FundamentalAgentContext":
        if state.market_data is None:
            raise ValueError("Fundamental Agent requires market data")
        return cls(
            ticker=state.ticker,
            investment_horizon=state.investment_horizon,
            market_data=state.market_data,
            research_evidence=list(state.research_evidence),
        )


class RiskAgentContext(BaseModel):
    ticker: str
    investment_horizon: str
    market_data: MarketData
    technical_result: TechnicalAnalysis
    fundamental_result: FundamentalAnalysis
    price_forecast: PriceForecast | None = None
    portfolio_positions: list[PortfolioPosition] = Field(default_factory=list)
    research_evidence: list[Citation] = Field(default_factory=list)

    @classmethod
    def from_state(cls, state: AgentState) -> "RiskAgentContext":
        if state.market_data is None or state.technical_result is None or state.fundamental_result is None:
            raise ValueError("Risk Agent requires data and both specialist results")
        return cls(
            ticker=state.ticker,
            investment_horizon=state.investment_horizon,
            market_data=state.market_data,
            technical_result=state.technical_result,
            fundamental_result=state.fundamental_result,
            price_forecast=state.price_forecast,
            portfolio_positions=list(state.portfolio_positions),
            research_evidence=list(state.research_evidence),
        )


class ManagerContext(BaseModel):
    ticker: str
    investment_horizon: str
    user_goal: str
    technical_result: TechnicalAnalysis | None = None
    fundamental_result: FundamentalAnalysis | None = None
    risk_result: RiskAnalysis | None = None
    research_evidence_count: int = 0

    @classmethod
    def from_state(cls, state: AgentState) -> "ManagerContext":
        return cls(
            ticker=state.ticker,
            investment_horizon=state.investment_horizon,
            user_goal=state.user_goal,
            technical_result=state.technical_result,
            fundamental_result=state.fundamental_result,
            risk_result=state.risk_result,
            research_evidence_count=len(state.research_evidence),
        )
