from __future__ import annotations

import os
import re

from stock_multi_agent.agents import DataAgent, FundamentalAgent, ManagerAgent, ResearchAgent, RiskAgent, TechnicalAgent
from stock_multi_agent.graph.stock_agent_graph import StockAgentGraph
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.models.data import PortfolioPosition
from stock_multi_agent.providers.market_provider import MarketDataProvider, VnstockMarketDataProvider
from stock_multi_agent.services.price_forecast import PriceForecastService
from stock_multi_agent.services.agent_llm import AgentLLMAdvisor
from stock_multi_agent.services.research_store import ResearchStore
from stock_multi_agent.tools import FinancialToolkit, MarketDataTool, TechnicalIndicatorToolkit


class AnalysisService:
    """Public use-case API. Real vnstock data is the default and only UI provider."""

    def __init__(
        self,
        provider: MarketDataProvider | None = None,
        price_forecast_service: PriceForecastService | None = None,
        llm_advisor: AgentLLMAdvisor | None = None,
        research_store: ResearchStore | None = None,
        manager_routing_enabled: bool | None = None,
    ) -> None:
        source = os.getenv("VNSTOCK_SOURCE", "VCI")
        market_provider = provider or VnstockMarketDataProvider(source=source)
        technical_tools = TechnicalIndicatorToolkit()
        advisor = llm_advisor or AgentLLMAdvisor()
        self.data_agent = DataAgent(MarketDataTool(market_provider))
        self.research_agent = ResearchAgent(research_store)
        self.technical_agent = TechnicalAgent(technical_tools, llm_advisor=advisor)
        self.fundamental_agent = FundamentalAgent(FinancialToolkit())
        self.risk_agent = RiskAgent(technical_tools)
        self.manager_agent = ManagerAgent(
            llm_advisor=advisor,
            llm_provider=advisor.provider,
            routing_enabled=manager_routing_enabled,
        )
        self.price_forecast_service = price_forecast_service or PriceForecastService()
        self.graph = StockAgentGraph(
            data_agent=self.data_agent,
            research_agent=self.research_agent,
            technical_agent=self.technical_agent,
            fundamental_agent=self.fundamental_agent,
            risk_agent=self.risk_agent,
            manager_agent=self.manager_agent,
            price_forecast_service=self.price_forecast_service,
        )

    def analyze(
        self,
        ticker: str,
        investment_horizon: str = "3 months",
        max_iterations: int = 2,
        portfolio_positions: list[PortfolioPosition] | None = None,
    ) -> AgentState:
        symbol = self._validate_ticker(ticker)
        if not investment_horizon.strip():
            raise ValueError("Investment horizon is required")
        initial = self._initial_state(
            symbol, investment_horizon, max_iterations=max_iterations,
            required_data=["ohlcv", "financials", "benchmark"], portfolio_positions=portfolio_positions,
        )
        result = self.graph.invoke(initial)
        if result.market_data is not None and result.price_forecast is None:
            result.price_forecast = self.price_forecast_service.forecast_next_session(result.market_data)
        return result

    def fetch_data(
        self,
        ticker: str,
        investment_horizon: str = "3 months",
        include_financials: bool = False,
        include_benchmark: bool = False,
        include_forecast: bool = True,
    ) -> AgentState:
        required_data = ["ohlcv", "financials"] if include_financials else ["ohlcv"]
        if include_benchmark:
            required_data.append("benchmark")
        state = self._initial_state(
            self._validate_ticker(ticker),
            investment_horizon,
            required_data=required_data,
        )
        result = self._merge(state, self.data_agent.run(state))
        if result.market_data is not None and include_forecast:
            result.price_forecast = self.price_forecast_service.forecast_next_session(result.market_data)
        return result

    def analyze_technical(
        self,
        ticker: str,
        investment_horizon: str = "3 months",
        requested_indicators: list[str] | None = None,
        include_benchmark: bool = False,
        allow_llm_planning: bool = True,
        include_forecast: bool = True,
    ) -> AgentState:
        state = self.fetch_data(
            ticker, investment_horizon, include_financials=False, include_benchmark=include_benchmark,
            include_forecast=include_forecast,
        )
        state.requested_indicators = requested_indicators or []
        return self._merge(
            state,
            self.technical_agent.run(state, allow_llm_planning=allow_llm_planning),
        )

    def analyze_fundamental(self, ticker: str, investment_horizon: str = "3 months") -> AgentState:
        state = self.fetch_data(ticker, investment_horizon, include_financials=True)
        state = self._merge(state, self.research_agent.run(state))
        return self._merge(state, self.fundamental_agent.run(state))

    def analyze_risk(self, ticker: str, investment_horizon: str = "3 months") -> AgentState:
        state = self.fetch_data(ticker, investment_horizon, include_financials=True, include_benchmark=True)
        state = self._merge(state, self.research_agent.run(state))
        state = self._merge(state, self.technical_agent.run(state))
        state = self._merge(state, self.fundamental_agent.run(state))
        return self._merge(state, self.risk_agent.run(state))

    def _initial_state(
        self,
        symbol: str,
        investment_horizon: str,
        max_iterations: int = 2,
        required_data: list[str] | None = None,
        portfolio_positions: list[PortfolioPosition] | None = None,
    ) -> AgentState:
        return AgentState(
            ticker=symbol,
            investment_horizon=investment_horizon.strip(),
            user_goal=f"Analyze {symbol} for an investment horizon of {investment_horizon.strip()}",
            max_iterations=max_iterations,
            required_data=required_data or ["ohlcv", "financials"],
            portfolio_positions=portfolio_positions or [],
        )

    @staticmethod
    def _validate_ticker(ticker: str) -> str:
        symbol = ticker.strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9]{1,9}", symbol):
            raise ValueError("Ticker must contain 2–10 uppercase letters/digits, e.g. FPT, VNM, HPG.")
        return symbol

    @staticmethod
    def _merge(state: AgentState, update: dict[str, object]) -> AgentState:
        values = state.model_dump()
        for key, value in update.items():
            if key in {"messages", "trace", "completed_tasks", "llm_reviews", "agent_reports"}:
                values[key] = [*values.get(key, []), *value]  # type: ignore[arg-type]
            else:
                values[key] = value
        return AgentState.model_validate(values)
