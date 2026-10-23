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
from stock_multi_agent.services.request_telemetry import current_metrics


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
        research_web = None
        if isinstance(market_provider, VnstockMarketDataProvider):
            from stock_multi_agent.services.research_orchestrator import ResearchOrchestrator
            from stock_multi_agent.services.research_planner import ResearchPlanner
            planner = ResearchPlanner()
            planner.max_searches, planner.max_pages, planner.timeout_seconds = 3, 4, 20
            research_web = ResearchOrchestrator(planner=planner)
        self.research_agent = ResearchAgent(research_store, web_research=research_web, llm_advisor=advisor)
        self.technical_agent = TechnicalAgent(technical_tools, llm_advisor=advisor)
        self.fundamental_agent = FundamentalAgent(FinancialToolkit(), llm_advisor=advisor)
        self.risk_agent = RiskAgent(technical_tools, llm_advisor=advisor)
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
        allow_llm_planning: bool = True,
        force_refresh: bool = False,
    ) -> AgentState:
        symbol = self._validate_ticker(ticker)
        if not investment_horizon.strip():
            raise ValueError("Investment horizon is required")
        initial = self._initial_state(
            symbol, investment_horizon, max_iterations=max_iterations,
            required_data=["ohlcv", "financials", "benchmark"], portfolio_positions=portfolio_positions,
            allow_llm_planning=allow_llm_planning,
            force_data_refresh=force_refresh,
        )
        result = self.graph.invoke(initial)
        return result

    def resume_analysis(self, run_id: str) -> AgentState:
        return self.graph.resume(run_id)

    def create_state(self, ticker: str, investment_horizon: str = '3 months', **kwargs) -> AgentState:
        return self._initial_state(self._validate_ticker(ticker), investment_horizon, **kwargs)

    @staticmethod
    def merge_update(state: AgentState, update: dict[str, object]) -> AgentState:
        return AnalysisService._merge(state, update)

    def run_stage(self, state: AgentState, stage: str, **kwargs) -> AgentState:
        """Public typed stage API shared by graph-backed and selective routes."""
        from stock_multi_agent.services.agent_execution import execute_agent, execution_for_state
        with execution_for_state(state):
            if stage == 'forecast':
                update = execute_agent(state, self.data_agent, stage, self.graph._forecast,
                                       allowed_outputs={'price_forecast'})
            else:
                agent = {'data': self.data_agent, 'research': self.research_agent,
                         'technical': self.technical_agent, 'fundamental': self.fundamental_agent,
                         'risk': self.risk_agent}.get(stage)
                if agent is None:
                    raise ValueError('Unsupported analysis stage')
                update = execute_agent(state, agent, stage, lambda value: agent.run(value, **kwargs),
                                       essential=stage == 'data' and state.market_data is None)
        return self._merge(state, update)

    def fetch_data(
        self,
        ticker: str,
        investment_horizon: str = "3 months",
        include_financials: bool = False,
        include_benchmark: bool = False,
        include_forecast: bool = True,
        force_refresh: bool = False,
    ) -> AgentState:
        required_data = ["ohlcv", "financials"] if include_financials else ["ohlcv"]
        if include_benchmark:
            required_data.append("benchmark")
        state = self._initial_state(
            self._validate_ticker(ticker),
            investment_horizon,
            required_data=required_data,
        )
        result = self.run_stage(state, 'data', refresh=force_refresh)
        if result.market_data is not None and include_forecast:
            result = self.run_stage(result, 'forecast')
        return result

    def analyze_technical(
        self,
        ticker: str,
        investment_horizon: str = "3 months",
        requested_indicators: list[str] | None = None,
        include_benchmark: bool = False,
        allow_llm_planning: bool = True,
        include_forecast: bool = True,
        force_refresh: bool = False,
    ) -> AgentState:
        state = self.fetch_data(
            ticker, investment_horizon, include_financials=False, include_benchmark=include_benchmark,
            include_forecast=include_forecast, force_refresh=force_refresh,
        )
        state.requested_indicators = requested_indicators or []
        return self.run_stage(state, 'technical', allow_llm_planning=allow_llm_planning)

    def analyze_fundamental(self, ticker: str, investment_horizon: str = "3 months") -> AgentState:
        state = self.fetch_data(ticker, investment_horizon, include_financials=True)
        state = self.run_stage(state, 'research')
        return self.run_stage(state, 'fundamental')

    def analyze_risk(self, ticker: str, investment_horizon: str = "3 months") -> AgentState:
        state = self.fetch_data(ticker, investment_horizon, include_financials=True, include_benchmark=True)
        state = self.run_stage(state, 'research')
        state = self.run_stage(state, 'technical')
        state = self.run_stage(state, 'fundamental')
        return self.run_stage(state, 'risk')

    def _initial_state(
        self,
        symbol: str,
        investment_horizon: str,
        max_iterations: int = 2,
        required_data: list[str] | None = None,
        portfolio_positions: list[PortfolioPosition] | None = None,
        allow_llm_planning: bool = True,
        force_data_refresh: bool = False,
    ) -> AgentState:
        from stock_multi_agent.models.research_contracts import RequestBudget
        from stock_multi_agent.services.execution_profiles import constrain_budget, current_profile
        if not investment_horizon.strip():
            raise ValueError('Investment horizon is required')
        profile = current_profile()
        if profile.mode.value in {'QUICK', 'OFFLINE'}:
            allow_llm_planning = False
            max_iterations = 0
        return AgentState(
            ticker=symbol,
            investment_horizon=investment_horizon.strip(),
            user_goal=f"Analyze {symbol} for an investment horizon of {investment_horizon.strip()}",
            max_iterations=max_iterations,
            required_data=required_data or ["ohlcv", "financials"],
            portfolio_positions=portfolio_positions or [],
            allow_llm_planning=allow_llm_planning,
            force_data_refresh=force_data_refresh,
            execution_mode=profile.mode.value,
            execution_profile=profile,
            request_budget=constrain_budget(
                current_metrics().budget.model_copy(deep=True) if current_metrics() else RequestBudget(), profile,
            ),
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
            if key in {"messages", "trace", "completed_tasks", "llm_reviews", "agent_reports", "task_executions",
                       "reanalysis_history", "evidence_follow_up_outcomes"}:
                values[key] = [*values.get(key, []), *value]  # type: ignore[arg-type]
            else:
                values[key] = value
        return AgentState.model_validate(values)
