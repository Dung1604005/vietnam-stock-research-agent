from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from stock_multi_agent.agents import DataAgent, FundamentalAgent, ManagerAgent, ResearchAgent, RiskAgent, TechnicalAgent
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.price_forecast import PriceForecastService


class StockAgentGraph:
    """Explicit supervisor graph with parallel specialists and a bounded cycle."""

    def __init__(
        self,
        data_agent: DataAgent,
        research_agent: ResearchAgent,
        technical_agent: TechnicalAgent,
        fundamental_agent: FundamentalAgent,
        risk_agent: RiskAgent,
        manager_agent: ManagerAgent,
        price_forecast_service: PriceForecastService,
    ) -> None:
        self.data_agent = data_agent
        self.research_agent = research_agent
        self.technical_agent = technical_agent
        self.fundamental_agent = fundamental_agent
        self.risk_agent = risk_agent
        self.manager_agent = manager_agent
        self.price_forecast_service = price_forecast_service
        self.compiled = self._build().compile()

    def _build(self) -> StateGraph:
        graph = StateGraph(AgentState)
        from stock_multi_agent.services.request_telemetry import observed
        register_node = graph.add_node
        def add_observed_node(name, action, **kwargs):
            return register_node(name, observed(f'graph.{name}')(action), **kwargs)
        graph.add_node = add_observed_node
        graph.add_node("manager_plan", self.manager_agent.plan)
        graph.add_node("data", lambda state: self.data_agent.run(state, refresh=state.force_data_refresh))
        graph.add_node("forecast", self._forecast)
        graph.add_node("research", self.research_agent.run)
        graph.add_node("dispatch_specialists", self.manager_agent.dispatch_specialists)
        graph.add_node("technical", lambda state: self.technical_agent.run(
            state,
            extended=False,
            allow_llm_planning=state.allow_llm_planning,
        ))
        graph.add_node("fundamental", lambda state: self.fundamental_agent.run(state, allow_llm_review=False))
        graph.add_node("dispatch_risk", self.manager_agent.dispatch_risk)
        graph.add_node("risk", lambda state: self.risk_agent.run(state, allow_llm_review=False))
        graph.add_node("manager_review", self.manager_agent.review)
        graph.add_node("technical_reanalysis", lambda state: self.technical_agent.run(state, extended=True, allow_llm_planning=state.allow_llm_planning))
        graph.add_node("risk_after_technical", lambda state: self.risk_agent.run(state, allow_llm_review=False))
        graph.add_node("research_reanalysis", lambda state: self.research_agent.run(state, expanded=True))
        graph.add_node("fundamental_after_research", lambda state: self.fundamental_agent.run(state, allow_llm_review=False))
        graph.add_node("risk_after_research", lambda state: self.risk_agent.run(state, allow_llm_review=False))
        graph.add_node("data_reanalysis", lambda state: self.data_agent.run(state, refresh=True))
        graph.add_node("forecast_reanalysis", self._forecast)
        graph.add_node("research_after_data", lambda state: self.research_agent.run(state, expanded=True))
        graph.add_node("technical_after_data", lambda state: self.technical_agent.run(state, extended=True, allow_llm_planning=state.allow_llm_planning))
        graph.add_node("fundamental_after_data", lambda state: self.fundamental_agent.run(state, allow_llm_review=False))
        graph.add_node("risk_after_data", lambda state: self.risk_agent.run(state, allow_llm_review=False))
        graph.add_node('specialist_reviews', self._specialist_reviews)
        graph.add_node('evidence_follow_up', self._evidence_follow_up)
        graph.add_node("finalize", self.manager_agent.finalize)

        graph.add_edge(START, "manager_plan")
        graph.add_edge("manager_plan", "data")
        graph.add_edge("data", "forecast")
        graph.add_edge("forecast", "research")
        graph.add_edge("research", "dispatch_specialists")
        graph.add_edge("dispatch_specialists", "technical")
        graph.add_edge("dispatch_specialists", "fundamental")
        graph.add_edge(["technical", "fundamental"], "dispatch_risk")
        graph.add_edge("dispatch_risk", "risk")
        graph.add_edge("risk", "manager_review")
        graph.add_conditional_edges(
            "manager_review",
            self.manager_agent.route_after_review,
            {
                "technical": "technical_reanalysis",
                "research_fundamental": "research_reanalysis",
                "data_pipeline": "data_reanalysis",
                "finalize": "specialist_reviews",
            },
        )
        graph.add_edge("technical_reanalysis", "risk_after_technical")
        graph.add_edge("risk_after_technical", "manager_review")
        graph.add_edge("research_reanalysis", "fundamental_after_research")
        graph.add_edge("fundamental_after_research", "risk_after_research")
        graph.add_edge("risk_after_research", "manager_review")
        graph.add_edge("data_reanalysis", "forecast_reanalysis")
        graph.add_edge("forecast_reanalysis", "research_after_data")
        graph.add_edge("research_after_data", "technical_after_data")
        graph.add_edge("research_after_data", "fundamental_after_data")
        graph.add_edge(["technical_after_data", "fundamental_after_data"], "risk_after_data")
        graph.add_edge("risk_after_data", "manager_review")
        graph.add_edge('specialist_reviews', 'evidence_follow_up')
        graph.add_edge('evidence_follow_up', 'finalize')
        graph.add_edge("finalize", END)
        return graph

    def invoke(self, state: AgentState) -> AgentState:
        from stock_multi_agent.services.request_telemetry import current_metrics, request_scope
        if current_metrics() is None:
            with request_scope(state.request_budget):
                return self.invoke(state)
        output = self.compiled.invoke(state, config={'recursion_limit': max(40, 15 + state.max_iterations * 10)})
        return output if isinstance(output, AgentState) else AgentState.model_validate(output)

    def _specialist_reviews(self, state):
        if not state.allow_llm_planning:
            return {}
        reviews, trace = [], []
        for agent, result in ((self.fundamental_agent, state.fundamental_result), (self.risk_agent, state.risk_result)):
            if result is not None:
                review = agent.review_result(state, result)
                if review:
                    reviews.append(review)
                    trace.append(agent.event('REASON', 'LLM_FINAL_EVIDENCE_REVIEW', f'LLM={review.status}; final post-conflict evidence pack', 'configured_llm'))
        return {'llm_reviews': reviews, 'trace': trace}

    def _evidence_follow_up(self, state):
        from stock_multi_agent.services.evidence_follow_up import run_evidence_follow_up
        return run_evidence_follow_up(state, self)

    def _forecast(self, state: AgentState) -> dict[str, object]:
        if state.market_data is None:
            return {}
        forecast = self.price_forecast_service.forecast_next_session(state.market_data)
        return {
            "price_forecast": forecast,
            "trace": [self.data_agent.event(
                "ACT", "CALL_FORECAST_TOOL", f"Forecast status={forecast.status.value}", "price_forecast",
            )],
            "completed_tasks": ["price_forecast"],
        }

