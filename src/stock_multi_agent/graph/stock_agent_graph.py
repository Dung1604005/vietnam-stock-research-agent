from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from stock_multi_agent.agents import DataAgent, FundamentalAgent, ManagerAgent, RiskAgent, TechnicalAgent
from stock_multi_agent.models.state import AgentState


class StockAgentGraph:
    """Explicit supervisor graph with parallel specialists and a bounded cycle."""

    def __init__(
        self,
        data_agent: DataAgent,
        technical_agent: TechnicalAgent,
        fundamental_agent: FundamentalAgent,
        risk_agent: RiskAgent,
        manager_agent: ManagerAgent,
    ) -> None:
        self.data_agent = data_agent
        self.technical_agent = technical_agent
        self.fundamental_agent = fundamental_agent
        self.risk_agent = risk_agent
        self.manager_agent = manager_agent
        self.compiled = self._build().compile()

    def _build(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("manager_plan", self.manager_agent.plan)
        graph.add_node("data", self.data_agent.run)
        graph.add_node("dispatch_specialists", self.manager_agent.dispatch_specialists)
        graph.add_node("technical", lambda state: self.technical_agent.run(state, extended=False))
        graph.add_node("fundamental", self.fundamental_agent.run)
        graph.add_node("dispatch_risk", self.manager_agent.dispatch_risk)
        graph.add_node("risk", self.risk_agent.run)
        graph.add_node("manager_review", self.manager_agent.review)
        graph.add_node("technical_reanalysis", lambda state: self.technical_agent.run(state, extended=True))
        graph.add_node("risk_reanalysis", self.risk_agent.run)
        graph.add_node("finalize", self.manager_agent.finalize)

        graph.add_edge(START, "manager_plan")
        graph.add_edge("manager_plan", "data")
        graph.add_edge("data", "dispatch_specialists")
        graph.add_edge("dispatch_specialists", "technical")
        graph.add_edge("dispatch_specialists", "fundamental")
        graph.add_edge(["technical", "fundamental"], "dispatch_risk")
        graph.add_edge("dispatch_risk", "risk")
        graph.add_edge("risk", "manager_review")
        graph.add_conditional_edges(
            "manager_review",
            self.manager_agent.route_after_review,
            {"reanalyze": "technical_reanalysis", "finalize": "finalize"},
        )
        graph.add_edge("technical_reanalysis", "risk_reanalysis")
        graph.add_edge("risk_reanalysis", "manager_review")
        graph.add_edge("finalize", END)
        return graph

    def invoke(self, state: AgentState) -> AgentState:
        output = self.compiled.invoke(state)
        return output if isinstance(output, AgentState) else AgentState.model_validate(output)

