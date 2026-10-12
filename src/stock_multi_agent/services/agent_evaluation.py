from __future__ import annotations

from stock_multi_agent.models.agent_contracts import AgentEvaluationSummary
from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.models.state import AgentState


class AgentEvaluationService:
    """Evaluate agent execution contracts without asking an LLM to grade itself."""

    OUTPUT_KEYS = {
        AgentName.DATA: "market_data",
        AgentName.RESEARCH: "research_evidence",
        AgentName.TECHNICAL: "technical_result",
        AgentName.FUNDAMENTAL: "fundamental_result",
        AgentName.RISK: "risk_result",
        AgentName.MANAGER: "final_decision",
    }
    EVIDENCE_AGENTS = {
        AgentName.DATA,
        AgentName.RESEARCH,
        AgentName.TECHNICAL,
        AgentName.FUNDAMENTAL,
        AgentName.RISK,
    }

    def evaluate(
        self,
        state: AgentState,
        expected_agents: list[AgentName] | None = None,
    ) -> AgentEvaluationSummary:
        expected = list(dict.fromkeys(expected_agents or self._infer_expected_agents(state)))
        latest_reports = {}
        for report in state.agent_reports:
            latest_reports[report.agent] = report
        executed = [agent for agent in expected if agent in latest_reports]
        missing_agents = [agent for agent in expected if agent not in latest_reports]

        missing_outputs = []
        for agent in expected:
            output_key = self.OUTPUT_KEYS[agent]
            value = getattr(state, output_key)
            # An empty research result remains a valid bounded output when its
            # report explicitly states PARTIAL/UNAVAILABLE.
            if value is None or (agent is not AgentName.RESEARCH and value == []):
                missing_outputs.append(output_key)

        evidence_expected = [agent for agent in expected if agent in self.EVIDENCE_AGENTS]
        evidence_present = [
            agent for agent in evidence_expected
            if agent in latest_reports and latest_reports[agent].evidence_count > 0
        ]
        low_confidence = [
            agent for agent in executed if latest_reports[agent].confidence < 0.5
        ]
        contract_coverage = len(executed) / len(expected) if expected else 1.0
        output_coverage = 1.0 - len(missing_outputs) / len(expected) if expected else 1.0
        evidence_coverage = (
            len(evidence_present) / len(evidence_expected) if evidence_expected else 1.0
        )
        overall = 0.4 * contract_coverage + 0.35 * output_coverage + 0.25 * evidence_coverage
        return AgentEvaluationSummary(
            expected_agents=expected,
            executed_agents=executed,
            missing_agents=missing_agents,
            missing_outputs=missing_outputs,
            low_confidence_agents=low_confidence,
            contract_coverage=round(contract_coverage, 3),
            output_coverage=round(max(0.0, output_coverage), 3),
            evidence_coverage=round(evidence_coverage, 3),
            overall_score=round(overall, 3),
            passed=not missing_agents and not missing_outputs and overall >= 0.75,
        )

    @classmethod
    def _infer_expected_agents(cls, state: AgentState) -> list[AgentName]:
        expected = [AgentName.DATA] if state.market_data is not None else []
        if state.research_evidence or any(report.agent is AgentName.RESEARCH for report in state.agent_reports):
            expected.append(AgentName.RESEARCH)
        if state.technical_result is not None:
            expected.append(AgentName.TECHNICAL)
        if state.fundamental_result is not None:
            expected.append(AgentName.FUNDAMENTAL)
        if state.risk_result is not None:
            expected.append(AgentName.RISK)
        if state.final_decision is not None:
            expected.append(AgentName.MANAGER)
        return expected
