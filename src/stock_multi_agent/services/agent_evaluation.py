from __future__ import annotations

from stock_multi_agent.models.agent_contracts import AgentEvaluationSummary
from stock_multi_agent.models.analysis import Signal
from stock_multi_agent.models.data import QualitySeverity, QualityStatus
from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.models.state import AgentState


class AgentEvaluationService:
    """Separate orchestration health from supported analysis using typed outputs.

    Scores are heuristic coverage measures, not estimates of predictive accuracy.
    """

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
        expected = list(dict.fromkeys(
            self._infer_expected_agents(state) if expected_agents is None else expected_agents
        ))
        latest_reports = {}
        for report in state.agent_reports:
            previous = latest_reports.get(report.agent)
            primary = self.OUTPUT_KEYS.get(report.agent)
            # Auxiliary work (for example forecast owned by Data) must not
            # overwrite the authoritative acquisition report for market_data.
            if previous is None or report.output_key == primary or previous.output_key != primary:
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
        unavailable = [agent for agent in executed if latest_reports[agent].status == "UNAVAILABLE"]
        low_confidence = []
        insufficient = []
        analytical_issues = []
        supported_scores = []
        for agent in expected:
            report = latest_reports.get(agent)
            output_key = self.OUTPUT_KEYS[agent]
            output = getattr(state, output_key)
            issues = []
            confidence = min(
                report.confidence if report else 0.0,
                getattr(output, "confidence", 1.0),
            )
            if report is None:
                issues.append("missing execution report")
            elif report.output_key != output_key:
                issues.append(f"report describes {report.output_key}, expected {output_key}")
            if agent in unavailable:
                issues.append("reported UNAVAILABLE")
            if output_key in missing_outputs:
                issues.append(f"missing {output_key}")
            if report is not None and confidence < 0.5:
                low_confidence.append(agent)
                issues.append("confidence below 0.5")
            if getattr(output, "signal", None) is Signal.UNKNOWN:
                issues.append("analysis signal is UNKNOWN")
            if getattr(output, "assessment_status", None) == "UNAVAILABLE":
                issues.append("assessment is UNAVAILABLE")
            if agent in evidence_expected and (
                report is None or report.evidence_count == 0 or not self._has_evidence(agent, output)
            ):
                issues.append("no supporting evidence")
            if agent is AgentName.DATA and output is not None and (
                output.quality_report.status is QualityStatus.FAIL
                or any(item.severity is QualitySeverity.ERROR for item in output.quality_report.issues)
            ):
                issues.append("data quality checks failed")
            if agent is AgentName.MANAGER and output is not None and output.evidence_completeness < 0.75:
                issues.append("final evidence completeness below 0.75")
            if issues:
                insufficient.append(agent)
                analytical_issues.extend(f"{agent.value}: {issue}" for issue in issues)
                supported_scores.append(0.0)
            else:
                supported_scores.append(confidence * (0.9 if report.status == "PARTIAL" else 1.0))

        # A successful conservative AVOID is analytically valid when the
        # evidence is sound. Data/time/metric failures still prevent a pass.
        if AgentName.MANAGER in expected and all((state.technical_result, state.fundamental_result, state.risk_result)):
            from stock_multi_agent.agents.manager_agent import ManagerAgent
            unresolved = [
                item for item in ManagerAgent.assess_conflicts(state)
                if item.blocks_buy and item.dimension in {"data", "time", "metric"}
            ]
            analytical_issues.extend(item.detail for item in unresolved)
            if unresolved and AgentName.MANAGER not in insufficient:
                insufficient.append(AgentName.MANAGER)
                supported_scores[expected.index(AgentName.MANAGER)] = 0.0

        contract_coverage = len(executed) / len(expected) if expected else 0.0
        output_coverage = 1.0 - len(missing_outputs) / len(expected) if expected else 0.0
        evidence_coverage = (
            len(evidence_present) / len(evidence_expected) if evidence_expected else float(bool(expected))
        )
        execution_score = 0.4 * contract_coverage + 0.35 * output_coverage + 0.25 * evidence_coverage
        reports_match_outputs = all(
            latest_reports[agent].output_key == self.OUTPUT_KEYS[agent] for agent in executed
        )
        execution_passed = (
            bool(expected) and not missing_agents and not missing_outputs
            and reports_match_outputs and execution_score >= 0.75
        )
        analytical_sufficiency = sum(supported_scores) / len(expected) if expected else 0.0
        analytical_passed = execution_passed and not insufficient and analytical_sufficiency >= 0.5
        return AgentEvaluationSummary(
            expected_agents=expected,
            executed_agents=executed,
            missing_agents=missing_agents,
            missing_outputs=missing_outputs,
            low_confidence_agents=low_confidence,
            unavailable_agents=unavailable,
            insufficient_agents=insufficient,
            analytical_issues=analytical_issues,
            contract_coverage=round(contract_coverage, 3),
            output_coverage=round(max(0.0, output_coverage), 3),
            evidence_coverage=round(evidence_coverage, 3),
            overall_score=round(min(execution_score, analytical_sufficiency), 3),
            passed=analytical_passed,
            execution_score=round(execution_score, 3),
            execution_passed=execution_passed,
            analytical_sufficiency=round(analytical_sufficiency, 3),
            analytical_passed=analytical_passed,
        )

    @staticmethod
    def _has_evidence(agent: AgentName, output) -> bool:
        if output is None:
            return False
        if agent is AgentName.DATA:
            return bool(output.source_refs)
        if agent is AgentName.RESEARCH:
            return bool(output)
        if agent is AgentName.RISK:
            return any(section.evidence for section in output.sections)
        return bool(output.evidence)

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
        for report in state.agent_reports:
            if report.agent in cls.OUTPUT_KEYS and report.output_key == cls.OUTPUT_KEYS[report.agent]:
                expected.append(report.agent)
        return list(dict.fromkeys(expected))
