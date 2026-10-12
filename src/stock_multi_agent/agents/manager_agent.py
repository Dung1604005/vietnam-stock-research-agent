from __future__ import annotations

from stock_multi_agent.agents.base_agent import BaseAgent
from stock_multi_agent.models.agent_contracts import AgentSpec, ManagerContext
from stock_multi_agent.models.analysis import Recommendation, RiskLevel, Signal, FinalDecision
from stock_multi_agent.models.messages import AgentMessage, AgentName, MessageType
from stock_multi_agent.models.state import AgentState


class ManagerAgent(BaseAgent):
    name = AgentName.MANAGER
    goal = "Plan and coordinate specialists, resolve conflict, and produce a cautious educational recommendation."
    spec = AgentSpec(
        role=name,
        goal=goal,
        required_context=[
            "user_goal", "technical_result", "fundamental_result", "risk_result", "research_evidence",
        ],
        allowed_tools=[],
        output_type="FinalDecision",
        can_use_llm=False,
        decision_constraints=[
            "Do not recompute specialist metrics",
            "Do not emit BUY when a risk veto or HIGH data uncertainty is present",
            "Require sufficient evidence coverage before BUY",
            "Use deterministic synthesis so the same inputs produce the same decision",
        ],
    )

    def __init__(self, llm_advisor=None) -> None:
        # Retain the argument for backwards compatibility. The Manager is kept
        # deterministic; LLM reasoning belongs to bounded specialist tasks.
        del llm_advisor
        self.llm_advisor = None

    def plan(self, state: AgentState) -> dict[str, object]:
        context = ManagerContext.from_state(state)
        plan = [
            "Acquire and validate real market/company data",
            "Run independent technical and fundamental assessments",
            "Ask Risk Agent to criticize both assessments",
            "Detect conflicts and request bounded re-analysis if useful",
            "Synthesize a conservative educational decision",
        ]
        task = self.message(AgentName.DATA, MessageType.TASK, "Fetch and normalize real data", {"ticker": state.ticker})
        return {
            "current_plan": plan,
            "messages": [task],
            "trace": [
                self.event("OBSERVE", "READ_GOAL", state.user_goal),
                self.event("PLAN", "CREATE_PLAN", " → ".join(plan)),
                self.event("ACT", "SEND_TASK", "Delegate real-data acquisition to Data Agent"),
            ],
            "agent_reports": [self.report(
                stage="planning",
                status="COMPLETED",
                summary=f"Created a bounded specialist plan for {context.ticker}.",
                confidence=1.0,
                output_key="current_plan",
            )],
        }

    def dispatch_specialists(self, state: AgentState) -> dict[str, object]:
        messages = [
            self.message(AgentName.TECHNICAL, MessageType.TASK, "Assess trend for requested horizon"),
            self.message(AgentName.FUNDAMENTAL, MessageType.TASK, "Assess business quality and valuation independently"),
        ]
        return {
            "messages": messages,
            "trace": [self.event("ACT", "DISPATCH_PARALLEL", "Technical and Fundamental receive independent tasks")],
        }

    def dispatch_risk(self, state: AgentState) -> dict[str, object]:
        del state
        return {
            "messages": [self.message(AgentName.RISK, MessageType.TASK, "Critique specialist conclusions and test uncertainty")],
            "trace": [self.event("OBSERVE", "COLLECT_RESULTS", "Both specialist branches joined"), self.event("ACT", "SEND_TASK", "Send both results to Risk Agent")],
        }

    @staticmethod
    def conflicts(state: AgentState) -> list[str]:
        if state.technical_result is None or state.fundamental_result is None or state.risk_result is None:
            return ["Required specialist result is missing"]
        conflicts: list[str] = []
        if (
            state.technical_result.signal != state.fundamental_result.signal
            and Signal.UNKNOWN not in (state.technical_result.signal, state.fundamental_result.signal)
        ):
            conflicts.append(
                f"Technical {state.technical_result.signal.value} vs Fundamental {state.fundamental_result.signal.value}"
            )
        if state.technical_result.signal is Signal.UNKNOWN:
            conflicts.append("Technical evidence is insufficient")
        if state.fundamental_result.signal is Signal.UNKNOWN:
            conflicts.append("Fundamental evidence is insufficient")
        if state.risk_result.risk_level == RiskLevel.HIGH:
            conflicts.append("Risk Agent rates the evidence HIGH risk")
        if state.risk_result.data_uncertainty_level == RiskLevel.HIGH:
            conflicts.append("Data uncertainty is HIGH")
        if state.risk_result.veto:
            conflicts.append("Risk Agent issued a veto")
        if min(state.technical_result.confidence, state.fundamental_result.confidence) < 0.5:
            conflicts.append("Specialist confidence is low")
        return conflicts

    def review(self, state: AgentState) -> dict[str, object]:
        conflicts = self.conflicts(state)
        already_extended = bool(state.technical_result and state.technical_result.timeframe.startswith("extended"))
        can_retry = state.iteration < state.max_iterations and not already_extended
        technical_recheck_needed = bool(
            state.technical_result
            and state.fundamental_result
            and (
                state.technical_result.signal is Signal.UNKNOWN
                or (
                    state.technical_result.signal != state.fundamental_result.signal
                    and state.fundamental_result.signal is not Signal.UNKNOWN
                )
                or (state.risk_result is not None and state.risk_result.risk_level is RiskLevel.HIGH)
            )
        )
        request = bool(conflicts) and can_retry and technical_recheck_needed
        messages: list[AgentMessage] = []
        trace = [self.event("OBSERVE", "DETECT_CONFLICT", "; ".join(conflicts) if conflicts else "No material conflict")]
        iteration = state.iteration
        plan = state.current_plan
        if request:
            iteration += 1
            plan = [*state.current_plan, "Re-check long-term trend, SMA200, support/resistance, and drawdown"]
            messages.append(self.message(
                AgentName.TECHNICAL,
                MessageType.REQUEST_REANALYSIS,
                "Re-analyze using an extended one-year context",
                {"conflicts": conflicts, "iteration": iteration, "max_iterations": state.max_iterations},
            ))
            trace.extend([
                self.event("PLAN", "REVISE_PLAN", "Conflict warrants extended technical evidence"),
                self.event("ACT", "REQUEST_REANALYSIS", f"Iteration {iteration}/{state.max_iterations}"),
            ])
        elif conflicts and not can_retry:
            trace.append(self.event("PLAN", "STOP_LOOP", "Use conservative synthesis; re-analysis exhausted or already extended"))
        return {
            "conflict_detected": bool(conflicts),
            "reanalysis_requested": request,
            "iteration": iteration,
            "current_plan": plan,
            "messages": messages,
            "trace": trace,
        }

    @staticmethod
    def route_after_review(state: AgentState) -> str:
        return "reanalyze" if state.reanalysis_requested and state.iteration <= state.max_iterations else "finalize"

    def finalize(self, state: AgentState) -> dict[str, object]:
        if state.technical_result is None or state.fundamental_result is None or state.risk_result is None:
            raise ValueError("Manager cannot finalize without all specialist results")
        technical = state.technical_result
        fundamental = state.fundamental_result
        risk = state.risk_result
        conflicts = self.conflicts(state)
        evidence_completeness = self._evidence_completeness(state)
        agreements: list[str] = []
        if technical.signal == fundamental.signal:
            agreements.append(f"Technical and Fundamental both return {technical.signal.value}")
        if (
            technical.signal == Signal.BUY
            and fundamental.signal == Signal.BUY
            and risk.risk_level != RiskLevel.HIGH
            and risk.data_uncertainty_level != RiskLevel.HIGH
            and not risk.veto
            and evidence_completeness >= 0.75
        ):
            recommendation = Recommendation.BUY
        elif risk.veto or (risk.risk_level == RiskLevel.HIGH and Signal.SELL in (technical.signal, fundamental.signal)):
            recommendation = Recommendation.AVOID
        elif technical.signal == fundamental.signal == Signal.HOLD and risk.risk_level != RiskLevel.HIGH:
            recommendation = Recommendation.HOLD
        else:
            recommendation = Recommendation.WATCH
        recommendation_label = {
            Recommendation.BUY: "MUA",
            Recommendation.HOLD: "NẮM GIỮ",
            Recommendation.WATCH: "THEO DÕI",
            Recommendation.AVOID: "TRÁNH",
        }[recommendation]

        base_confidence = (technical.confidence + fundamental.confidence + max(0.1, 1 - risk.risk_score / 10)) / 3
        confidence = max(
            0.15,
            min(0.9, (base_confidence - 0.05 * len(conflicts)) * (0.55 + 0.45 * evidence_completeness)),
        )
        evidence = [
            *(f"{item.label}: {item.value}" for item in technical.evidence[:3]),
            *(f"{item.label}: {item.value}" for item in fundamental.evidence[:3]),
            risk.reasoning_summary,
            *(f"Research: {item.title} ({item.publisher})" for item in state.research_evidence[:2]),
        ]
        final = FinalDecision(
            ticker=state.ticker,
            recommendation=recommendation,
            confidence=round(confidence, 2),
            technical_summary=technical.reasoning_summary,
            fundamental_summary=fundamental.reasoning_summary,
            risk_summary=risk.reasoning_summary,
            agreements=agreements,
            conflicts=conflicts,
            final_reasoning=(
                f"Manager đã tổng hợp bằng chứng độc lập từ các chuyên gia sau {state.iteration} vòng phân tích lại. "
                f"Các giới hạn rủi ro và yếu tố chưa chắc chắn dẫn đến kết luận {recommendation_label}."
            ),
            evidence=evidence,
            evidence_completeness=evidence_completeness,
        )
        return {
            "final_decision": final,
            "reanalysis_requested": False,
            "messages": [self.message(AgentName.USER, MessageType.FINAL_DECISION, "Final educational assessment", final.model_dump(mode="json"), final.confidence)],
            "trace": [
                self.event("ACT", "FINAL_DECISION", f"{recommendation.value}, confidence={confidence:.2f}"),
            ],
            "completed_tasks": ["final_decision"],
            "agent_reports": [self.report(
                stage="final_synthesis",
                status="PARTIAL" if evidence_completeness < 0.75 else "COMPLETED",
                summary=(
                    f"Produced {recommendation.value} with evidence completeness "
                    f"{evidence_completeness:.0%}."
                ),
                confidence=final.confidence,
                output_key="final_decision",
                tools_used=[],
                assumptions=["Specialist scores and vetoes are authoritative inputs"],
                missing_information=(
                    ["Evidence completeness is below the BUY threshold"]
                    if evidence_completeness < 0.75 else []
                ),
                evidence_count=len(evidence),
            )],
        }

    @staticmethod
    def _evidence_completeness(state: AgentState) -> float:
        """Measure coverage, not whether the evidence is positive or negative."""
        technical = state.technical_result
        fundamental = state.fundamental_result
        risk = state.risk_result
        score = 0.0
        if state.market_data is not None and state.market_data.source_refs:
            score += 0.25
        if technical is not None and technical.evidence:
            score += 0.20
        if fundamental is not None and fundamental.evidence and fundamental.signal is not Signal.UNKNOWN:
            score += 0.25
        if risk is not None and any(section.evidence for section in risk.sections):
            score += 0.15
        if state.research_evidence:
            score += 0.10
        if state.price_forecast is not None and state.price_forecast.forecast_price is not None:
            score += 0.05
        return round(min(1.0, score), 2)
