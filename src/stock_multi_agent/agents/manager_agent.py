from __future__ import annotations

from stock_multi_agent.agents.base_agent import BaseAgent
from stock_multi_agent.models.analysis import Recommendation, RiskLevel, Signal, FinalDecision
from stock_multi_agent.models.messages import AgentMessage, AgentName, MessageType
from stock_multi_agent.models.state import AgentState


class ManagerAgent(BaseAgent):
    name = AgentName.MANAGER
    goal = "Plan and coordinate specialists, resolve conflict, and produce a cautious educational recommendation."

    def plan(self, state: AgentState) -> dict[str, object]:
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
        if state.technical_result.signal != state.fundamental_result.signal:
            conflicts.append(
                f"Technical {state.technical_result.signal.value} vs Fundamental {state.fundamental_result.signal.value}"
            )
        if state.risk_result.risk_level == RiskLevel.HIGH:
            conflicts.append("Risk Agent rates the evidence HIGH risk")
        if state.risk_result.veto:
            conflicts.append("Risk Agent issued a veto")
        if min(state.technical_result.confidence, state.fundamental_result.confidence) < 0.5:
            conflicts.append("Specialist confidence is low")
        return conflicts

    def review(self, state: AgentState) -> dict[str, object]:
        conflicts = self.conflicts(state)
        already_extended = bool(state.technical_result and state.technical_result.timeframe.startswith("extended"))
        can_retry = state.iteration < state.max_iterations and not already_extended
        request = bool(conflicts) and can_retry
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
        agreements: list[str] = []
        if technical.signal == fundamental.signal:
            agreements.append(f"Technical and Fundamental both return {technical.signal.value}")
        if technical.signal == Signal.BUY and fundamental.signal == Signal.BUY and risk.risk_level != RiskLevel.HIGH and not risk.veto:
            recommendation = Recommendation.BUY
        elif risk.veto or (risk.risk_level == RiskLevel.HIGH and Signal.SELL in (technical.signal, fundamental.signal)):
            recommendation = Recommendation.AVOID
        elif technical.signal == fundamental.signal == Signal.HOLD and risk.risk_level != RiskLevel.HIGH:
            recommendation = Recommendation.HOLD
        else:
            recommendation = Recommendation.WATCH

        base_confidence = (technical.confidence + fundamental.confidence + max(0.1, 1 - risk.risk_score / 10)) / 3
        confidence = max(0.15, min(0.9, base_confidence - 0.05 * len(conflicts)))
        evidence = [
            *(f"{item.label}: {item.value}" for item in technical.evidence[:3]),
            *(f"{item.label}: {item.value}" for item in fundamental.evidence[:3]),
            risk.reasoning_summary,
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
                f"Manager combined independent specialist evidence after {state.iteration} re-analysis round(s). "
                f"Risk constraints and unresolved uncertainty lead to {recommendation.value}."
            ),
            evidence=evidence,
        )
        return {
            "final_decision": final,
            "reanalysis_requested": False,
            "messages": [self.message(AgentName.USER, MessageType.FINAL_DECISION, "Final educational assessment", final.model_dump(mode="json"), final.confidence)],
            "trace": [self.event("ACT", "FINAL_DECISION", f"{recommendation.value}, confidence={confidence:.2f}")],
            "completed_tasks": ["final_decision"],
        }
