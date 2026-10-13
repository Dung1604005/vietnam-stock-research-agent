from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from pydantic import BaseModel, Field

from stock_multi_agent.agents.base_agent import BaseAgent
from stock_multi_agent.models.agent_contracts import AgentSpec, ManagerContext
from stock_multi_agent.models.analysis import Recommendation, RiskLevel, Signal, FinalDecision
from stock_multi_agent.models.chat import AssistantResponse, ParsedQuery, QueryIntent, ResponseType
from stock_multi_agent.models.conflicts import ConflictKind, ConflictRecord, ReanalysisAction
from stock_multi_agent.models.messages import AgentMessage, AgentName, MessageType
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.providers.llm_provider import SemanticLLMProvider, configured_semantic_provider


class _ManagerRouteChoice(BaseModel):
    intent: QueryIntent
    confidence: float = Field(ge=0.0, le=1.0)


class ManagerRouteSelection(BaseModel):
    selected_intent: QueryIntent
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    llm_used: bool = False
    accepted: bool = False
    fallback_reason: str | None = None


class _ManagerSynthesisOutput(BaseModel):
    lines: list[str] = Field(min_length=4, max_length=6)
    cited_source_ids: list[str] = Field(default_factory=list, max_length=8)


class ManagerResponseSynthesis(BaseModel):
    status: str
    lines: list[str] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None


class ManagerAgent(BaseAgent):
    name = AgentName.MANAGER
    goal = "Plan and coordinate specialists, resolve conflict, and produce a cautious educational recommendation."
    spec = AgentSpec(
        role=name,
        goal=goal,
        required_context=[
            "user_goal", "technical_result", "fundamental_result", "risk_result", "research_evidence",
        ],
        allowed_tools=["structured_semantic_router"],
        output_type="FinalDecision",
        can_use_llm=True,
        decision_constraints=[
            "LLM may select only a typed intent; deterministic routing validates and selects executable agents",
            "Do not recompute specialist metrics",
            "Do not emit BUY when a risk veto or HIGH data uncertainty is present",
            "Require sufficient evidence coverage before BUY",
            "Use deterministic synthesis so the same inputs produce the same decision",
        ],
    )

    def __init__(
        self,
        llm_advisor=None,
        llm_provider: SemanticLLMProvider | None = None,
        routing_enabled: bool | None = None,
        synthesis_enabled: bool | None = None,
    ) -> None:
        self.llm_advisor = llm_advisor
        self.llm_provider = llm_provider if llm_provider is not None else configured_semantic_provider()
        configured = os.getenv("STOCK_AGENT_LLM_MANAGER_ROUTING_ENABLED", "false").strip().lower()
        self.routing_enabled = routing_enabled if routing_enabled is not None else configured in {
            "1", "true", "yes", "on",
        }
        synthesis_configured = os.getenv("STOCK_AGENT_LLM_MANAGER_SYNTHESIS_ENABLED", "false").strip().lower()
        self.synthesis_enabled = synthesis_enabled if synthesis_enabled is not None else synthesis_configured in {
            "1", "true", "yes", "on",
        }

    def summarize_response(self, query: ParsedQuery, response: AssistantResponse) -> ManagerResponseSynthesis:
        """Create one grounded Vietnamese summary after all deterministic work is complete."""
        if not self.synthesis_enabled:
            return ManagerResponseSynthesis(status="SKIPPED", fallback_reason="manager_synthesis_disabled")
        if self.llm_provider is None:
            return ManagerResponseSynthesis(status="UNAVAILABLE", fallback_reason="no_configured_llm")
        if response.response_type not in {
            ResponseType.ANALYSIS,
            ResponseType.COMPARISON,
            ResponseType.MARKET_OVERVIEW,
            ResponseType.STOCK_LIST,
        }:
            return ManagerResponseSynthesis(status="SKIPPED", fallback_reason="unsupported_response_type")

        allowed_source_ids = list(dict.fromkeys(item.source_id for item in response.source_details))[:20]
        safe_metadata_keys = {
            "recommendation", "confidence", "technical_score", "technical_signal",
            "fundamental_score", "fundamental_signal", "risk_score", "risk_level", "veto",
            "requested_count", "returned_count", "screen_status", "forecast_status",
        }
        facts: dict[str, Any] = {
            "user_request": query.raw_query[:800],
            "intent": query.intent.value,
            "tickers": response.tickers[:12],
            "investment_horizon": query.investment_horizon,
            "deterministic_response": response.message[:3500],
            "tables": [
                {
                    "title": table.title,
                    "rows": table.rows[:10],
                    "note": table.note,
                }
                for table in response.tables[:2]
            ],
            "decision_facts": {
                key: value for key, value in response.metadata.items() if key in safe_metadata_keys
            },
            "warnings": response.warnings[:6],
            "allowed_source_ids": allowed_source_ids,
        }
        system = """Bạn là Manager Agent của hệ thống nghiên cứu chứng khoán Việt Nam.
Từ facts đã được các agent và Python tính xong, hãy viết 4 đến 6 câu tiếng Việt ngắn để tổng kết:
mục tiêu người dùng, bằng chứng chính, điểm mâu thuẫn hoặc bất định, và kết luận hiện có.
Không tự tính lại, không thay đổi số liệu, tín hiệu, dự báo, điểm số hay kết luận deterministic.
Không tạo nguồn mới và chỉ trích dẫn mã có trong allowed_source_ids. Không đưa ra cam kết lợi nhuận hay lệnh giao dịch.
Mỗi phần tử lines là đúng một câu tiếng Việt; trả về đúng JSON schema."""
        try:
            result = self.llm_provider.parse_structured(system, facts, _ManagerSynthesisOutput)
        except Exception as exc:
            return ManagerResponseSynthesis(
                status="UNAVAILABLE",
                fallback_reason=f"manager_synthesis_error:{type(exc).__name__}",
            )
        joined = " ".join(result.lines)
        if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]", joined):
            return ManagerResponseSynthesis(status="UNAVAILABLE", fallback_reason="non_vietnamese_output")
        allowed = set(allowed_source_ids)
        return ManagerResponseSynthesis(
            status="GROUNDED",
            lines=[line.strip() for line in result.lines if line.strip()],
            cited_source_ids=[item for item in result.cited_source_ids if item in allowed],
        )

    def choose_chat_intent(self, query: ParsedQuery) -> ManagerRouteSelection:
        """Let the LLM interpret the request, then keep execution behind typed guards."""
        fallback = ManagerRouteSelection(
            selected_intent=query.intent,
            confidence=query.confidence,
            rationale="Giữ kết quả semantic parser và deterministic router.",
        )
        if not self.routing_enabled:
            return fallback.model_copy(update={"fallback_reason": "manager_llm_routing_disabled"})
        if self.llm_provider is None:
            return fallback.model_copy(update={"fallback_reason": "no_configured_llm"})
        if query.needs_clarification:
            return fallback.model_copy(update={"fallback_reason": "query_requires_clarification"})

        system = """Bạn là Manager Agent điều phối hệ thống nghiên cứu chứng khoán Việt Nam.
Hãy xác định đúng một intent từ JSON schema dựa trên câu hỏi và các entity đã trích xuất.
STOCK_RECOMMENDATION dùng khi người dùng hỏi nên mua/đầu tư mã nào hoặc yêu cầu top mã để mua.
STOCK_DISCOVERY chỉ dùng cho sàng lọc trung lập như mã đáng chú ý, RSI thấp hoặc lọc theo ngành.
Không tự tạo ticker, số lượng hay dữ liệu. Bạn chỉ chọn intent; deterministic router sẽ chọn agent và
kiểm tra quyền thực thi. Trả đúng JSON schema và không thêm nội dung ngoài schema."""
        payload = {
            "query": query.raw_query,
            "normalized_query": query.normalized_query,
            "semantic_intent": query.intent.value,
            "tickers": query.tickers,
            "sector": query.sector,
            "requested_result_count": query.requested_result_count,
            "requested_actions": query.requested_actions,
            "requested_indicators": query.requested_indicators,
            "investment_horizon": query.investment_horizon,
            "allowed_intents": [item.value for item in QueryIntent],
        }
        try:
            choice = self.llm_provider.parse_structured(system, payload, _ManagerRouteChoice)
        except Exception as exc:
            return fallback.model_copy(update={
                "fallback_reason": f"manager_llm_error:{type(exc).__name__}",
            })

        invalid_reason = self._invalid_chat_intent(query, choice.intent)
        if invalid_reason:
            return fallback.model_copy(update={
                "llm_used": True,
                "fallback_reason": invalid_reason,
            })
        return ManagerRouteSelection(
            selected_intent=choice.intent,
            confidence=choice.confidence,
            rationale=f"Manager LLM chọn intent {choice.intent.value}; deterministic router sẽ kiểm tra và chọn agent.",
            llm_used=True,
            accepted=True,
        )

    @staticmethod
    def _invalid_chat_intent(query: ParsedQuery, intent: QueryIntent) -> str | None:
        ticker_required = {
            QueryIntent.STOCK_ANALYSIS,
            QueryIntent.TECHNICAL_ANALYSIS,
            QueryIntent.FUNDAMENTAL_ANALYSIS,
            QueryIntent.RISK_ANALYSIS,
            QueryIntent.STOCK_CHART,
            QueryIntent.INDICATOR_QUERY,
            QueryIntent.MIXED,
        }
        if intent in ticker_required and not query.tickers:
            return "manager_selected_intent_without_required_ticker"
        if intent is QueryIntent.STOCK_COMPARISON and len(query.tickers) < 2:
            return "manager_selected_comparison_without_two_tickers"
        if intent is QueryIntent.STOCK_RECOMMENDATION and query.tickers:
            return "manager_selected_universe_recommendation_for_named_ticker"
        if not query.tickers and "BUY" in query.requested_actions and intent is not QueryIntent.STOCK_RECOMMENDATION:
            return "explicit_buy_shortlist_must_use_recommendation"
        return None

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
    def assess_conflicts(state: AgentState) -> list[ConflictRecord]:
        if state.technical_result is None or state.fundamental_result is None or state.risk_result is None:
            return [ConflictRecord(
                kind=ConflictKind.MISSING_SPECIALIST_RESULT,
                detail="Required specialist result is missing",
            )]
        conflicts: list[ConflictRecord] = []
        if (
            state.technical_result.signal != state.fundamental_result.signal
            and Signal.UNKNOWN not in (state.technical_result.signal, state.fundamental_result.signal)
        ):
            action = (
                ReanalysisAction.RESEARCH_FUNDAMENTAL
                if state.fundamental_result.confidence <= state.technical_result.confidence
                else ReanalysisAction.TECHNICAL
            )
            conflicts.append(ConflictRecord(
                kind=ConflictKind.SPECIALIST_DISAGREEMENT,
                detail=f"Technical {state.technical_result.signal.value} vs Fundamental {state.fundamental_result.signal.value}",
                suggested_action=action,
            ))
        if state.technical_result.signal is Signal.UNKNOWN:
            conflicts.append(ConflictRecord(
                kind=ConflictKind.TECHNICAL_EVIDENCE_INSUFFICIENT,
                detail="Technical evidence is insufficient",
                suggested_action=ReanalysisAction.TECHNICAL,
            ))
        if state.fundamental_result.signal is Signal.UNKNOWN:
            conflicts.append(ConflictRecord(
                kind=ConflictKind.FUNDAMENTAL_EVIDENCE_INSUFFICIENT,
                detail="Fundamental evidence is insufficient",
                suggested_action=ReanalysisAction.RESEARCH_FUNDAMENTAL,
            ))
        if state.risk_result.risk_level == RiskLevel.HIGH:
            conflicts.append(ConflictRecord(
                kind=ConflictKind.HIGH_MEASURED_RISK,
                detail="Risk Agent rates the evidence HIGH risk",
                suggested_action=ReanalysisAction.TECHNICAL,
            ))
        if state.risk_result.data_uncertainty_level == RiskLevel.HIGH:
            conflicts.append(ConflictRecord(
                kind=ConflictKind.HIGH_DATA_UNCERTAINTY,
                detail="Data uncertainty is HIGH",
                suggested_action=ReanalysisAction.DATA_PIPELINE,
            ))
        if state.risk_result.veto:
            conflicts.append(ConflictRecord(
                kind=ConflictKind.RISK_VETO,
                detail="Risk Agent issued a veto",
            ))
        if min(state.technical_result.confidence, state.fundamental_result.confidence) < 0.5:
            action = (
                ReanalysisAction.TECHNICAL
                if state.technical_result.confidence < state.fundamental_result.confidence
                else ReanalysisAction.RESEARCH_FUNDAMENTAL
            )
            conflicts.append(ConflictRecord(
                kind=ConflictKind.LOW_SPECIALIST_CONFIDENCE,
                detail="Specialist confidence is low",
                suggested_action=action,
            ))
        recoverable_market_codes = {
            "STALE_PRICE", "PRICE_GAPS", "INSUFFICIENT_HISTORY", "INVALID_CHRONOLOGY",
            "FUTURE_AS_OF", "MISSING_SOURCE_REF", "LOW_VOLUME_COVERAGE",
        }
        recoverable_financial_codes = {
            "INSUFFICIENT_FINANCIALS", "MISSING_FINANCIAL_PERIOD",
            "STALE_FINANCIALS", "STALE_FINANCIAL_RATIOS",
        }
        quality_codes = {item.code for item in state.market_data.quality_report.issues} if state.market_data else set()
        relevant_codes = quality_codes & recoverable_market_codes
        if (
            state.fundamental_result.signal is Signal.UNKNOWN
            or state.fundamental_result.confidence < 0.5
        ):
            relevant_codes |= quality_codes & recoverable_financial_codes
        if relevant_codes:
            conflicts.append(ConflictRecord(
                kind=ConflictKind.DATA_QUALITY_LIMITATION,
                detail=f"Recoverable data-quality limitations: {', '.join(sorted(relevant_codes))}",
                suggested_action=ReanalysisAction.DATA_PIPELINE,
            ))
        return conflicts

    @classmethod
    def conflicts(cls, state: AgentState) -> list[str]:
        """Backward-compatible human-readable conflict list used by final synthesis."""
        return [item.detail for item in cls.assess_conflicts(state)]

    @staticmethod
    def _evidence_fingerprint(state: AgentState) -> str:
        market = state.market_data
        payload = {
            "market": None if market is None else {
                "as_of": market.as_of.isoformat() if market.as_of else None,
                "bars": len(market.bars),
                "last_bar": market.bars[-1].model_dump(mode="json") if market.bars else None,
                "financials": market.financials.model_dump(mode="json"),
                "financial_periods": sorted(
                    item.period_end.isoformat()
                    for item in [*market.financial_observations, *market.financial_statement_observations]
                    if item.period_end is not None
                ),
                "source_ids": sorted(item.source_id for item in market.source_refs),
                "quality_codes": sorted(item.code for item in market.quality_report.issues),
            },
            "research_chunks": sorted(item.chunk_id for item in state.research_evidence),
            "technical": state.technical_result.model_dump(mode="json") if state.technical_result else None,
            "fundamental": state.fundamental_result.model_dump(mode="json") if state.fundamental_result else None,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _select_reanalysis_action(
        conflicts: list[ConflictRecord],
        history: list[ReanalysisAction],
    ) -> ReanalysisAction:
        attempted = set(history)
        if ReanalysisAction.DATA_PIPELINE in attempted:
            attempted.update({
                ReanalysisAction.RESEARCH_FUNDAMENTAL,
                ReanalysisAction.TECHNICAL,
            })
        if (
            any(item.kind is ConflictKind.RISK_VETO for item in conflicts)
            and not any(item.suggested_action is ReanalysisAction.DATA_PIPELINE for item in conflicts)
        ):
            return ReanalysisAction.NONE
        priorities = (
            ReanalysisAction.DATA_PIPELINE,
            ReanalysisAction.RESEARCH_FUNDAMENTAL,
            ReanalysisAction.TECHNICAL,
        )
        suggested = {item.suggested_action for item in conflicts}
        for action in priorities:
            if action in suggested and action not in attempted:
                return action
        return ReanalysisAction.NONE

    def review(self, state: AgentState) -> dict[str, object]:
        conflict_records = self.assess_conflicts(state)
        conflicts = [item.detail for item in conflict_records]
        current_fingerprint = self._evidence_fingerprint(state)
        previous_attempt_changed_evidence = not (
            state.reanalysis_history
            and state.reanalysis_evidence_fingerprint == current_fingerprint
        )
        can_retry = state.iteration < state.max_iterations and previous_attempt_changed_evidence
        action = (
            self._select_reanalysis_action(conflict_records, state.reanalysis_history)
            if conflicts and can_retry
            else ReanalysisAction.NONE
        )
        request = action is not ReanalysisAction.NONE
        messages: list[AgentMessage] = []
        trace = [self.event("OBSERVE", "DETECT_CONFLICT", "; ".join(conflicts) if conflicts else "No material conflict")]
        iteration = state.iteration
        plan = state.current_plan
        if request:
            iteration += 1
            task_by_action = {
                ReanalysisAction.DATA_PIPELINE: "Refresh market and financial evidence, then rerun affected specialists",
                ReanalysisAction.RESEARCH_FUNDAMENTAL: "Expand cited research and reassess fundamentals",
                ReanalysisAction.TECHNICAL: "Re-check long-term trend, SMA200, support/resistance, and drawdown",
            }
            recipient_by_action = {
                ReanalysisAction.DATA_PIPELINE: AgentName.DATA,
                ReanalysisAction.RESEARCH_FUNDAMENTAL: AgentName.RESEARCH,
                ReanalysisAction.TECHNICAL: AgentName.TECHNICAL,
            }
            task = task_by_action[action]
            plan = [*state.current_plan, task]
            messages.append(self.message(
                recipient_by_action[action],
                MessageType.REQUEST_REANALYSIS,
                task,
                {
                    "conflicts": conflicts,
                    "action": action.value,
                    "iteration": iteration,
                    "max_iterations": state.max_iterations,
                },
            ))
            trace.extend([
                self.event("PLAN", "REVISE_PLAN", f"Conflict classified; selected {action.value}"),
                self.event("ACT", "REQUEST_REANALYSIS", f"{action.value} iteration {iteration}/{state.max_iterations}"),
            ])
        elif conflicts:
            reason = (
                "Evidence did not change after the previous re-analysis"
                if not previous_attempt_changed_evidence
                else "Re-analysis budget exhausted or no untried evidence-producing action remains"
            )
            trace.append(self.event("PLAN", "STOP_LOOP", f"{reason}; use conservative synthesis"))
        return {
            "conflict_detected": bool(conflicts),
            "reanalysis_requested": request,
            "active_conflicts": conflict_records,
            "reanalysis_action": action,
            "reanalysis_history": [action] if request else [],
            "reanalysis_evidence_fingerprint": current_fingerprint if request else state.reanalysis_evidence_fingerprint,
            "iteration": iteration,
            "current_plan": plan,
            "messages": messages,
            "trace": trace,
        }

    @staticmethod
    def route_after_review(state: AgentState) -> str:
        if not state.reanalysis_requested or state.iteration > state.max_iterations:
            return "finalize"
        return {
            ReanalysisAction.TECHNICAL: "technical",
            ReanalysisAction.RESEARCH_FUNDAMENTAL: "research_fundamental",
            ReanalysisAction.DATA_PIPELINE: "data_pipeline",
        }.get(state.reanalysis_action, "finalize")

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
