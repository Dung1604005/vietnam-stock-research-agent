from __future__ import annotations

from typing import Any, Literal

from stock_multi_agent.models.messages import AgentMessage, AgentName, MessageType
from stock_multi_agent.models.agent_contracts import AgentExecutionReport, AgentSpec
from stock_multi_agent.models.state import TraceEvent


class BaseAgent:
    name: AgentName
    goal: str
    spec: AgentSpec
    llm_advisor: Any | None = None

    def message(
        self,
        receiver: AgentName,
        message_type: MessageType,
        task: str,
        payload: dict[str, Any] | None = None,
        confidence: float | None = None,
    ) -> AgentMessage:
        return AgentMessage(
            sender=self.name,
            receiver=receiver,
            message_type=message_type,
            task=task,
            payload=payload or {},
            confidence=confidence,
        )

    def event(self, phase: str, action: str, detail: str, tool: str | None = None) -> TraceEvent:
        return TraceEvent(agent=self.name, phase=phase, action=action, detail=detail, tool=tool)

    def llm_review(self, task: str, facts: dict[str, Any], source_ids: list[str]):
        if not self.spec.can_use_llm or self.llm_advisor is None:
            return None
        return self.llm_advisor.review(self.name, task, facts, source_ids)

    def report(
        self,
        *,
        stage: str,
        status: Literal["COMPLETED", "PARTIAL", "UNAVAILABLE"],
        summary: str,
        confidence: float,
        output_key: str,
        tools_used: list[str] | None = None,
        assumptions: list[str] | None = None,
        missing_information: list[str] | None = None,
        evidence_count: int = 0,
    ) -> AgentExecutionReport:
        return AgentExecutionReport(
            agent=self.name,
            stage=stage,
            status=status,
            summary=summary,
            confidence=confidence,
            output_key=output_key,
            tools_used=tools_used or [],
            assumptions=assumptions or [],
            missing_information=missing_information or [],
            evidence_count=evidence_count,
        )

