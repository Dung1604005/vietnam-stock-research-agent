from __future__ import annotations

from typing import Any, Literal

from stock_multi_agent.models.messages import AgentMessage, AgentName, MessageType
from stock_multi_agent.models.agent_contracts import AgentExecutionReport, AgentSpec
from stock_multi_agent.models.state import TraceEvent
from stock_multi_agent.services.request_telemetry import observed


class BaseAgent:
    name: AgentName
    goal: str
    spec: AgentSpec
    llm_advisor: Any | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for method in ("run", "plan", "review", "finalize"):
            if method in cls.__dict__:
                setattr(cls, method, observed(f"{cls.__name__}.{method}")(cls.__dict__[method]))

    def message(
        self,
        receiver: AgentName,
        message_type: MessageType,
        task: str,
        payload: dict[str, Any] | None = None,
        confidence: float | None = None,
    ) -> AgentMessage:
        from stock_multi_agent.services.agent_execution import current_execution
        from uuid import uuid4
        context = current_execution()
        return AgentMessage(
            sender=self.name,
            receiver=receiver,
            message_type=message_type,
            task=task,
            payload=payload or {},
            confidence=confidence,
            **({'run_id': context.run_id, 'task_id': str(uuid4()) if message_type in {MessageType.TASK, MessageType.REQUEST_REANALYSIS} else context.task_id,
                'reply_to': context.reply_to, 'input_fingerprint': context.input_fingerprint}
               if context else {}),
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

