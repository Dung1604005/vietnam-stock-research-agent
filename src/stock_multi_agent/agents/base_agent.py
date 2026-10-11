from __future__ import annotations

from typing import Any

from stock_multi_agent.models.messages import AgentMessage, AgentName, MessageType
from stock_multi_agent.models.state import TraceEvent


class BaseAgent:
    name: AgentName
    goal: str
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
        if self.llm_advisor is None:
            return None
        return self.llm_advisor.review(self.name, task, facts, source_ids)

