from datetime import datetime

import pytest
from pydantic import ValidationError

from stock_multi_agent.models.messages import AgentMessage, AgentName, MessageType
from stock_multi_agent.models.state import AgentState


def test_agent_message_is_typed_and_timestamped() -> None:
    message = AgentMessage(
        sender=AgentName.MANAGER,
        receiver=AgentName.TECHNICAL,
        message_type=MessageType.TASK,
        task="Analyze trend",
        confidence=0.8,
    )
    assert message.timestamp.utcoffset() is not None
    assert message.message_type is MessageType.TASK


def test_agent_message_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError):
        AgentMessage(
            sender=AgentName.MANAGER,
            receiver=AgentName.DATA,
            message_type=MessageType.TASK,
            task="Get data",
            timestamp=datetime(2026, 1, 1),
        )


def test_agent_state_enforces_iteration_bounds() -> None:
    state = AgentState(ticker="FPT", investment_horizon="3 months", user_goal="Analyze FPT", max_iterations=0)
    assert state.iteration == 0
    with pytest.raises(ValidationError):
        AgentState(ticker="FPT", investment_horizon="3 months", user_goal="Analyze FPT", max_iterations=-1)

