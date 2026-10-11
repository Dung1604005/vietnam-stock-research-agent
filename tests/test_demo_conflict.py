from stock_multi_agent.models.messages import MessageType
from stock_multi_agent.services.analysis_service import AnalysisService

from conftest import InMemoryProvider, conflict_market_data


def test_demo_conflict_requests_reanalysis_and_finishes() -> None:
    service = AnalysisService(provider=InMemoryProvider(conflict_market_data()))
    state = service.analyze("FPT", "3 months", max_iterations=3)
    message_types = [message.message_type for message in state.messages]
    assert MessageType.REQUEST_REANALYSIS in message_types
    assert MessageType.CRITIQUE in message_types
    assert MessageType.FINAL_DECISION in message_types
    assert state.iteration == 1
    assert "technical_reanalysis" in state.completed_tasks
    assert state.final_decision is not None
    assert state.final_decision.recommendation.value in {"WATCH", "HOLD", "AVOID"}


def test_conditional_graph_respects_zero_iterations() -> None:
    service = AnalysisService(provider=InMemoryProvider(conflict_market_data()))
    state = service.analyze("FPT", "3 months", max_iterations=0)
    assert MessageType.REQUEST_REANALYSIS not in [message.message_type for message in state.messages]
    assert state.iteration == 0
    assert state.final_decision is not None
