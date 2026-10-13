from stock_multi_agent.models.messages import MessageType
from stock_multi_agent.models.data import DataQualityIssue, DataQualityReport, QualitySeverity, QualityStatus
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


def test_recoverable_data_conflict_refreshes_full_analysis_pipeline() -> None:
    data = conflict_market_data()
    data.quality_report = DataQualityReport(
        status=QualityStatus.WARN,
        issues=[DataQualityIssue(
            code="STALE_PRICE",
            severity=QualitySeverity.WARNING,
            message="Latest price is stale",
            field="bars.date",
        )],
    )
    service = AnalysisService(provider=InMemoryProvider(data))

    state = service.analyze("FPT", "3 months", max_iterations=2)

    assert "data_reanalysis" in state.completed_tasks
    assert "research_reanalysis" in state.completed_tasks
    assert "technical_reanalysis" in state.completed_tasks
    assert state.reanalysis_history[0].value == "DATA_PIPELINE"
    assert state.iteration == 1
    assert state.final_decision is not None
