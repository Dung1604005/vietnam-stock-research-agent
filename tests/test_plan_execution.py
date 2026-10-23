"""Execution contracts use offline fixtures; they do not measure market accuracy."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from conftest import conflict_market_data
from stock_multi_agent.chat.router import IntentRouter, RouteStep, RoutingPlanValidator
from stock_multi_agent.models.chat import AssistantResponse, ConversationState, ParsedQuery, QueryIntent, ResponseType
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.chat_service import ChatService
from stock_multi_agent.services.plan_executor import PlanExecutor, TaskOutputError
from stock_multi_agent.services.request_telemetry import BudgetExceeded, current_metrics, request_scope


class Provider:
    def __init__(self):
        self.calls = []

    def get_market_data(self, ticker, include_financials=True):
        self.calls.append(ticker)
        return conflict_market_data(ticker)


@pytest.fixture
def service():
    return ChatService(analysis_service=AnalysisService(provider=Provider()), universe={"default": ["FPT", "CMG"]})


def query(intent=QueryIntent.STOCK_ANALYSIS, **updates):
    return ParsedQuery(raw_query="offline contract fixture", intent=intent, tickers=["FPT"], confidence=.99,
        **updates)


def forbidden(*args, **kwargs):
    raise AssertionError("This capability is not approved by the task plan")


def test_data_only_plan_cannot_dispatch_full_graph_or_enrichment(service, monkeypatch):
    parsed = query()
    plan = IntentRouter().plan(parsed)
    plan.tasks = plan.tasks[:1]
    monkeypatch.setattr(service.router, "plan", lambda *args: plan)
    monkeypatch.setattr(service.analysis, "analyze", forbidden)
    monkeypatch.setattr(service.llm_analyst, "enrich", forbidden)
    monkeypatch.setattr(service.analysis.manager_agent, "summarize_response", forbidden)

    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed)

    assert response.metadata["task_plan_status"] == "REJECTED"
    assert response.metadata["task_execution"][0]["status"] == "REJECTED"
    assert not service.analysis.data_agent.market_tool.provider.calls
    assert any("capabilities" in warning for warning in response.warnings)


def test_missing_secondary_capability_is_rejected_before_primary_execution(service, monkeypatch):
    parsed = query(secondary_intents=[QueryIntent.STOCK_CHART])
    plan = IntentRouter().plan(parsed)
    plan.tasks = [task for task in plan.tasks if task.step != RouteStep.CHART]
    monkeypatch.setattr(service.router, "plan", lambda *args: plan)
    monkeypatch.setattr(service.analysis, "analyze", forbidden)
    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed)
    assert response.metadata["task_plan_status"] == "REJECTED"


@pytest.mark.parametrize("mutation", ["outputs", "inputs", "operations", "contract", "intent", "condition", "budget"])
def test_contract_mutations_are_rejected(mutation):
    parsed = query(QueryIntent.TECHNICAL_ANALYSIS)
    plan = IntentRouter().plan(parsed)
    task = plan.tasks[-1]
    if mutation == "outputs": task.expected_outputs = ["arbitrary_output"]
    elif mutation == "inputs": task.required_inputs = []
    elif mutation == "operations": task.nested_operations.append("arbitrary_external_action")
    elif mutation == "contract": task.execution_contract = "unregistered-contract"
    elif mutation == "intent": plan.intent = QueryIntent.STOCK_ANALYSIS
    elif mutation == "condition": task.completion_condition = "Assume completion without output."
    elif mutation == "budget": task.budget_units = 0
    assert RoutingPlanValidator.validate(plan, parsed)


def test_selective_plan_executes_its_stages_without_full_graph_or_extra_specialists(service, monkeypatch):
    parsed = query(QueryIntent.TECHNICAL_ANALYSIS, requested_indicators=["RSI"])
    calls = []
    original = service.analysis.run_stage
    def stage(state, name, **kwargs):
        calls.append(name)
        return original(state, name, **kwargs)
    monkeypatch.setattr(service.analysis, "run_stage", stage)
    monkeypatch.setattr(service.analysis, "analyze", forbidden)
    monkeypatch.setattr(service.analysis, "analyze_technical", forbidden)
    monkeypatch.setattr(service.analysis.fundamental_agent, "run", forbidden)
    monkeypatch.setattr(service.analysis.risk_agent, "run", forbidden)
    monkeypatch.setattr(service.analysis.manager_agent, "summarize_response", forbidden)

    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed)

    assert calls == ["data", "forecast", "technical"]
    assert response.metadata["routing_plan"] == ["DATA", "TECHNICAL"]
    assert [item["step"] for item in response.metadata["task_execution"]] == ["DATA", "TECHNICAL"]
    assert response.metadata["technical_score"] is not None
    assert "optional_price_forecast" in response.metadata["task_plan"][0]["nested_operations"]
    assert response.metadata["manager_synthesis"]["status"] == "SKIPPED"


def test_risk_secondary_reuses_data_and_technical_result(service, monkeypatch):
    parsed = query(QueryIntent.TECHNICAL_ANALYSIS, secondary_intents=[QueryIntent.RISK_ANALYSIS])
    calls = []
    original = service.analysis.run_stage
    def stage(state, name, **kwargs):
        calls.append(name)
        return original(state, name, **kwargs)
    monkeypatch.setattr(service.analysis, "run_stage", stage)

    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed)

    assert calls == ["data", "forecast", "technical", "research", "fundamental", "risk"]
    assert response.metadata["secondary_outputs"]["risk"]["risk_level"]
    assert [item["step"] for item in response.metadata["task_execution"]] == response.metadata["routing_plan"]


def test_mixed_without_indicator_metadata_runs_the_planned_specialist(service, monkeypatch):
    parsed = query(QueryIntent.MIXED, normalized_query="giai thich va ap dung rsi")
    monkeypatch.setattr(service.knowledge, "answer", lambda *args, **kwargs: AssistantResponse(message="RSI evidence"))
    monkeypatch.setattr(service.analysis.fundamental_agent, "run", forbidden)

    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed)

    assert response.metadata["routing_plan"] == ["KNOWLEDGE", "DATA", "TECHNICAL"]
    assert response.metadata["mixed_components"] == ["knowledge", "technical"]


def test_graph_result_and_chart_share_the_same_approved_data(service, monkeypatch):
    parsed = query(secondary_intents=[QueryIntent.STOCK_CHART])
    monkeypatch.setattr(service.analysis, "fetch_data", forbidden)
    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed)
    assert response.response_type is ResponseType.ANALYSIS
    assert response.charts
    assert response.metadata["task_execution"][-1]["step"] == "CHART"
    assert response.metadata["task_execution"][-1]["status"] == "COMPLETED"


def test_missing_actual_output_is_not_marked_completed(service, monkeypatch):
    parsed = query(QueryIntent.TECHNICAL_ANALYSIS)
    original = service.analysis.run_stage
    monkeypatch.setattr(service.analysis, "run_stage",
        lambda state, name, **kwargs: state if name == "technical" else original(state, name, **kwargs))

    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed)

    technical = response.metadata["task_execution"][-1]
    assert technical["status"] == "UNAVAILABLE"
    assert technical["missing_outputs"] == ["technical_result"]
    assert response.metadata["task_plan_status"] == "PARTIAL"
    assert response.source_details


def test_full_graph_failed_specialist_is_visible_in_task_completion(service, monkeypatch):
    parsed = query()
    def fail(state, **kwargs):
        raise RuntimeError("offline specialist failure")
    monkeypatch.setattr(service.analysis.fundamental_agent, "run", fail)

    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed)

    fundamental = next(item for item in response.metadata["task_execution"] if item["step"] == "FUNDAMENTAL")
    assert response.response_type is ResponseType.ANALYSIS
    assert fundamental["status"] == "UNAVAILABLE"
    assert response.metadata["task_plan_status"] == "PARTIAL"
    assert response.metadata["recommendation"] != "BUY"


def test_secondary_error_retains_completed_full_analysis(service, monkeypatch):
    parsed = query(secondary_intents=[QueryIntent.STOCK_CHART])
    monkeypatch.setattr(service.charts, "build_artifact", forbidden)
    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed)
    assert response.response_type is ResponseType.ANALYSIS
    assert "recommendation" in response.metadata
    assert response.metadata["task_plan_status"] == "PARTIAL"
    assert response.metadata["task_execution"][-1]["status"] == "FAILED"


def test_schema_invalid_output_is_rejected():
    parsed = query(QueryIntent.STOCK_CHART)
    plan = IntentRouter().plan(parsed)
    executor = PlanExecutor(plan)
    with pytest.raises(ValidationError):
        executor.run(plan.tasks[0], lambda: {"market_data": "pretend data"})
    assert executor.snapshot()[0]["status"] == "FAILED"


def test_missing_dependency_output_prevents_its_consumer_from_running():
    plan = IntentRouter().plan(query(QueryIntent.STOCK_CHART))
    executor = PlanExecutor(plan)
    executor.run(plan.tasks[0], lambda: {"market_data": None})
    with pytest.raises(TaskOutputError, match="incomplete dependency"):
        executor.run(plan.tasks[1], forbidden)
    assert executor.snapshot()[1]["status"] == "SKIPPED"


def test_overdue_completed_artifacts_are_retained_and_next_task_cannot_start():
    plan = IntentRouter().plan(query(QueryIntent.STOCK_CHART))
    executor = PlanExecutor(plan)
    data = conflict_market_data()
    with request_scope() as metrics:
        def late_data():
            metrics.started -= metrics.budget.deadline_seconds + 1
            return {"market_data": data}
        result = executor.run(plan.tasks[0], late_data)
        assert result["market_data"] is data
        with pytest.raises(BudgetExceeded):
            executor.run(plan.tasks[1], forbidden)
    assert executor.snapshot()[0]["status"] == "PARTIAL"
    assert executor.snapshot()[0]["deadline_exceeded"] is True
    assert executor.snapshot()[1]["status"] == "SKIPPED"


def test_late_full_graph_keeps_primary_and_skips_external_postprocessing(service, monkeypatch):
    state = service.analysis.analyze("FPT", allow_llm_planning=False)
    def late_analysis(*args, **kwargs):
        metrics = current_metrics()
        metrics.started -= metrics.budget.deadline_seconds + 1
        return state
    monkeypatch.setattr(service.analysis, "analyze", late_analysis)
    monkeypatch.setattr(service.llm_analyst, "enrich", forbidden)
    monkeypatch.setattr(service.analysis.manager_agent, "summarize_response", forbidden)
    monkeypatch.setattr(service, "_discover_missing_data_source_links", forbidden)
    parsed = query()
    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed)
    assert response.response_type is ResponseType.ANALYSIS
    assert response.metadata["task_plan_status"] == "PARTIAL"
    assert any(item.get("deadline_exceeded") for item in response.metadata["task_execution"])


def test_comparison_marks_unavailable_risk_as_partial_and_unknown(service, monkeypatch):
    def fail(state, **kwargs):
        raise RuntimeError("risk unavailable")
    monkeypatch.setattr(service.analysis.risk_agent, "run", fail)
    parsed = ParsedQuery(raw_query="Compare offline fixtures", intent=QueryIntent.STOCK_COMPARISON,
        tickers=["FPT", "CMG"], confidence=.99)
    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed)
    assert response.response_type is ResponseType.COMPARISON
    risk = next(item for item in response.metadata["task_execution"] if item["step"] == "RISK")
    assert risk["status"] == "UNAVAILABLE"
    assert all("UNKNOWN" in row["Risk"] or "UNAVAILABLE" in row["Risk"] for row in response.tables[0].rows)


def test_predicate_recommendation_authorizes_screening_only(service, monkeypatch):
    parsed = query(QueryIntent.STOCK_RECOMMENDATION, filters={"rsi_lt": 70})
    monkeypatch.setattr(service.recommendations, "recommend", forbidden)
    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed)
    assert response.metadata["routing_plan"] == ["SCREENING"]
    assert response.response_type is ResponseType.STOCK_LIST
    assert "screen_spec" in response.metadata


def test_essential_failure_exposes_resumable_run_id(service, monkeypatch):
    def fail(*args, **kwargs):
        exc = RuntimeError("upstream data unavailable")
        exc.analysis_run_id = "offline-resumable-run"
        raise exc
    monkeypatch.setattr(service.analysis, "analyze", fail)
    parsed = query()
    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed)
    assert response.metadata["analysis_run_id"] == "offline-resumable-run"
    assert response.metadata["analysis_status"] == "FAILED"
    assert "upstream data unavailable" not in response.message


def test_full_analysis_with_multi_symbol_charts_declares_and_fetches_chart_data(service, monkeypatch):
    parsed = ParsedQuery(raw_query="Analyze FPT and chart FPT CMG", intent=QueryIntent.STOCK_ANALYSIS,
        tickers=["FPT", "CMG"], confidence=.99, secondary_intents=[QueryIntent.STOCK_CHART])
    monkeypatch.setattr(service.analysis, "fetch_data", forbidden)
    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed)
    assert response.response_type is ResponseType.ANALYSIS
    assert [chart.request.ticker for chart in response.charts[-2:]] == ["FPT", "CMG"]
    assert response.metadata["task_execution"][-1]["status"] == "COMPLETED"


def test_comparison_entity_limit_returns_chat_without_running_graph(service, monkeypatch):
    parsed = ParsedQuery(raw_query="Compare six symbols", intent=QueryIntent.STOCK_COMPARISON,
        tickers=["FPT", "CMG", "HPG", "VNM", "VCB", "TCB"], confidence=.99)
    monkeypatch.setattr(service.analysis, "analyze", forbidden)
    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed)
    assert response.metadata["routing_plan"] == ["CHAT"]
    assert "5 mã" in response.message


def test_quick_selective_plan_disables_stage_generation_and_optional_narrative(service, monkeypatch):
    options = []
    original = service.analysis.run_stage
    def run(state, name, **kwargs):
        options.append((name, kwargs))
        return original(state, name, **kwargs)
    monkeypatch.setattr(service.analysis, "run_stage", run)
    monkeypatch.setattr(service.llm_analyst, "enrich", forbidden)
    parsed = query(QueryIntent.TECHNICAL_ANALYSIS)
    response = service.handle(parsed.raw_query, ConversationState(), parsed_query=parsed, execution_mode="QUICK")
    assert ("technical", {"allow_llm_planning": False}) in options
    assert response.metadata["request_metrics"]["llm_calls"] == 0
    assert "technical_score" in response.metadata
