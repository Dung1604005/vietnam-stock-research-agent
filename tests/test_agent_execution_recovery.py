"""Offline failure injection, contract authority and durable resume regressions."""
from datetime import datetime, timezone
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import pytest

from conftest import InMemoryProvider
from stock_multi_agent.chat.response import ResponseComposer
from stock_multi_agent.models.analysis import Recommendation, RiskLevel, Signal
from stock_multi_agent.models.execution_profile import ExecutionProfile
from stock_multi_agent.models.forecast import ForecastStatus, PriceForecast
from stock_multi_agent.models.messages import AgentMessage, AgentName, MessageType
from stock_multi_agent.models.llm import AgentLLMReview
from stock_multi_agent.models.research import Citation, EvidenceBundle
from stock_multi_agent.models.state import TaskExecution, TraceEvent
from stock_multi_agent.services.agent_execution import AgentContractError, execute_agent, input_fingerprint, validate_update
from stock_multi_agent.services.agent_llm import AgentLLMAdvisor
from stock_multi_agent.services.agent_evaluation import AgentEvaluationService
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.execution_profiles import current_profile, execution_scope
from stock_multi_agent.services.request_telemetry import current_metrics, request_scope
from stock_multi_agent.models.research_contracts import RequestBudget


class _NoForecast:
    def forecast_next_session(self, data):
        return PriceForecast(ticker=data.ticker, as_of=data.as_of, current_price=data.bars[-1].close,
            status=ForecastStatus.UNAVAILABLE, method="Offline fixture", model_version="fixture-v1", confidence=0)


class _Research:
    def search(self, query, **kwargs):
        now = datetime.now(timezone.utc)
        return EvidenceBundle(query=query, citations=[Citation(
            document_id="fixture-document", chunk_id="fixture-chunk", title="Fixture research",
            source_url="https://example.invalid/research", publisher="Offline fixture",
            published_at=now, retrieved_at=now, content_sha256="b" * 64,
            quote="Synthetic observations used only for orchestration tests.", retrieval_score=0.8,
        )], retrieval_note="Offline fixture")


@pytest.fixture
def recovery_factory(monkeypatch, isolated_runtime_root, market_data):
    runtime = isolated_runtime_root / "recovery" / uuid4().hex
    monkeypatch.setenv("STOCK_AGENT_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("STOCK_AGENT_CHECKPOINTS_ENABLED", "true")
    monkeypatch.setenv("STOCK_AGENT_REQUEST_TELEMETRY_ENABLED", "false")
    monkeypatch.setenv("STOCK_AGENT_BENCHMARK_ENABLED", "false")
    monkeypatch.setenv("STOCK_AGENT_LLM_MANAGER_SYNTHESIS_ENABLED", "false")

    def create():
        return AnalysisService(provider=InMemoryProvider(market_data), research_store=_Research(),
            price_forecast_service=_NoForecast(), llm_advisor=AgentLLMAdvisor(enabled=False),
            manager_routing_enabled=False)
    return create


def _initial(service):
    return service.create_state("FPT", max_iterations=0, allow_llm_planning=False)


def _fail(*args, **kwargs):
    raise RuntimeError("Offline injected stage failure")


@pytest.mark.parametrize("role", ["technical", "fundamental"])
def test_specialist_failure_preserves_sibling_and_returns_partial_decision(recovery_factory, monkeypatch, role):
    service = recovery_factory()
    monkeypatch.setattr(getattr(service, f"{role}_agent"), "run", _fail)
    result = service.graph.invoke(_initial(service))
    failed = getattr(result, f"{role}_result")
    sibling = result.fundamental_result if role == "technical" else result.technical_result

    assert failed.signal is Signal.UNKNOWN
    assert failed.confidence == 0
    assert sibling.signal is not Signal.UNKNOWN
    assert result.market_data is not None
    assert result.final_decision.recommendation is not Recommendation.BUY
    assert any(report.agent.value == role and report.status == "UNAVAILABLE" for report in result.agent_reports)
    task = next(item for item in result.task_executions if item.node == role)
    assert task.status == "UNAVAILABLE"
    assert task.error_type == "RuntimeError"
    response = ResponseComposer.full_analysis(result)
    assert response.metadata["analysis_status"] == "PARTIAL"
    assert not response.metadata["agent_evaluation"]["analytical_passed"]


def test_risk_failure_is_unknown_coverage_and_never_a_low_risk_claim(recovery_factory, monkeypatch):
    service = recovery_factory()
    monkeypatch.setattr(service.risk_agent, "run", _fail)
    result = service.graph.invoke(_initial(service))
    risk = result.risk_result
    assert risk.assessment_status == "UNAVAILABLE"
    assert risk.confidence == 0
    assert risk.data_uncertainty_level is RiskLevel.HIGH
    assert risk.sections and all(item.status == "UNKNOWN" for item in risk.sections)
    assert not risk.veto
    assert result.final_decision.recommendation is not Recommendation.BUY

    for compose in (ResponseComposer.risk, ResponseComposer.full_analysis):
        response = compose(result)
        assert response.metadata["risk_assessment_status"] == "UNAVAILABLE"
        assert response.metadata["risk_level"] == "UNKNOWN"
        assert response.metadata["risk_score"] is None
        assert response.metadata["issuer_risk_level"] == "UNKNOWN"
        assert response.metadata["issuer_risk_score"] is None
        risk_line = next(line for line in response.message.splitlines() if line.startswith("**Rủi ro đo lường:**"))
        assert "CHƯA ĐÁNH GIÁ" in risk_line
        assert "0.0/10" not in risk_line


def test_late_risk_result_is_not_published_as_a_completed_buy(recovery_factory, monkeypatch):
    service = recovery_factory()
    fundamental = service.fundamental_agent.run
    def bullish(*args, **kwargs):
        update = fundamental(*args, **kwargs)
        update['fundamental_result'].signal = Signal.BUY
        update['fundamental_result'].confidence = 0.8
        return update
    risk = service.risk_agent.run
    def late(*args, **kwargs):
        update = risk(*args, **kwargs)
        result = update['risk_result']
        result.risk_level, result.risk_score, result.veto = RiskLevel.LOW, 1, False
        result.data_uncertainty_level, result.data_uncertainty_score = RiskLevel.LOW, 1
        result.confidence = 0.9
        # Model a computation that finishes only after the enclosing request's
        # deadline. The completed numerical output must not be published.
        current_metrics().started -= current_metrics().budget.deadline_seconds + 1
        return update
    monkeypatch.setattr(service.fundamental_agent, 'run', bullish)
    monkeypatch.setattr(service.risk_agent, 'run', late)
    result = service.graph.invoke(_initial(service))
    assert result.technical_result.signal is result.fundamental_result.signal is Signal.BUY
    assert result.risk_result.assessment_status == 'UNAVAILABLE'
    assert result.final_decision.recommendation is Recommendation.WATCH
    assert next(item for item in result.task_executions if item.node == 'risk').status == 'UNAVAILABLE'


def test_unavailable_risk_score_card_renders_without_formatting_none(recovery_factory, monkeypatch):
    service = recovery_factory()
    monkeypatch.setattr(service.risk_agent, "run", _fail)
    result = service.graph.invoke(_initial(service))
    response = ResponseComposer.full_analysis(result)
    from stock_multi_agent.ui import components
    cards = []
    monkeypatch.setattr(components.st, "columns", lambda count: [
        SimpleNamespace(markdown=lambda text, **kwargs: cards.append(text)) for _ in range(count)
    ])
    components.render_analysis_card(response)
    assert any("CHƯA ĐÁNH GIÁ" in item or "UNKNOWN" in item for item in cards)
    assert not any('score-label">Risk<' in item and "0.0/10" in item for item in cards)


def test_failed_research_refresh_retains_verified_citations(recovery_factory, monkeypatch):
    service = recovery_factory()
    state = service.run_stage(_initial(service), "data")
    state = service.run_stage(state, "research")
    citations = [item.model_dump() for item in state.research_evidence]
    monkeypatch.setattr(service.research_agent, "run", _fail)
    resumed = service.run_stage(state, "research", expanded=True)
    assert [item.model_dump() for item in resumed.research_evidence] == citations
    assert resumed.agent_reports[-1].status == "UNAVAILABLE"
    assert resumed.task_executions[-1].status == "UNAVAILABLE"


def test_optional_forecast_failure_does_not_erase_successful_data_contract(recovery_factory, monkeypatch):
    service = recovery_factory()
    monkeypatch.setattr(service.price_forecast_service, "forecast_next_session", _fail)
    result = service.graph.invoke(_initial(service))
    assert result.market_data is not None
    assert result.price_forecast is None
    assert any(item.output_key == "price_forecast" and item.status == "UNAVAILABLE" for item in result.agent_reports)
    evaluation = AgentEvaluationService().evaluate(result, [AgentName.DATA])
    assert evaluation.execution_passed
    assert evaluation.analytical_passed


@pytest.mark.parametrize("violation", ["final_decision", "sender", "report", "run_id", "trace", "review"])
def test_nonmanager_cannot_cross_role_authority(recovery_factory, violation):
    service = recovery_factory()
    state = _initial(service)
    if violation == "final_decision":
        update = {"final_decision": None}
    elif violation == "sender":
        update = {"messages": [AgentMessage(sender=AgentName.MANAGER, receiver=AgentName.USER,
            message_type=MessageType.FINAL_DECISION, task="Impersonation") ]}
    elif violation == "report":
        update = {"agent_reports": [service.manager_agent.report(stage="fake", status="COMPLETED",
            summary="Impersonation", confidence=1, output_key="final_decision")]}
    elif violation == "trace":
        update = {"trace": [TraceEvent(agent=AgentName.MANAGER, phase="ACT", action="FAKE", detail="Impersonation")]}
    elif violation == "review":
        update = {"llm_reviews": [AgentLLMReview(agent=AgentName.RISK, task="Fake review",
            status="GROUNDED", assessment="Impersonation")]}
    else:
        update = {"run_id": str(uuid4())}
    with pytest.raises(AgentContractError):
        validate_update(state, AgentName.TECHNICAL, update)


def test_contract_violation_produces_unknown_instead_of_leaking_an_update(recovery_factory):
    service = recovery_factory()
    state = _initial(service)
    update = execute_agent(state, service.technical_agent, "technical", lambda value: {"final_decision": None})
    assert "final_decision" not in update
    assert update["technical_result"].signal is Signal.UNKNOWN
    assert update["task_executions"][0].error_type == "AgentContractError"


def test_in_place_state_mutation_does_not_escape_the_stage_authority(recovery_factory):
    service = recovery_factory()
    state = service.run_stage(_initial(service), "data")
    original = state.model_dump()
    def mutate(value):
        value.user_goal = "Tampered goal"
        value.market_data.financials.pe = -999
        return {}
    execute_agent(state, service.technical_agent, "technical", mutate)
    assert state.model_dump() == original


def test_messages_correlate_requests_results_and_execution_records(recovery_factory):
    service = recovery_factory()
    result = service.graph.invoke(_initial(service))
    executions = {item.task_id: item for item in result.task_executions}
    requests = {message.id: message for message in result.messages
        if message.message_type in {MessageType.TASK, MessageType.REQUEST_REANALYSIS}}
    delegated_ids = [message.task_id for message in requests.values()]
    assert len(delegated_ids) == len(set(delegated_ids))
    for message in result.messages:
        assert message.run_id == result.run_id
        assert message.input_fingerprint and len(message.input_fingerprint) == 64
        if message.message_type in {MessageType.RESULT, MessageType.CRITIQUE}:
            assert message.task_id in executions
            assert message.input_fingerprint == executions[message.task_id].input_fingerprint
            if message.sender in {AgentName.DATA, AgentName.TECHNICAL, AgentName.FUNDAMENTAL, AgentName.RISK}:
                assert message.reply_to in requests
                request = requests[message.reply_to]
                assert request.receiver == message.sender
                assert request.task_id == message.task_id
    assert all(isinstance(item, TaskExecution) for item in result.task_executions)


def test_input_fingerprint_tracks_evidence_but_ignores_retrieval_clock(recovery_factory):
    service = recovery_factory()
    state = service.run_stage(_initial(service), "data")
    before = input_fingerprint(state)
    state.market_data.fetched_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert input_fingerprint(state) == before
    state.market_data.financials.pe = 12
    assert input_fingerprint(state) != before


def _interrupt_parallel(service, monkeypatch):
    ready = Event()
    original_technical = service.technical_agent.run
    completed = []
    def technical(*args, **kwargs):
        result = original_technical(*args, **kwargs)
        completed.append("technical")
        ready.set()
        return result
    def interrupt(*args, **kwargs):
        assert ready.wait(5), "Technical sibling did not finish"
        raise KeyboardInterrupt("Simulated process interruption after sibling completed")
    monkeypatch.setattr(service.technical_agent, "run", technical)
    monkeypatch.setattr(service.fundamental_agent, "run", interrupt)
    initial = _initial(service)
    with pytest.raises(KeyboardInterrupt):
        service.graph.invoke(initial)
    assert completed == ["technical"]
    return initial


def test_sqlite_resume_after_service_recreation_does_not_replay_completed_sibling(recovery_factory, monkeypatch):
    first = recovery_factory()
    initial = _interrupt_parallel(first, monkeypatch)
    checkpointer_path = first.graph.checkpointer.path
    second = recovery_factory()
    assert second.graph.checkpointer.path == checkpointer_path
    assert second.graph.checkpointer is not first.graph.checkpointer
    monkeypatch.setattr(second.data_agent, "run", lambda *args, **kwargs: pytest.fail("Data was replayed"))
    monkeypatch.setattr(second.technical_agent, "run", lambda *args, **kwargs: pytest.fail("Successful sibling was replayed"))

    result = second.resume_analysis(initial.run_id)
    assert result.final_decision is not None
    assert result.technical_result.signal is not Signal.UNKNOWN
    assert sum(item.node == "technical" for item in result.task_executions) == 1
    assert sum(item.node == "fundamental" for item in result.task_executions) == 1
    assert all(isinstance(item, AgentMessage) for item in result.messages)
    assert all(isinstance(item, TaskExecution) for item in result.task_executions)

    again = recovery_factory().resume_analysis(initial.run_id)
    assert again.model_dump() == result.model_dump()
    with pytest.raises(ValueError, match="already exists"):
        second.graph.invoke(initial)


def test_resume_restores_offline_profile_even_if_caller_is_standard(recovery_factory, monkeypatch):
    first = recovery_factory()
    with execution_scope("OFFLINE"):
        initial = _interrupt_parallel(first, monkeypatch)
    assert initial.execution_mode == "OFFLINE"
    second = recovery_factory()
    original = second.fundamental_agent.run
    seen = []
    def observe(*args, **kwargs):
        seen.append(current_profile())
        return original(*args, **kwargs)
    monkeypatch.setattr(second.fundamental_agent, "run", observe)
    result = second.resume_analysis(initial.run_id)
    assert seen and all(item.mode.value == "OFFLINE" and not item.external_allowed for item in seen)
    assert result.execution_mode == "OFFLINE"
    assert current_profile().mode.value == "STANDARD"


def test_resume_does_not_broaden_a_custom_execution_profile(recovery_factory, monkeypatch):
    first = recovery_factory()
    selected = ExecutionProfile(mode="STANDARD", external_allowed=False, max_llm_calls=0,
        max_tool_calls=80, deadline_seconds=20, description="Restricted local analysis")
    with execution_scope(selected):
        initial = _interrupt_parallel(first, monkeypatch)
    second = recovery_factory()
    original = second.fundamental_agent.run
    seen = []
    def observe(*args, **kwargs):
        seen.append(current_profile())
        return original(*args, **kwargs)
    monkeypatch.setattr(second.fundamental_agent, "run", observe)
    second.resume_analysis(initial.run_id)
    assert seen and all(item == selected for item in seen)


def test_unknown_and_incompatible_checkpoints_are_rejected(recovery_factory, monkeypatch):
    service = recovery_factory()
    with pytest.raises(ValueError, match="Unknown"):
        service.resume_analysis(str(uuid4()))
    initial = _interrupt_parallel(service, monkeypatch)
    service.graph.compiled.update_state({"configurable": {"thread_id": initial.run_id}},
        {"workflow_version": "future-incompatible-version"})
    with pytest.raises(ValueError, match="version"):
        recovery_factory().resume_analysis(initial.run_id)


def test_initial_data_failure_preserves_empty_checkpoint_for_deliberate_resume(recovery_factory, monkeypatch):
    first = recovery_factory()
    initial = _initial(first)
    monkeypatch.setattr(first.data_agent, "run", _fail)
    with pytest.raises(RuntimeError) as failure:
        first.graph.invoke(initial)
    assert failure.value.analysis_run_id == initial.run_id
    saved = first.graph.compiled.get_state({"configurable": {"thread_id": initial.run_id}})
    assert saved.values.get("market_data") is None
    assert saved.values.get("final_decision") is None
    assert saved.next == ("data",)

    second = recovery_factory()
    monkeypatch.setattr(second.manager_agent, "plan", lambda *args: pytest.fail("Completed plan was replayed"))
    result = second.resume_analysis(initial.run_id)
    assert result.market_data is not None
    assert result.final_decision is not None
    assert sum(item.node == "manager_plan" for item in result.task_executions) == 1


def test_public_analyze_failure_exposes_a_resumable_identity(recovery_factory, monkeypatch):
    service = recovery_factory()
    monkeypatch.setattr(service.data_agent, "run", _fail)
    with pytest.raises(RuntimeError) as failure:
        service.analyze("FPT", max_iterations=0, allow_llm_planning=False)
    run_id = failure.value.analysis_run_id
    assert recovery_factory().resume_analysis(run_id).final_decision is not None


def test_run_guard_rejects_concurrent_execution_across_service_instances(recovery_factory):
    first, second = recovery_factory(), recovery_factory()
    initial = _initial(first)
    with first.graph.checkpointer.run_guard(initial.run_id):
        with pytest.raises(RuntimeError, match="already executing"):
            second.graph.invoke(initial)
    assert second.graph.invoke(initial).final_decision is not None


def test_saved_budget_is_not_broadened_when_resumed_under_a_larger_request(recovery_factory, monkeypatch):
    first = recovery_factory()
    with execution_scope(ExecutionProfile(max_tool_calls=10, max_llm_calls=0, deadline_seconds=20)):
        initial = _interrupt_parallel(first, monkeypatch)
    second = recovery_factory()
    original = second.fundamental_agent.run
    seen = []
    def observe(*args, **kwargs):
        seen.append(current_metrics().budget)
        return original(*args, **kwargs)
    monkeypatch.setattr(second.fundamental_agent, "run", observe)
    with request_scope(RequestBudget(max_tool_calls=200, max_llm_calls=8, deadline_seconds=180)):
        second.resume_analysis(initial.run_id)
    assert seen and all(item.max_tool_calls <= 10 and item.max_llm_calls == 0 and item.deadline_seconds <= 20 for item in seen)
