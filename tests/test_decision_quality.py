"""Offline regressions for supported decisions, not financial performance."""
from datetime import date

import pytest

from stock_multi_agent.agents.manager_agent import ManagerAgent
from stock_multi_agent.models.agent_contracts import AgentExecutionReport
from stock_multi_agent.models.analysis import (
    EvidenceItem, FundamentalAnalysis, Recommendation, RiskAnalysis, RiskLevel,
    RiskSection, Signal, TechnicalAnalysis,
)
from stock_multi_agent.models.conflicts import ConflictKind, ConflictRecord
from stock_multi_agent.models.data import (
    DataQualityIssue, DataQualityReport, FinancialMetricObservation,
    FinancialStatementObservation, QualitySeverity, QualityStatus,
)
from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.agent_evaluation import AgentEvaluationService


@pytest.fixture
def supported_state(market_data):
    evidence = EvidenceItem(label="Fixture observation", value="Supported", source_ids=["fixture:1"])
    return AgentState(
        ticker="FPT", investment_horizon="3 months", user_goal="Analyze FPT", market_data=market_data,
        technical_result=TechnicalAnalysis(
            signal=Signal.BUY, score=8, confidence=0.8, evidence=[evidence], risks=[],
            tools_used=["SMA20"], reasoning_summary="Positive trend", timeframe="3 months",
        ),
        fundamental_result=FundamentalAnalysis(
            signal=Signal.BUY, score=8, confidence=0.8, evidence=[evidence], strengths=[],
            weaknesses=[], reasoning_summary="Positive fundamentals",
        ),
        risk_result=RiskAnalysis(
            risk_level=RiskLevel.LOW, risk_score=1, confidence=0.8, veto=False,
            criticisms=[], warnings=[], reasoning_summary="Measured risk is low",
            data_uncertainty_level=RiskLevel.LOW, data_uncertainty_score=1,
            sections=[RiskSection(name="market", status="ASSESSED", coverage=1, evidence=[evidence])],
        ),
    )


def _observations(**second_changes):
    first = FinancialMetricObservation(
        metric="pe", value=10, unit="multiple", period_label="FY2025",
        period_start=date(2025, 1, 1), period_end=date(2025, 12, 31),
        reporting_basis="ANNUAL", frequency="annual", consolidation="CONSOLIDATED",
        currency="VND", scale=1, source_id="fixture:source-a",
    )
    second = first.model_copy(update={"value": 100, "source_id": "fixture:source-b", **second_changes})
    return [first, second]


def _manager():
    return ManagerAgent(routing_enabled=False, synthesis_enabled=False)


def test_same_scope_metric_conflict_blocks_buy_even_with_complete_evidence(supported_state):
    supported_state.market_data.financial_observations = _observations()
    manager = _manager()
    conflict = next(item for item in manager.assess_conflicts(supported_state) if item.kind is ConflictKind.METRIC_CONFLICT)
    update = manager.finalize(supported_state)

    assert conflict.blocks_buy
    assert update["final_decision"].evidence_completeness >= 0.75
    assert update["final_decision"].recommendation is Recommendation.WATCH
    assert update["agent_reports"][0].status == "PARTIAL"
    assert conflict.detail in update["agent_reports"][0].missing_information


@pytest.mark.parametrize("changes", [
    {"consolidation": "SEPARATE"},
    {"reporting_basis": "YTD"},
    {"frequency": "quarter"},
    {"period_start": date(2025, 10, 1)},
    {"period_end": date(2024, 12, 31)},
    {"currency": "USD"},
    {"scale": 1_000_000},
    {"unit": "percent"},
])
def test_incomparable_metric_scopes_are_not_conflicts(supported_state, changes):
    supported_state.market_data.financial_observations = _observations(**changes)
    manager = _manager()

    assert not any(item.kind is ConflictKind.METRIC_CONFLICT for item in manager.assess_conflicts(supported_state))
    assert manager.finalize(supported_state)["final_decision"].recommendation is Recommendation.BUY


def test_undated_observations_need_the_same_period_label_to_conflict(supported_state):
    observations = _observations(period_end=None, period_label="Q4 2025")
    observations[0].period_end = None
    observations[0].period_label = "Q3 2025"
    supported_state.market_data.financial_observations = observations
    assert not any(item.kind is ConflictKind.METRIC_CONFLICT for item in _manager().assess_conflicts(supported_state))


def test_metric_roundoff_and_label_aliases_do_not_create_false_conflicts(supported_state):
    supported_state.market_data.financial_observations = _observations(
        value=10.000000001, period_label="2025", metric=" PE ", unit=" MULTIPLE ", currency="vnd",
    )
    assert not any(item.kind is ConflictKind.METRIC_CONFLICT for item in _manager().assess_conflicts(supported_state))


def test_statement_conflicts_also_block_buy(supported_state):
    supported_state.market_data.financial_statement_observations = [
        FinancialStatementObservation(
            statement="income_statement", metric="revenue", value=value, unit="VND",
            period_label="2025", period_end=date(2025, 12, 31), source_id=f"fixture:{value}",
            reporting_basis="ANNUAL", consolidation="CONSOLIDATED", currency="VND", scale=1,
        ) for value in (100, 200)
    ]
    assert _manager().finalize(supported_state)["final_decision"].recommendation is Recommendation.WATCH


@pytest.mark.parametrize("fault", ["horizon", "quality_error", "quality_failed", "risk_confidence", "technical_confidence"])
def test_material_limits_block_buy(supported_state, fault):
    if fault == "horizon":
        supported_state.technical_result.timeframe = "1 week"
    elif fault == "quality_error":
        supported_state.market_data.quality_report = DataQualityReport(issues=[DataQualityIssue(
            code="NEW_VALIDATOR_ERROR", severity=QualitySeverity.ERROR, message="Invalid input",
        )])
    elif fault == "quality_failed":
        supported_state.market_data.quality_report = DataQualityReport(status=QualityStatus.FAIL)
    elif fault == "risk_confidence":
        supported_state.risk_result.confidence = 0.1
    else:
        supported_state.technical_result.confidence = 0.1
    manager = _manager()
    assert any(item.blocks_buy for item in manager.assess_conflicts(supported_state))
    assert manager.finalize(supported_state)["final_decision"].recommendation is Recommendation.WATCH


def test_resolved_metric_conflict_does_not_survive_a_corrected_input(supported_state):
    manager = _manager()
    supported_state.market_data.financial_observations = _observations()
    supported_state.active_conflicts = manager.assess_conflicts(supported_state)
    supported_state.market_data.financial_observations[1].value = 10
    assert manager.finalize(supported_state)["final_decision"].recommendation is Recommendation.BUY


def test_changed_financial_value_changes_reanalysis_fingerprint(supported_state):
    manager = _manager()
    supported_state.market_data.financial_observations = _observations()
    first = manager._evidence_fingerprint(supported_state)
    supported_state.market_data.financial_observations[1].value = 10
    assert manager._evidence_fingerprint(supported_state) != first


def test_interpretive_disagreement_is_not_a_material_data_block(supported_state, monkeypatch):
    manager = _manager()
    conflict = ConflictRecord(kind=ConflictKind.SPECIALIST_DISAGREEMENT, detail="Interpretation differs")
    assert not conflict.blocks_buy
    monkeypatch.setattr(manager, "assess_conflicts", lambda state: [conflict])
    assert manager.finalize(supported_state)["final_decision"].recommendation is Recommendation.BUY
    assert not conflict.model_copy(update={"dimension": "metric", "resolved": True}).blocks_buy
    assert not conflict.model_copy(update={"dimension": "metric", "severity": "WARNING"}).blocks_buy


def test_risk_confidence_affects_synthesis_and_risk_score_does_not_impersonate_it(supported_state):
    manager = _manager()
    supported_state.risk_result.confidence = 0.5
    lower = manager.finalize(supported_state)["final_decision"]
    supported_state.risk_result.confidence = 0.99
    higher = manager.finalize(supported_state)["final_decision"]
    supported_state.risk_result.risk_score = 3
    same_confidence = manager.finalize(supported_state)["final_decision"]
    assert lower.confidence < higher.confidence
    assert same_confidence.confidence == higher.confidence
    assert higher.calibration_status == "UNCALIBRATED_NOT_PREDICTIVE"


def _reports(state, agents):
    state.agent_reports = [AgentExecutionReport(
        agent=agent, stage="fixture", status="COMPLETED", summary="Fixture report",
        confidence=0.8, evidence_count=1, output_key=AgentEvaluationService.OUTPUT_KEYS[agent],
    ) for agent in agents]


def test_unavailable_and_unknown_results_do_not_pass_as_high_quality(supported_state):
    expected = [AgentName.DATA, AgentName.TECHNICAL, AgentName.FUNDAMENTAL, AgentName.RISK]
    _reports(supported_state, expected)
    for agent, output in [
        (AgentName.TECHNICAL, supported_state.technical_result),
        (AgentName.FUNDAMENTAL, supported_state.fundamental_result),
    ]:
        output.signal = Signal.UNKNOWN
        output.confidence = 0.1
        next(report for report in supported_state.agent_reports if report.agent is agent).status = "UNAVAILABLE"
    evaluation = AgentEvaluationService().evaluate(supported_state, expected)

    assert evaluation.execution_passed
    assert evaluation.contract_coverage == evaluation.output_coverage == 1
    assert evaluation.execution_score == 1
    assert not evaluation.passed
    assert not evaluation.analytical_passed
    assert evaluation.overall_score == evaluation.analytical_sufficiency == 0.4
    assert evaluation.unavailable_agents == [AgentName.TECHNICAL, AgentName.FUNDAMENTAL]
    assert evaluation.low_confidence_agents == [AgentName.TECHNICAL, AgentName.FUNDAMENTAL]


@pytest.mark.parametrize("fault", ["report_unavailable", "unknown_signal", "report_confidence", "output_confidence", "no_evidence", "wrong_output_key"])
def test_present_output_and_report_are_insufficient_by_themselves(supported_state, fault):
    _reports(supported_state, [AgentName.TECHNICAL])
    report = supported_state.agent_reports[0]
    if fault == "report_unavailable":
        report.status = "UNAVAILABLE"
    elif fault == "unknown_signal":
        supported_state.technical_result.signal = Signal.UNKNOWN
    elif fault == "report_confidence":
        report.confidence = 0.4
    elif fault == "output_confidence":
        supported_state.technical_result.confidence = 0.4
    elif fault == "no_evidence":
        supported_state.technical_result.evidence = []
    else:
        report.output_key = "current_plan"
    evaluation = AgentEvaluationService().evaluate(supported_state, [AgentName.TECHNICAL])
    assert evaluation.execution_passed is (fault != "wrong_output_key")
    assert not evaluation.passed
    assert evaluation.insufficient_agents == [AgentName.TECHNICAL]
    assert evaluation.overall_score == 0


def test_data_only_execution_stays_healthy_and_does_not_require_other_agents(supported_state):
    _reports(supported_state, [AgentName.DATA])
    evaluation = AgentEvaluationService().evaluate(supported_state, [AgentName.DATA])
    assert evaluation.execution_passed
    assert evaluation.analytical_passed
    assert evaluation.passed
    assert evaluation.analytical_sufficiency == 0.8


def test_empty_research_output_completes_execution_but_is_not_supported_analysis(supported_state):
    _reports(supported_state, [AgentName.RESEARCH])
    supported_state.agent_reports[0].status = "UNAVAILABLE"
    supported_state.agent_reports[0].evidence_count = 0
    evaluation = AgentEvaluationService().evaluate(supported_state, [AgentName.RESEARCH])
    assert evaluation.execution_passed
    assert not evaluation.passed
    assert evaluation.overall_score == 0


def test_no_agents_is_not_a_successful_evaluation(supported_state):
    evaluation = AgentEvaluationService().evaluate(supported_state, [])
    assert not evaluation.execution_passed
    assert not evaluation.passed
    assert evaluation.overall_score == 0


def test_final_decision_with_unresolved_metric_conflict_is_not_analytically_sufficient(supported_state):
    supported_state.market_data.financial_observations = _observations()
    supported_state.final_decision = _manager().finalize(supported_state)["final_decision"]
    expected = [AgentName.DATA, AgentName.TECHNICAL, AgentName.FUNDAMENTAL, AgentName.RISK, AgentName.MANAGER]
    _reports(supported_state, expected)
    evaluation = AgentEvaluationService().evaluate(supported_state, expected)
    assert evaluation.execution_passed
    assert not evaluation.analytical_passed
    assert AgentName.MANAGER in evaluation.insufficient_agents


def test_well_supported_risk_veto_is_a_complete_conservative_analysis(supported_state):
    supported_state.risk_result.risk_level = RiskLevel.HIGH
    supported_state.risk_result.risk_score = 9
    supported_state.risk_result.veto = True
    update = _manager().finalize(supported_state)
    assert update["final_decision"].recommendation is Recommendation.AVOID
    assert update["agent_reports"][0].status == "COMPLETED"
    assert not update["agent_reports"][0].missing_information

    supported_state.final_decision = update["final_decision"]
    expected = [AgentName.DATA, AgentName.TECHNICAL, AgentName.FUNDAMENTAL, AgentName.RISK, AgentName.MANAGER]
    _reports(supported_state, expected)
    evaluation = AgentEvaluationService().evaluate(supported_state, expected)
    assert evaluation.passed


def test_data_quality_error_cannot_be_hidden_by_a_completed_report(supported_state):
    supported_state.market_data.quality_report.issues = [DataQualityIssue(
        code="NEW_VALIDATOR_ERROR", severity=QualitySeverity.ERROR, message="Invalid input",
    )]
    _reports(supported_state, [AgentName.DATA])
    evaluation = AgentEvaluationService().evaluate(supported_state, [AgentName.DATA])
    assert evaluation.execution_passed
    assert not evaluation.analytical_passed
    assert evaluation.insufficient_agents == [AgentName.DATA]


def test_unavailable_risk_is_authoritative_even_with_legacy_confident_numbers(supported_state):
    supported_state.risk_result.assessment_status = "UNAVAILABLE"
    assert _manager().finalize(supported_state)["final_decision"].recommendation is Recommendation.WATCH
    _reports(supported_state, [AgentName.RISK])
    evaluation = AgentEvaluationService().evaluate(supported_state, [AgentName.RISK])
    assert evaluation.execution_passed
    assert not evaluation.analytical_passed
