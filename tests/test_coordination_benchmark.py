from __future__ import annotations

import pytest

from stock_multi_agent.models.research_contracts import RequestBudget
from stock_multi_agent.services.coordination_benchmark import (
    CoordinationBenchmarkService, snapshot_sha256, synthetic_coordination_cases,
)


@pytest.fixture
def report(monkeypatch):
    monkeypatch.setenv('STOCK_AGENT_REQUEST_TELEMETRY_ENABLED', 'false')
    return CoordinationBenchmarkService().run(synthetic_coordination_cases())


def test_benchmark_compares_same_snapshots_and_budget_without_network(report):
    assert len(report['rows']) == 12
    for case_id in {row['case_id'] for row in report['rows']}:
        rows = [row for row in report['rows'] if row['case_id'] == case_id]
        assert {row['variant'] for row in rows} == set(CoordinationBenchmarkService.VARIANTS)
        assert len({row['snapshot_sha256'] for row in rows}) == 1
        assert all(row['budget'] == report['budget'] for row in rows)
        assert all(row['llm_calls'] == row['api_calls'] == 0 for row in rows)
        assert all(row['error_type'] is None for row in rows)


def test_benchmark_no_risk_baseline_is_explicitly_unassessed(report):
    rows = [row for row in report['rows'] if row['variant'] == 'no_risk_baseline']
    assert all(not row['risk_assessed'] and row['unresolved_conflicts'] is None for row in rows)
    assert all(row['missing_specialist_results'] == ['risk_result'] for row in rows)
    assert report['summary']['no_risk_baseline']['unresolved_conflicts'] is None
    assert any('No return' in item for item in report['limitations'])


def test_benchmark_preserves_constraints_in_both_production_graph_variants(report):
    for variant in ('single_pass', 'bounded_coordination'):
        assert report['summary'][variant]['constraint_violation_count'] == 0
    assert all(row['iterations'] == 0 for row in report['rows'] if row['variant'] == 'single_pass')
    assert all(row['iterations'] <= row['allowed_iterations'] for row in report['rows'])
    missing = [row for row in report['rows'] if row['case_id'] == 'missing_fundamentals']
    assert all(row['unknown_specialist_signals'] >= 1 for row in missing)
    suspension = [row for row in report['rows'] if row['case_id'] == 'verified_suspension' and row['variant'] != 'no_risk_baseline']
    assert all(row['risk_veto'] is True and row['recommendation'] == 'AVOID' for row in suspension)


def test_synthetic_fixture_hashes_and_inputs_are_stable(monkeypatch):
    monkeypatch.setenv('STOCK_AGENT_REQUEST_TELEMETRY_ENABLED', 'false')
    first, second = synthetic_coordination_cases(), synthetic_coordination_cases()
    assert [snapshot_sha256(case.market_data) for case in first] == [snapshot_sha256(case.market_data) for case in second]
    hashes = [snapshot_sha256(case.market_data) for case in first]
    CoordinationBenchmarkService().run(first)
    assert [snapshot_sha256(case.market_data) for case in first] == hashes


def test_benchmark_rejects_live_llm_budget_and_duplicate_ids():
    cases = synthetic_coordination_cases()
    with pytest.raises(ValueError, match='max_llm_calls=0'):
        CoordinationBenchmarkService().run(cases, budget=RequestBudget(max_llm_calls=1))
    with pytest.raises(ValueError, match='unique'):
        CoordinationBenchmarkService().run([cases[0], cases[0]])
