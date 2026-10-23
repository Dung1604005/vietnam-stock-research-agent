"""Offline orchestration tests, not financial performance evaluation."""
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from conftest import conflict_market_data
from stock_multi_agent.agents.manager_agent import ManagerAgent
from stock_multi_agent.models.llm import AgentFollowUpProposal, AgentLLMReview
from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.models.research import Citation
from stock_multi_agent.models.research_contracts import RequestBudget
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.evidence_follow_up import approve_proposal, evidence_fingerprint, run_evidence_follow_up
from stock_multi_agent.services.request_telemetry import request_scope
from stock_multi_agent.services.research_financial_facts import apply_research_financial_proposals


def state_and_review(kind='FETCH_FINANCIAL_HISTORY'):
    state = AgentState(ticker='FPT', investment_horizon='3 months', user_goal='Phân tích FPT', market_data=conflict_market_data())
    review = AgentLLMReview(agent=AgentName.FUNDAMENTAL, task='Review', status='GROUNDED',
        assessment='Cần báo cáo cập nhật để kiểm tra.', follow_up_proposals=[AgentFollowUpProposal(
            task_kind=kind, reason='Cần lịch sử có kỳ để kiểm tra tăng trưởng.')])
    state.llm_reviews = [review]
    return state, review


@pytest.mark.parametrize('change,reason', [({'allow_llm_planning': False}, 'iteration_or_batch_budget'),
    ({'max_iterations': 0}, 'iteration_or_batch_budget'), ({'iteration': 2}, 'iteration_or_batch_budget')])
def test_manager_refuses_disabled_or_exhausted_loop(change, reason):
    state, review = state_and_review()
    for key, value in change.items(): setattr(state, key, value)
    assert approve_proposal(state, review, review.follow_up_proposals[0]) == (False, reason)


def test_manager_refuses_unknown_source_future_period_and_low_budget():
    state, review = state_and_review()
    proposal = review.follow_up_proposals[0].model_copy(update={'source_ids': ['unknown']})
    assert approve_proposal(state, review, proposal)[1] == 'unknown_source'
    proposal = proposal.model_copy(update={'source_ids': [], 'period_end': '2099-12-31'})
    assert approve_proposal(state, review, proposal)[1] == 'future_reporting_period'
    with request_scope(RequestBudget(max_tool_calls=2)):
        assert approve_proposal(state, review, review.follow_up_proposals[0])[1] == 'remaining_request_budget'


def graph_fixture(new_data, *, fail=False):
    class Data:
        name = AgentName.DATA
        def run(self, state, refresh=False):
            if fail: raise RuntimeError('provider failed')
            return {'market_data': new_data}
    return SimpleNamespace(manager_agent=ManagerAgent(), data_agent=Data(),
        technical_agent=SimpleNamespace(name=AgentName.TECHNICAL, run=lambda *a, **k: {'completed_tasks': ['technical_recomputed']}),
        fundamental_agent=SimpleNamespace(name=AgentName.FUNDAMENTAL, run=lambda *a, **k: {'completed_tasks': ['fundamental_recomputed']}),
        risk_agent=SimpleNamespace(name=AgentName.RISK, run=lambda *a, **k: {'completed_tasks': ['risk_recomputed']}),
        _forecast=lambda *a: {}, _specialist_reviews=lambda *a: {})


def test_new_evidence_recomputes_all_affected_specialists_once():
    state, review = state_and_review()
    updated = state.market_data.model_copy(deep=True)
    updated.financials.pe = 10
    graph = graph_fixture(updated)
    result = run_evidence_follow_up(state, graph)
    assert result['market_data'].financials.pe == 10 and state.market_data.financials.pe != 10
    assert result['evidence_follow_up_outcomes'][-1]['status'] == 'NEW_EVIDENCE'
    assert result['completed_tasks'] == ['technical_recomputed', 'fundamental_recomputed', 'risk_recomputed']
    state.evidence_follow_up_outcomes = result['evidence_follow_up_outcomes']
    assert run_evidence_follow_up(state, graph) == {}


def test_no_evidence_change_preserves_original_snapshot_and_does_not_recompute():
    state, _ = state_and_review()
    updated = state.market_data.model_copy(deep=True)
    updated.fetched_at = datetime.now(timezone.utc)
    assert evidence_fingerprint(state) == evidence_fingerprint(state.model_copy(update={'market_data': updated}))
    result = run_evidence_follow_up(state, graph_fixture(updated))
    assert result['market_data'].fetched_at == state.market_data.fetched_at
    assert result['completed_tasks'] == []
    assert result['evidence_follow_up_outcomes'][-1]['status'] == 'NO_NEW_EVIDENCE'


def test_metadata_changes_are_evidence_changes_and_failure_is_atomic():
    state, _ = state_and_review()
    updated = state.market_data.model_copy(deep=True)
    updated.price_unit = 'VND'
    assert evidence_fingerprint(state) != evidence_fingerprint(state.model_copy(update={'market_data': updated}))
    result = run_evidence_follow_up(state, graph_fixture(updated, fail=True))
    assert 'market_data' not in result
    assert result['evidence_follow_up_outcomes'][-1]['status'] == 'FAILED'


def test_exact_research_fact_enters_only_runtime_snapshot_with_reduced_weight():
    data = conflict_market_data()
    quote = 'Doanh thu: 120 tỷ đồng cả năm 2025 hợp nhất.'
    citation = Citation(document_id='real-fixture-doc', chunk_id='real-fixture-chunk', title='Báo cáo',
        source_url='https://issuer.invalid/report', publisher='Offline fixture', published_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        retrieved_at=data.fetched_at, content_sha256='a' * 64, quote=quote, tickers=['FPT'], quality_tier='T1')
    proposal = dict(ticker='FPT', statement='income_statement', metric='revenue', value=120,
        unit='tỷ đồng', currency='VND', scale=1e9, reporting_basis='ANNUAL', consolidation='CONSOLIDATED',
        period_label='2025', period_end='2025-12-31', source_id=citation.chunk_id, evidence_quote=quote)
    updated, audit = apply_research_financial_proposals(data, [citation], [proposal], 'FPT')
    assert audit[0]['status'] == 'ACCEPTED_EXTRACTION'
    assert updated.financial_statement_observations[-1].source_id == citation.chunk_id
    assert not data.financial_statement_observations  # no mutation of provider/cache
    assert updated.quality_report.issues[-1].code == 'RESEARCH_EXTRACTED_FINANCIAL_FACTS'
    _, audit = apply_research_financial_proposals(data, [citation.model_copy(update={'quality_tier': 'T4'})], [proposal], 'FPT')
    assert audit[0]['status'] == 'REJECTED'


def test_research_followup_has_different_queries_and_web_objective():
    from stock_multi_agent.agents.research_agent import ResearchAgent
    from stock_multi_agent.models.research import EvidenceBundle
    calls = []
    class Store:
        def search(self, query, **kwargs):
            calls.append(query)
            return EvidenceBundle(query=query, ticker='FPT', citations=[], retrieval_note='Offline fixture')
    class Web:
        def research(self, query, **kwargs):
            calls.append(query)
            return SimpleNamespace(external_citations=[])
    state, _ = state_and_review()
    ResearchAgent(Store(), web_research=Web()).run(state, expanded=True, follow_up_task='CHECK_RISK_EVENT')
    assert 'kiện tụng' in calls[0] and 'kiện tụng' in calls[-1]


def test_financial_chart_controls_and_context_follow_up():
    from stock_multi_agent.chat.query_parser import QueryParser
    from stock_multi_agent.models.chat import ConversationState, QueryIntent
    from stock_multi_agent.services.chart_service import ChartService
    query = QueryParser(use_configured_provider=False).parse('Vẽ biểu đồ doanh thu theo quý trong 8 quý',
        ConversationState(active_tickers=['FPT']))
    assert query.tickers == ['FPT'] and query.intent == QueryIntent.STOCK_CHART
    requests = ChartService().requests_from_query(query)
    assert requests[0].period == '8Q' and 'QUARTER' in requests[0].indicators


def test_full_graph_executes_approved_follow_up_and_finalizes_fresh_data(monkeypatch):
    from stock_multi_agent.services.analysis_service import AnalysisService
    from stock_multi_agent.services.agent_llm import AgentLLMAdvisor
    from stock_multi_agent.services.llm_gateway import LLMGateway
    monkeypatch.setenv('STOCK_AGENT_LLM_SPECIALIST_ROLES', 'fundamental,risk')
    class Provider:
        calls = 0
        def get_market_data(self, ticker, include_financials=True):
            data = conflict_market_data(ticker)
            if ticker == 'FPT':
                self.calls += 1
                data.financials.pe = 10 if self.calls > 1 else 28
            return data
    class Model:
        def parse_structured(self, system, payload, schema):
            return schema.model_validate({'assessment': 'Cần đối chiếu lịch sử báo cáo để kiểm tra.',
                'supporting_observations': ['Nguồn đã được cung cấp.', 'Giá trị cần kiểm tra theo kỳ.', 'Đánh giá có giới hạn dữ liệu.'],
                'counterarguments': ['Chưa biết chất lượng kỳ báo cáo.', 'Dữ liệu có thể chưa cập nhật.'],
                'missing_information': ['Cần lịch sử tài chính.'], 'cited_source_ids': payload['allowed_source_ids'][:1],
                'follow_up_proposals': [{'task_kind': 'FETCH_FINANCIAL_HISTORY', 'reason': 'Cần lịch sử tài chính để đối chiếu.', 'source_ids': []}]})
    provider = Provider()
    analysis = AnalysisService(provider=provider, llm_advisor=AgentLLMAdvisor(provider=LLMGateway(Model()), enabled=True))
    with request_scope(RequestBudget(max_llm_calls=6, max_output_tokens=12000)):
        state = analysis.analyze('FPT', max_iterations=2)
    assert state.evidence_follow_up_outcomes[-1]['status'] == 'NEW_EVIDENCE'
    assert state.market_data.financials.pe == 10 and state.final_decision is not None
    assert state.iteration <= 2
