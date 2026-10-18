from stock_multi_agent.models.llm import GroundedClaim
from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.services.agent_llm import AgentLLMAdvisor, _StructuredReview
from stock_multi_agent.services.claim_grounding import audit_review, audit_structured_claim, build_evidence_pack, numeric_tokens
from stock_multi_agent.services.llm_gateway import LLMGateway


def claim(**overrides):
    return GroundedClaim.model_validate({
        'claim_vi': 'ROE của FPT đạt 20% trong năm 2025.', 'source_id': 'doc1',
        'exact_quote': 'FPT ROE đạt 20% năm 2025.', 'metric': 'roe', 'ticker': 'FPT',
        'unit': 'percent', 'period_end': '2025-12-31', 'value': 20, **overrides})


def pack():
    return build_evidence_pack({'ticker': 'FPT', 'research_context': [
        {'chunk_id': 'doc1', 'quote': 'FPT ROE đạt 20% năm 2025. Doanh thu đạt 100 tỷ đồng.',
         'tickers': ['FPT'], 'period_end': '2025-12-31'},
        {'chunk_id': 'doc2', 'quote': 'VCB ROE đạt 21% năm 2024.', 'tickers': ['VCB'], 'period_end': '2024-12-31'}]}, ['doc1', 'doc2'])


def test_exact_source_quote_and_metric_period_issuer_anchors():
    assert audit_structured_claim(claim(), pack(), 'FPT') == []
    assert 'quote_not_in_cited_source' in audit_structured_claim(claim(source_id='doc2'), pack(), 'FPT')
    assert 'metric_not_anchored_in_quote_and_claim' in audit_structured_claim(
        claim(metric='revenue', claim_vi='Doanh thu của FPT tăng 20% năm 2025.'), pack(), 'FPT')
    assert 'period_mismatch' in audit_structured_claim(claim(period_end='2024-12-31'), pack(), 'FPT')
    assert 'wrong_analysis_ticker' in audit_structured_claim(claim(ticker='VCB'), pack(), 'FPT')
    assert 'unit_not_anchored' in audit_structured_claim(claim(unit='USD'), pack(), 'FPT')


def test_quote_cannot_lend_number_to_another_metric_or_issuer():
    assert 'additional_metric_not_in_quote' in audit_structured_claim(
        claim(claim_vi='ROE và doanh thu của FPT tăng 20% năm 2025.'), pack(), 'FPT')
    assert 'unverified_issuer_symbol_in_claim' in audit_structured_claim(
        claim(claim_vi='ROE của VCB đạt 20% năm 2025.'), pack(), 'FPT')
    assert 'numbers_not_in_cited_quote' in audit_structured_claim(
        claim(claim_vi='ROE của FPT đạt 21% năm 2025.'), pack(), 'FPT')


def test_numeric_claim_cannot_omit_anchors_to_bypass_validation():
    errors = audit_structured_claim(claim(metric=None, period_end=None, ticker=None, unit=None), pack(), 'FPT')
    assert {'numeric_claim_missing_metric_anchor', 'numeric_claim_missing_period_anchor',
            'numeric_claim_missing_ticker_anchor', 'numeric_claim_missing_unit_anchor'}.issubset(errors)


def test_valid_claim_cannot_launder_reused_number_in_unrelated_review_text():
    review = _StructuredReview(assessment='Doanh thu của FPT tăng 20% năm 2025.', claims=[claim()])
    audit = audit_review(review, pack(), 'FPT')
    assert audit['status'] == 'REJECTED'
    assert 'narrative_metric_number_not_source_scoped' in audit['issues']


def test_conservative_numeric_formatting_not_scale_conversion():
    assert numeric_tokens('20.0%') == numeric_tokens('20 %')
    assert numeric_tokens('1,000') == numeric_tokens('1000.0')
    assert numeric_tokens('1,5') == numeric_tokens('1.5')
    assert numeric_tokens('0.2') != numeric_tokens('20%')
    LLMGateway.validate({'assessment': 'ROE là 20.0%.'}, {'quote': 'ROE là 20%'})


def test_metadata_and_unlinked_financials_are_not_evidence():
    facts = {'ticker': 'FPT', 'reported_metrics': {'roe': 20}, 'source_provenance': [{'source_id': 's1', 'as_of': '2025-12-31'}]}
    assert build_evidence_pack(facts, ['s1']) == []
    facts['metric_periods'] = [{'metric': 'roe', 'value': 20, 'unit': 'percent', 'source_id': 's1', 'period_end': '2025-12-31'}]
    assert len(build_evidence_pack(facts, ['s1'])) == 1


class Stub:
    def __init__(self, updates=None):
        self.updates = updates or {}

    def parse_structured(self, system, payload, output_model):
        return output_model.model_validate({
            'assessment': 'Dữ liệu có nguồn để đối chiếu. Cần cân nhắc giới hạn của dữ liệu.',
            'supporting_observations': ['Dữ liệu có nguồn rõ ràng.', 'Thông tin được đối chiếu theo kỳ.', 'Cần xem xét chất lượng doanh nghiệp.'],
            'counterarguments': ['Chưa đủ dữ liệu để khẳng định bền vững.', 'Kịch bản bất lợi cần thêm bằng chứng.'],
            'missing_information': ['Thiếu lịch sử so sánh.'], 'cited_source_ids': ['doc1'], **self.updates})


def test_legacy_review_explicitly_not_claim_verified(monkeypatch):
    monkeypatch.setenv('STOCK_AGENT_LLM_SPECIALIST_ROLES', 'fundamental')
    review = AgentLLMAdvisor(Stub(), enabled=True).review(AgentName.FUNDAMENTAL, 'Đánh giá doanh nghiệp', {'ticker': 'FPT'}, ['doc1'])
    assert review.status == 'GROUNDED'
    assert review.grounding_status == 'LEGACY_REVIEW_LEVEL_ONLY'
    assert review.grounding_audit['semantic_truth_verified'] is False


def test_claim_audit_accepts_scoped_claim_and_rejects_wrong_period(monkeypatch):
    monkeypatch.setenv('STOCK_AGENT_LLM_SPECIALIST_ROLES', 'fundamental')
    facts = {'ticker': 'FPT', 'research_context': [{'chunk_id': 'doc1', 'quote': claim().exact_quote, 'tickers': ['FPT'], 'period_end': '2025-12-31'}]}
    for period, status in [('2025-12-31', 'GROUNDED'), ('2024-12-31', 'UNAVAILABLE')]:
        review = AgentLLMAdvisor(Stub({'claims': [claim(period_end=period).model_dump()]}), enabled=True).review(
            AgentName.FUNDAMENTAL, 'Đánh giá doanh nghiệp', facts, ['doc1'])
        assert review.status == status
        assert review.grounding_status == ('CLAIM_ANCHORS_VERIFIED' if status == 'GROUNDED' else 'REJECTED')


def test_followups_are_allowlisted_and_audited(monkeypatch):
    monkeypatch.setenv('STOCK_AGENT_LLM_SPECIALIST_ROLES', 'risk')
    updates = {'follow_up_proposals': [
        {'task_kind': 'VERIFY_FINANCIAL_FACT', 'reason': 'Cần kiểm tra số liệu.', 'source_ids': ['invented']},
        {'task_kind': 'FIND_RECENT_REPORT', 'reason': 'Thiếu báo cáo cập nhật.', 'source_ids': []}]}
    review = AgentLLMAdvisor(Stub(updates), enabled=True).review(AgentName.RISK, 'Đánh giá rủi ro', {'ticker': 'FPT'}, ['doc1'])
    assert len(review.follow_up_proposals) == 1
    assert review.follow_up_proposals[0].task_kind == 'FIND_RECENT_REPORT'
    assert review.grounding_audit['proposal_issues']


def test_statement_fact_mapper_returns_bounded_unverified_proposals(monkeypatch):
    from datetime import date, datetime, timezone
    from types import SimpleNamespace
    monkeypatch.setenv('STOCK_AGENT_LLM_SPECIALIST_ROLES', 'fundamental')
    citation = SimpleNamespace(chunk_id='doc1', tickers=['FPT'], period_end=date(2025, 12, 31),
        quote='FPT báo cáo hợp nhất cả năm 2025: doanh thu 100 tỷ đồng.', publisher='FPT',
        source_url='https://fpt.com/report', published_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    proposal = {'ticker': 'FPT', 'statement': 'income_statement', 'metric': 'revenue', 'value': 100,
        'unit': 'tỷ đồng', 'currency': 'VND', 'scale': 1000000000,
        'period_label': '2025', 'period_end': '2025-12-31', 'reporting_basis': 'ANNUAL',
        'consolidation': 'CONSOLIDATED', 'source_id': 'doc1', 'evidence_quote': citation.quote}

    class Mapper:
        def __init__(self): self.payload = None
        def parse_structured(self, system, payload, output_model):
            self.payload = payload
            assert output_model.__name__ == '_StatementFacts'
            assert output_model.model_json_schema()['properties']['facts']['maxItems'] == 4
            return output_model.model_validate({'facts': [proposal, {**proposal, 'ticker': 'VCB'},
                {**proposal, 'source_id': 'invented'}]})

    provider = Mapper()
    advisor = AgentLLMAdvisor(provider, enabled=True)
    result = advisor.map_statement_facts([citation], 'FPT')
    assert len(result) == 1
    assert result[0]['value'] == 100  # no scaling/scoring or claim of validation
    assert result[0]['scale'] == 1000000000
    assert 'status' not in result[0]
    assert provider.payload['allowed_source_ids'] == ['doc1']
    advisor.roles = {'risk'}
    assert advisor.map_statement_facts([citation], 'FPT') == []


def test_statement_fact_mapper_rejects_incomplete_or_unavailable_output(monkeypatch):
    from types import SimpleNamespace
    from datetime import datetime, timezone
    monkeypatch.setenv('STOCK_AGENT_LLM_SPECIALIST_ROLES', 'fundamental')
    source = SimpleNamespace(chunk_id='doc1', tickers=['FPT'], period_end=None, quote='Doanh thu thiếu kỳ và đơn vị.',
        publisher='FPT', source_url='https://fpt.com/report', published_at=datetime.now(timezone.utc))

    class Incomplete:
        def parse_structured(self, system, payload, output_model):
            return output_model.model_validate({'facts': [{'ticker': 'FPT', 'metric': 'revenue', 'value': 100}]})

    assert AgentLLMAdvisor(Incomplete(), enabled=True).map_statement_facts([source], 'FPT') == []
