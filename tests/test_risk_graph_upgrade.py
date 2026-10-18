"""Authored deterministic fixtures, not independent financial accuracy evidence."""
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

import pytest

from stock_multi_agent.agents.fundamental_agent import FundamentalAgent
from stock_multi_agent.agents.risk_agent import RiskAgent
from stock_multi_agent.agents.technical_agent import TechnicalAgent
from stock_multi_agent.models.analysis import FinancialMetrics
from stock_multi_agent.models.data import DataKind, FinancialMetricObservation, Instrument, IssuerEvent, SourceReference
from stock_multi_agent.models.graph_v2 import GraphEntity, GraphEvidence, GraphTriple
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.graph_v2 import InvestmentGraph
from stock_multi_agent.tools.financial_data import FinancialToolkit
from stock_multi_agent.tools.technical_indicators import TechnicalIndicatorToolkit


def risk(data):
    state = AgentState(ticker=data.ticker, investment_horizon='3 months', user_goal='Analyze', market_data=data)
    state.technical_result = TechnicalAgent(TechnicalIndicatorToolkit()).run(state)['technical_result']
    state.fundamental_result = FundamentalAgent(FinancialToolkit()).run(state)['fundamental_result']
    return RiskAgent(TechnicalIndicatorToolkit()).run(state)['risk_result']


def test_missing_financial_inputs_are_unknown_not_assessed(market_data):
    data = market_data.model_copy(update={'financials': FinancialMetrics()})
    result = risk(data)
    sections = {item.name: item for item in result.sections}
    assert sections['business_and_accounting'].status == 'UNKNOWN'
    assert sections['business_and_accounting'].coverage == 0
    assert sections['business_and_accounting'].evidence == []
    assert result.feature_contributions['leverage'] == 0
    assert sections['business_and_accounting'].missing_information


@pytest.mark.parametrize(('sector', 'template', 'missing'), [
    ('Ngân hàng', 'BANK', 'npl_ratio'),
    ('Chứng khoán', 'SECURITIES', 'margin_lending'),
    ('Bảo hiểm', 'INSURANCE', 'solvency_ratio'),
    ('Bất động sản', 'REAL_ESTATE', 'project_legal_status'),
])
def test_sector_checklists_do_not_invent_unavailable_metrics(market_data, sector, template, missing):
    data = market_data.model_copy(update={
        'instrument': Instrument(ticker='FPT', sector=sector),
        'financials': market_data.financials.model_copy(update={'debt_to_equity': 8}),
    })
    result = risk(data)
    section = next(item for item in result.sections if item.name == 'business_and_accounting')
    assert result.sector_template == template
    assert section.status == 'PARTIAL'
    assert missing in section.missing_information
    if template in {'BANK', 'SECURITIES', 'INSURANCE'}:
        assert result.feature_contributions['leverage'] == 0


def test_turnover_does_not_guess_price_scale(market_data):
    unknown = risk(market_data.model_copy(update={'price_unit': 'UNKNOWN'}))
    section = next(item for item in unknown.sections if item.name == 'liquidity')
    assert section.status == 'PARTIAL'
    assert unknown.feature_contributions['liquidity'] == 0
    assert 'unverified scale' in section.evidence[0].value
    vnd = risk(market_data.model_copy(update={'price_unit': 'VND'}))
    thousand = risk(market_data.model_copy(update={'price_unit': 'THOUSAND_VND'}))
    assert vnd.feature_contributions['liquidity'] == 1
    assert thousand.feature_contributions['liquidity'] == 0


def test_risk_factor_contributions_reconstruct_bounded_score(market_data):
    result = risk(market_data)
    assert sum(result.feature_contributions.values()) == pytest.approx(result.risk_score)
    assert result.coverage['business_and_accounting'] < 1
    assert result.data_uncertainty_score not in result.feature_contributions


def test_unsourced_or_future_published_event_cannot_veto(market_data):
    event = IssuerEvent(event_code='SUSP', title='Suspension', event_date=market_data.as_of,
                        published_date=market_data.as_of + timedelta(days=1), source_id='events')
    data = market_data.model_copy(update={'issuer_events': [event]})
    assert risk(data).feature_contributions['issuer_event'] == 0
    ref = SourceReference(provider='fixture', source_id='events', data_kind=DataKind.ISSUER_EVENTS,
                          fetched_at=market_data.fetched_at, as_of=market_data.as_of)
    data = data.model_copy(update={'source_refs': [*data.source_refs, ref]})
    assert risk(data).feature_contributions['issuer_event'] == 0


def test_specialized_metric_keeps_unit_period_and_does_not_change_score(market_data):
    ref = SourceReference(provider='fixture', source_id='ratios', data_kind=DataKind.FINANCIAL_RATIOS,
                          fetched_at=market_data.fetched_at, as_of=market_data.as_of)
    data = market_data.model_copy(update={
        'instrument': Instrument(ticker='FPT', sector='Banking'),
        'source_refs': [*market_data.source_refs, ref],
        'financial_observations': [FinancialMetricObservation(metric='npl_ratio', value=2, unit='percent',
            period_label='Annual', period_end=date(market_data.as_of.year - 1, 12, 31), source_id='ratios')],
    })
    result = risk(data)
    section = next(item for item in result.sections if item.name == 'business_and_accounting')
    assert 'npl_ratio' not in section.missing_information
    evidence = next(item for item in section.evidence if item.label == 'npl_ratio')
    assert 'percent' in evidence.value and evidence.source_ids == ['ratios']
    assert 'npl_ratio' not in result.feature_contributions


@pytest.fixture
def graph(isolated_runtime_root):
    return InvestmentGraph(isolated_runtime_root / f'graph-risks-{uuid4().hex}.sqlite3')


def test_reviewed_claim_has_explicit_evidence_path_and_no_causal_relation(graph):
    now = datetime(2026, 10, 17, tzinfo=timezone.utc)
    evidence = GraphEvidence(evidence_id='report:1', document_id='doc:1', source_url='https://example.invalid/report',
        quote='FPT có doanh thu tăng trong kỳ. Đây là dữ liệu minh họa.', retrieved_at=now, published_at=now, layer='research')
    graph.put_evidence(evidence)
    subject = GraphEntity(entity_id='ticker:FPT', kind='Ticker', label='FPT')
    with pytest.raises(ValueError, match='review'):
        graph.record_reviewed_claim(subject, 'FPT có doanh thu tăng trong kỳ.', 'report:1')
    with pytest.raises(ValueError, match='absent'):
        graph.record_reviewed_claim(subject, 'FPT chắc chắn tăng giá.', 'report:1', reviewed=True)
    claim_id = graph.record_reviewed_claim(subject, 'FPT có doanh thu tăng trong kỳ.', 'report:1', reviewed=True)
    before = graph.stats()
    graph.record_reviewed_claim(subject, 'FPT có doanh thu tăng trong kỳ.', 'report:1', reviewed=True)
    assert graph.stats() == before
    rows = graph.query('FPT', as_of=now)
    assert {row['triple']['relation'] for row in rows} == {'ABOUT', 'SUPPORTED_BY'}
    assert all(row['triple']['subject']['entity_id'] == claim_id for row in rows)
    assert all(row['evidence']['source_url'] == 'https://example.invalid/report' for row in rows)
    assert graph.query('FPT', as_of=now - timedelta(days=1)) == []


def test_risk_relation_cannot_be_created_by_metadata_bypass(graph):
    now = datetime(2026, 10, 17, tzinfo=timezone.utc)
    graph.put_evidence(GraphEvidence(evidence_id='risk:1', document_id='doc:1', source_url='https://example.invalid/report',
        quote='FPT ghi nhận rủi ro thanh khoản.', retrieved_at=now, layer='research'))
    edge = GraphTriple(subject=GraphEntity(entity_id='company:FPT', kind='Company', label='FPT'),
        relation='EXPOSED_TO', object=GraphEntity(entity_id='risk:liquidity', kind='RiskFactor', label='rủi ro thanh khoản'),
        evidence_id='risk:1', valid_from=now, confidence=.8, origin='deterministic')
    with pytest.raises(ValueError, match='review'):
        graph.put(edge, trusted_metadata=True)
    graph.put(edge, reviewed=True)
    assert graph.stats()['edges'] == 1
    fabricated = edge.model_copy(update={'object': GraphEntity(entity_id='risk:fraud', kind='RiskFactor', label='gian lận')})
    with pytest.raises(ValueError, match='absent'):
        graph.put(fabricated, reviewed=True, trusted_metadata=True)


def test_market_graph_links_sector_event_and_period_with_sources(graph, market_data):
    refs = [SourceReference(provider='fixture', source_id=kind.value, data_kind=kind,
            source_url='https://example.invalid/' + kind.value, fetched_at=market_data.fetched_at,
            as_of=market_data.as_of) for kind in [DataKind.COMPANY_PROFILE, DataKind.ISSUER_EVENTS, DataKind.FINANCIAL_RATIOS]]
    data = market_data.model_copy(update={
        'instrument': Instrument(ticker='FPT', name='Công ty FPT', sector='Công nghệ', exchange='HOSE'),
        'source_refs': refs,
        'financial_observations': [FinancialMetricObservation(metric='roe', value=.2, unit='ratio',
            period_label='Annual', period_end=date(market_data.as_of.year - 1, 12, 31), source_id=DataKind.FINANCIAL_RATIOS.value)],
        'issuer_events': [IssuerEvent(title='FPT họp đại hội cổ đông', event_date=market_data.as_of,
                                      source_id=DataKind.ISSUER_EVENTS.value)],
    })
    graph.ingest_market(data)
    stats = graph.stats()
    graph.ingest_market(data)
    assert graph.stats() == stats
    rows = graph.query('FPT', as_of=market_data.fetched_at, limit=50)
    relations = {row['triple']['relation'] for row in rows}
    assert {'LISTED_AS', 'IN_SECTOR', 'HAS_EVENT', 'HAS_PERIOD', 'REPORTS_METRIC'} <= relations
    assert not {'EXPOSED_TO', 'CAUSES'} & relations
    assert all(row['evidence']['source_url'].startswith('https://example.invalid/') for row in rows)
