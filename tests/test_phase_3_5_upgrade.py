"""Authored offline fixtures, NOT independent market/LLM performance evidence."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import BaseModel

from stock_multi_agent.models.graph_v2 import GraphEntity, GraphEvidence, GraphTriple
from stock_multi_agent.models.analysis import Signal, Recommendation
from stock_multi_agent.models.chat import AssistantResponse, ConversationState, QueryIntent
from stock_multi_agent.models.data import Instrument, PortfolioPosition
from stock_multi_agent.models.research_contracts import RequestBudget
from stock_multi_agent.models.screening import ScreenSpec
from stock_multi_agent.services.graph_v2 import InvestmentGraph
from stock_multi_agent.services.llm_gateway import LLMGateway
from stock_multi_agent.services.request_telemetry import request_scope, BudgetExceeded
from stock_multi_agent.services.research_workbench import ResearchWorkbench
from stock_multi_agent.services.research_store import ResearchStore
from stock_multi_agent.chat.query_parser import QueryParser
from stock_multi_agent.chat.router import IntentRouter, RoutingPlanValidator
from stock_multi_agent.services.screening_service import ScreeningService
from stock_multi_agent.services.universe_service import UniverseService
from stock_multi_agent.services.analysis_service import AnalysisService
from conftest import conflict_market_data

NOW = datetime(2026, 10, 17, tzinfo=timezone.utc)


@pytest.fixture
def root():
    root = Path(__file__).resolve().parents[1] / '.runtime' / 'phase-3-5-tests' / uuid4().hex
    root.mkdir(parents=True)
    return root


def make_edge(graph, ticker='FPT'):
    evidence = GraphEvidence(evidence_id=f'evidence:{ticker}', document_id=f'doc:{ticker}',
        source_url='https://example.invalid/report', quote=f'{ticker} thuộc ngành Công nghệ.',
        published_at=NOW, retrieved_at=NOW, layer='research')
    graph.put_evidence(evidence)
    triple = GraphTriple(subject=GraphEntity(entity_id=f'company:{ticker}', kind='Company', label=ticker),
        object=GraphEntity(entity_id='sector:technology', kind='Sector', label='Công nghệ'),
        relation='IN_SECTOR', evidence_id=evidence.evidence_id, valid_from=NOW, confidence=.8)
    graph.put(triple, reviewed=True)
    return triple


def test_graph_provenance_review_and_idempotency(root):
    graph = InvestmentGraph(root / 'graph.sqlite3')
    triple = make_edge(graph)
    graph.put(triple, reviewed=True)
    assert graph.stats() == {'nodes': 2, 'edges': 1, 'evidence': 1}
    with pytest.raises(ValueError, match='review'):
        graph.put(triple)
    with pytest.raises(ValueError, match='review'):
        graph.put(triple.model_copy(update={'origin': 'deterministic'}))
    with pytest.raises(ValueError, match='provenance'):
        graph.put(triple.model_copy(update={'evidence_id': 'absent'}), reviewed=True)
    with pytest.raises(ValueError, match='absent'):
        graph.put(triple.model_copy(update={'object': GraphEntity(entity_id='sector:bank', kind='Sector', label='Ngân hàng')}), reviewed=True)
    assert graph.query('FPT thuộc ngành nào?', as_of=NOW - timedelta(days=1)) == []
    assert graph.query('FPT thuộc ngành nào?', as_of=NOW)[0]['evidence']['source_url'].startswith('https:')


@pytest.mark.parametrize('ticker', ['FPT', 'CMG', 'CTR', 'ELC', 'FOX'])
@pytest.mark.parametrize('template', ['{ticker} thuộc ngành nào?', 'Nguồn nào nói về {ticker}?',
    'Quan hệ của {ticker} với Công nghệ?', 'Bằng chứng ngành của {ticker}'])
def test_twenty_synthetic_graph_queries_have_citations(root, ticker, template):
    graph = InvestmentGraph(root / 'graph.sqlite3')
    make_edge(graph, ticker)
    result = graph.query(template.format(ticker=ticker), as_of=NOW)
    assert any(item['triple']['subject']['label'] == ticker and item['evidence']['evidence_id'] == f'evidence:{ticker}' for item in result)


def test_alias_ambiguity_and_temporal_validity(root):
    graph = InvestmentGraph(root / 'graph.sqlite3')
    first = make_edge(graph, 'FPT')
    make_edge(graph, 'CMG')
    graph.add_alias('Công ty A', 'company:FPT', first.evidence_id)
    graph.add_alias('Công ty A', 'company:CMG', 'evidence:CMG')
    assert len(graph.resolve('Công ty A')) == 2
    graph.put(first.model_copy(update={'valid_to': NOW + timedelta(days=1)}), reviewed=True)
    assert graph.query('FPT', as_of=NOW + timedelta(days=2)) == []


def test_graph_projection_merge_and_sqlite_fallback(root):
    graph = InvestmentGraph(root / 'graph.sqlite3')
    make_edge(graph)
    class Result:
        def consume(self): pass
    class Backend:
        available, database = True, 'neo4j'
        def __init__(self):
            self.driver = self
            self.edge_ids = set()
        def session(self, **kwargs): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def run(self, query, **params):
            if 'id' in params:
                assert 'MERGE (s)-[r:RELATES_V2' in query
                assert params['source_url'] and params['published_at']
                self.edge_ids.add(params['id'])
            return Result()
    backend = Backend()
    assert graph.sync_neo4j(backend)['available']
    graph.sync_neo4j(backend)
    assert len(backend.edge_ids) == 1
    backend.available = False
    assert graph.sync_neo4j(backend)['backend'] == 'sqlite'


class Review(BaseModel):
    assessment: str
    cited_source_ids: list[str] = []


class StubLLM:
    model = 'test-offline'
    timeout_seconds = 120
    def __init__(self, output=None):
        self.calls = []
        self.output = output or {'assessment': 'Dữ liệu cho thấy cần theo dõi.', 'cited_source_ids': ['S1']}
    def parse_structured(self, system, payload, output_model):
        self.calls.append((self.timeout_seconds, system))
        return output_model.model_validate(self.output)


def test_gateway_cache_budget_and_privacy():
    provider = StubLLM()
    gateway = LLMGateway(provider)
    with request_scope(RequestBudget(max_llm_calls=1)) as metrics:
        for _ in range(2):
            gateway.parse_structured('prompt', {'facts': {'score': 7}, 'allowed_source_ids': ['S1']}, Review)
        assert len(provider.calls) == 1
        assert provider.calls[0][0] <= 45
        assert metrics.llm_calls == 1 and metrics.cache_hits == 1
        assert metrics.output_token_upper_bound > 0
        assert all('prompt' not in row and 'payload' not in row for row in metrics.nodes)
        with pytest.raises(BudgetExceeded):
            gateway.parse_structured('different', {'allowed_source_ids': ['S1']}, Review)


@pytest.mark.parametrize('output', [
    {'assessment': 'Giá sẽ lên 99999.', 'cited_source_ids': ['S1']},
    {'assessment': '数据分析', 'cited_source_ids': ['S1']},
    {'assessment': 'Dữ liệu chưa đủ.', 'cited_source_ids': ['invented']},
])
def test_gateway_rejects_fabrication_language_and_sources(output):
    gateway = LLMGateway(StubLLM(output), failure_threshold=1)
    with pytest.raises(ValueError):
        gateway.parse_structured('prompt', {'facts': {'price': 42}, 'allowed_source_ids': ['S1']}, Review)
    assert gateway.health()['state'] == 'OPEN'
    with pytest.raises(RuntimeError, match='circuit'):
        gateway.parse_structured('prompt', {'allowed_source_ids': ['S1']}, Review)


def test_gateway_output_budget():
    gateway = LLMGateway(StubLLM())
    with request_scope(RequestBudget(max_output_tokens=64)):
        with pytest.raises(BudgetExceeded):
            gateway.parse_structured('prompt', {'allowed_source_ids': ['S1']}, Review)


@pytest.mark.parametrize('question,intent', [
    ('Rủi ro danh mục FPT 60% HPG 40%', QueryIntent.PORTFOLIO_RISK),
    ('Kịch bản FPT -5% +10%', QueryIntent.SCENARIO_ANALYSIS),
    ('Vì sao kết luận FPT thay đổi?', QueryIntent.CHANGE_EXPLANATION),
    ('Tác động sự kiện FPT', QueryIntent.EVENT_IMPACT),
    ('So sánh báo cáo FPT hai kỳ', QueryIntent.REPORT_CHANGE),
    ('So sánh cùng ngành FPT CMG', QueryIntent.PEER_COMPARISON),
    ('Xuất báo cáo HTML', QueryIntent.REPORT_EXPORT),
])
def test_new_chat_features_have_typed_executable_routes(question, intent):
    query = QueryParser(use_configured_provider=False).parse(question)
    assert query.intent == intent
    assert not query.needs_clarification
    assert not RoutingPlanValidator.validate(IntentRouter().plan(query), query)
    if intent == QueryIntent.PORTFOLIO_RISK:
        assert query.portfolio_weights == {'FPT': .6, 'HPG': .4}
    if intent == QueryIntent.SCENARIO_ANALYSIS:
        assert query.scenario_shocks == [-.05, .1]


def test_bad_weights_are_not_silently_normalized():
    parser = QueryParser(use_configured_provider=False)
    assert parser.parse('Rủi ro danh mục FPT 80% HPG 40%').needs_clarification
    assert parser.parse('Rủi ro danh mục FPT 40% FPT 60%').needs_clarification


class MarketProvider:
    def __init__(self): self.calls = 0
    def get_market_data(self, ticker, include_financials=True):
        self.calls += 1
        data = conflict_market_data(ticker)
        data.instrument = Instrument(ticker=ticker, name=ticker, sector='technology', currency='VND')
        data.price_unit = 'provider_native'
        return data


def test_snapshot_explanation_and_html_export(root):
    analysis = AnalysisService(provider=MarketProvider())
    state = analysis.analyze('FPT', max_iterations=0, allow_llm_planning=False)
    state.market_data.source_refs[0].source_url = 'https://example.invalid/prices'
    service = ResearchWorkbench(root / 'research.sqlite3')
    first = service.capture(state)
    assert first['graph_sync_status'] == 'SYNCED'
    graph = InvestmentGraph(service.db_path)
    assert graph.query('FPT', as_of=state.market_data.fetched_at) == []  # Do not backdate a new forecast.
    state.technical_result.score += .5
    state.final_decision.recommendation = Recommendation.WATCH
    second = service.capture(state)
    assert second['graph_sync_status'] == 'SYNCED'
    assert first['snapshot_id'] != second['snapshot_id']
    response = service.explain_change('FPT')
    assert any(row['Mục'] == 'technical.score' for row in response.tables[0].rows)
    assert response.metadata['snapshot_comparison'][1]['sections']['technical']['tool']
    response.message += '<script>alert(1)</script>'
    report = service.export(response, 'html')
    assert '<script>' not in report and '&lt;script&gt;' in report


def test_portfolio_and_scenarios_handle_partial_and_alignment(root):
    provider = MarketProvider()
    fpt, hpg = provider.get_market_data('FPT'), provider.get_market_data('HPG')
    positions = [PortfolioPosition(ticker='FPT', weight=.6), PortfolioPosition(ticker='HPG', weight=.3)]
    result = ResearchWorkbench.portfolio(positions, {'FPT': fpt, 'HPG': hpg})
    assert result['quantitative_status'] == 'ASSESSED'
    assert result['cash_weight'] == pytest.approx(.1)
    assert result['annualized_volatility'] >= 0
    assert ResearchWorkbench.portfolio(positions, {'FPT': fpt})['quantitative_status'] == 'UNAVAILABLE'
    hpg.instrument.currency = 'USD'
    assert ResearchWorkbench.portfolio(positions, {'FPT': fpt, 'HPG': hpg})['quantitative_status'] == 'UNAVAILABLE'
    assert ResearchWorkbench.scenarios(fpt, [.1])[0]['Giá theo giả định'] == pytest.approx(fpt.bars[-1].close * 1.1)


def test_screen_warm_cache_does_not_repeat_universe_io():
    provider = MarketProvider()
    screening = ScreeningService(AnalysisService(provider=provider), UniverseService({'default': ['FPT', 'CMG']}),
        today_provider=lambda: date(2026, 10, 9))
    spec = ScreenSpec(universe_key='default', refresh_stale_data=False)
    first = screening.run_progressive(spec, '3 months', 1)
    calls = provider.calls
    second = screening.run_progressive(spec, '3 months', 2)
    assert provider.calls == calls
    assert first is not second


def test_filter_nlu_is_executable_and_not_ignored():
    query = QueryParser(use_configured_provider=False).parse('Tìm 10 mã cổ phiếu PE dưới 15')
    assert query.filters == {'pe_lt': 15}
    assert not query.needs_clarification


def test_thesis_is_not_confirmed_by_related_citations(root):
    store = ResearchWorkbench(root / 'research.sqlite3')
    thesis = store.track_thesis('FPT', 'FPT tăng trưởng', [])
    assert thesis['status'] == 'INSUFFICIENT_EVIDENCE'
    thesis = store.track_thesis('FPT', 'FPT tăng trưởng', [{'quote': 'Một đoạn liên quan.'}])
    assert thesis['status'] == 'NEEDS_REVIEW'


def test_graph_candidates_are_versioned_citations_and_backfill_is_idempotent(root):
    from stock_multi_agent.models.research import ResearchDocumentType
    from stock_multi_agent.services.embeddings import LocalHashEmbeddingProvider
    store = ResearchStore(root / 'research.sqlite3', embedding_provider=LocalHashEmbeddingProvider())
    kwargs = dict(title='Công bố FPT', document_type=ResearchDocumentType.NEWS,
        source_url='https://example.invalid/fpt', publisher='Authored fixture', published_at=NOW - timedelta(days=2),
        tickers=['FPT'], usage_note='Synthetic fixture, not issuer content')
    first = store.ingest_text(text='FPT có hoạt động kinh doanh công nghệ, doanh thu trong báo cáo.', **kwargs)
    assert store.last_graph_error is None
    citations = store.graph_citations('Nguồn nào nhắc FPT?')
    assert citations and citations[0].document_id == first.document_id
    graph = InvestmentGraph(store.database_path)
    stats = graph.stats()
    store.ingest_text(text='FPT có hoạt động kinh doanh công nghệ, doanh thu trong báo cáo.', **kwargs)
    assert graph.stats() == stats
    second = store.ingest_text(text='FPT cập nhật tài liệu với nội dung khác về chiến lược và sản phẩm.', **kwargs)
    assert second.supersedes_id == first.document_id
    assert all(item.document_id != first.document_id for item in store.graph_citations('FPT'))


def test_financial_change_and_peers_refuse_mixed_units(root):
    from stock_multi_agent.models.data import FinancialStatementObservation, FinancialMetricObservation
    provider = MarketProvider()
    data = provider.get_market_data('FPT')
    data.financial_statement_observations = [FinancialStatementObservation(statement='income_statement', metric='revenue',
        value=value, unit=unit, period_label=str(year), period_end=date(year, 12, 31), source_id=f'fixture:{year}',
        reporting_basis='ANNUAL', consolidation='CONSOLIDATED')
        for year, value, unit in [(2023, 100, 'VND'), (2024, 120, 'VND'), (2025, 200, 'USD')]]
    rows = ResearchWorkbench.report_changes(data)
    assert len(rows) == 1 and rows[0]['Thay đổi %'] == pytest.approx(20)
    states = []
    for ticker in ('FPT', 'CMG'):
        state = AnalysisService(provider=provider).fetch_data(ticker, include_forecast=False)
        state.market_data.financial_observations = [FinancialMetricObservation(metric='pe', value=10,
            unit='ratio', period_label='2025', period_end=date(2025, 12, 31), source_id=f'fixture:{ticker}',
            reporting_basis='ANNUAL', consolidation='CONSOLIDATED')]
        states.append(state)
    assert len(ResearchWorkbench.peer_percentiles(states)) == 2
    states[1].market_data.financial_observations[0].unit = 'USD'
    assert ResearchWorkbench.peer_percentiles(states) == []


def test_optional_research_query_llm_has_bounded_validated_fallback():
    from stock_multi_agent.services.research_planner import ResearchPlanner
    class Planner:
        def parse_structured(self, system, payload, output_model):
            return output_model.model_validate({'queries': ['FPT kết quả kinh doanh chính thức', 'HPG nội dung không phù hợp']})
    plan = ResearchPlanner(provider=Planner(), enabled=True).plan('Tìm báo cáo FPT', ticker='FPT')
    assert len(plan.queries) == 2 and all('FPT' in item.query for item in plan.queries)


def test_chat_end_to_end_workbench_features(root):
    from stock_multi_agent.services.chat_service import ChatService
    from stock_multi_agent.services.knowledge_store import KnowledgeStore
    from stock_multi_agent.services.knowledge_service import KnowledgeService
    from stock_multi_agent.services.knowledge_graph import Neo4jKnowledgeGraph
    from stock_multi_agent.services.embeddings import LocalHashEmbeddingProvider
    research = ResearchStore(root / 'research.sqlite3', embedding_provider=LocalHashEmbeddingProvider())
    knowledge = KnowledgeService(KnowledgeStore(root / 'kb.sqlite3', knowledge_graph=Neo4jKnowledgeGraph(enabled=False)), bootstrap=False)
    service = ChatService(analysis_service=AnalysisService(provider=MarketProvider(), research_store=research),
        research_store=research, knowledge_service=knowledge, query_parser=QueryParser(use_configured_provider=False), universe={'default': ['FPT', 'HPG']})
    context = ConversationState()
    response = service.handle('Phân tích FPT', context)
    assert response.metadata['research_brief']['sections']['technical']['tool']
    service.handle('Phân tích FPT', context)
    response = service.handle('Vì sao kết luận FPT thay đổi?', context)
    assert len(response.metadata['snapshot_comparison']) == 2
    response = service.handle('Rủi ro danh mục FPT 60% HPG 40%', context)
    assert response.metadata['portfolio_risk']['quantitative_status'] == 'ASSESSED'
    response = service.handle('Kịch bản FPT giảm 5%', context)
    assert response.tables[0].rows[0]['Giả định %'] == -5
    response = service.handle('Xuất báo cáo HTML', context)
    assert response.metadata['export_ready']


def test_custom_screen_filters_are_executed_not_only_parsed():
    provider = MarketProvider()
    screen = ScreeningService(AnalysisService(provider=provider), UniverseService({'default': ['FPT', 'CMG']}))
    result = screen.run_progressive(ScreenSpec(universe_key='default', minimum_technical_score=0,
        eligible_signals=list(Signal),
        refresh_stale_data=False, filters={'pe_lt': 1}), '3 months', 2)
    assert result.selected_tickers == []
    assert any('pe_lt' in reason for item in result.candidates for reason in item.reasons)


def test_financial_mapping_is_quote_unit_period_checked_and_never_scored():
    from stock_multi_agent.services.agent_llm import AgentLLMAdvisor
    from stock_multi_agent.models.research import Citation
    citation = Citation(document_id='d', chunk_id='S1', title='Fixture', source_url='https://example.invalid/report',
        publisher='Fixture', published_at=NOW, retrieved_at=NOW, content_sha256='a' * 64,
        quote='ROE đạt 20% trong kỳ báo cáo.', period_end=date(2026, 9, 30))
    class Mapper:
        def parse_structured(self, system, payload, output_model):
            return output_model.model_validate({'facts': [
                {'metric': 'roe', 'value': 20, 'unit': 'percent', 'period_end': '2026-09-30', 'source_id': 'S1', 'exact_quote': citation.quote},
                {'metric': 'roe', 'value': 99, 'unit': 'percent', 'period_end': '2026-09-30', 'source_id': 'S1', 'exact_quote': citation.quote}]})
    advisor = AgentLLMAdvisor(provider=Mapper(), enabled=True)
    advisor.roles = {'fundamental'}
    accepted = advisor.map_financial_facts([citation])
    assert len(accepted) == 1 and accepted[0]['value'] == 20
    assert accepted[0]['status'] == 'PROPOSAL_NOT_SCORED'
