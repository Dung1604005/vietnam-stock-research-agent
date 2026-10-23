from datetime import date, datetime, timezone
from types import SimpleNamespace

from stock_multi_agent.agents.fundamental_agent import FundamentalAgent
from stock_multi_agent.agents.research_agent import ResearchAgent
from stock_multi_agent.models.analysis import Signal, FinancialMetrics
from stock_multi_agent.models.data import FinancialMetricObservation, FinancialStatementObservation
from stock_multi_agent.models.research import EvidenceBundle
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.execution_profiles import execution_scope
from stock_multi_agent.services.financial_freshness import scoring_financials, fundamental_inputs
from stock_multi_agent.tools.financial_data import FinancialToolkit


def test_2018_ratios_provide_explicit_historical_assessment(market_data):
    observations = [FinancialMetricObservation(metric=name, value=value,
        unit={'pe': 'multiple', 'pb': 'multiple', 'eps': 'currency_per_share',
              'operating_cash_flow': 'currency'}.get(name, 'ratio'),
        period_label='2018', period_end=date(2018, 12, 31), source_id='old:ratio')
        for name, value in market_data.financials.model_dump().items() if value is not None]
    data = market_data.model_copy(update={'financial_observations': observations})
    eligible, notes = scoring_financials(data)
    assert eligible.available_count() == 0 and notes
    result = FundamentalAgent(FinancialToolkit()).run(AgentState(ticker=data.ticker,
        investment_horizon='3 months', user_goal='current analysis', market_data=data), allow_llm_review=False)['fundamental_result']
    assert result.signal is not Signal.UNKNOWN
    assert result.data_mode == 'LATEST_AVAILABLE'
    assert result.fallback_metrics
    assert set(result.metric_periods.values()) == {date(2018, 12, 31)}
    assert result.confidence <= 0.23
    assert all(item.as_of == date(2018, 12, 31) for item in result.evidence)
    assert all(item.confidence <= 0.25 for item in result.evidence)
    assert 'lịch sử' in result.reasoning_summary
    assert data.financials.available_count() > 0  # Preserve historical payload.


def ratio(metric, value, period, **kwargs):
    return FinancialMetricObservation(metric=metric, value=value,
        unit='multiple' if metric in {'pe', 'pb'} else 'ratio',
        period_label=period.isoformat() if period else 'unknown', period_end=period,
        source_id=f'ratio:{period}', **kwargs)


def test_latest_metric_wins_over_stale_aggregate_and_keeps_its_source(market_data):
    old, latest = date(2018, 12, 31), date(2025, 12, 31)
    observations = [ratio('pe', 99, old), ratio('pe', 12, latest), ratio('roe', 0.19, latest)]
    data = market_data.model_copy(update={'financials': FinancialMetrics(pe=99),
        'financial_observations': observations})
    selected = fundamental_inputs(data)
    assert selected.metrics.pe == 12
    assert selected.metrics.roe == 0.19  # Recover even if the aggregate is missing.
    assert not selected.fallback_metrics
    result = FundamentalAgent(FinancialToolkit()).run(AgentState(ticker=data.ticker,
        investment_horizon='3 months', user_goal='Analyze', market_data=data), allow_llm_review=False)['fundamental_result']
    pe = next(item for item in result.evidence if item.label == 'P/E')
    assert pe.value == '12.00' and pe.as_of == latest
    assert pe.source_ids == [f'ratio:{latest}']
    assert data.financials.pe == 99  # The input/cache is never rewritten.


def test_conflicting_latest_period_falls_back_without_guessing(market_data):
    old, latest = date(2023, 12, 31), date(2025, 12, 31)
    data = market_data.model_copy(update={'financial_observations': [
        ratio('pe', 14, old), ratio('pe', 14, latest), ratio('pe', 29, latest)]})
    selected = fundamental_inputs(data)
    assert selected.metrics.pe == 14 and selected.metric_periods['pe'] == old
    assert selected.fallback_metrics == ['pe']
    assert any('conflicting' in note for note in selected.notes)
    result = FundamentalAgent(FinancialToolkit()).run(AgentState(ticker=data.ticker,
        investment_horizon='3 months', user_goal='Analyze', market_data=data), allow_llm_review=False)['fundamental_result']
    pe = next(item for item in result.evidence if item.label == 'P/E')
    assert pe.as_of == old and pe.source_ids == [f'ratio:{old}']


def test_unknown_future_and_ambiguous_data_remain_unavailable(market_data):
    unknown_scale = ratio('roe', 19, date(2025, 12, 31)).model_copy(update={'unit': 'unknown_ratio'})
    data = market_data.model_copy(update={'financial_observations': [unknown_scale,
        ratio('pe', 12, None), ratio('pe', 11, date(2030, 12, 31)),
        ratio('pb', 2, date(2025, 12, 31), published_at=datetime(2030, 1, 1, tzinfo=timezone.utc))]})
    result = FundamentalAgent(FinancialToolkit()).run(AgentState(ticker=data.ticker,
        investment_horizon='3 months', user_goal='Analyze', market_data=data), allow_llm_review=False)['fundamental_result']
    assert result.signal is Signal.UNKNOWN and result.data_mode == 'UNAVAILABLE'


def test_older_complete_statements_remain_assessable(market_data):
    statements = [FinancialStatementObservation(statement='income_statement', metric='revenue',
        value=value, unit='currency', currency='VND', scale=1, reporting_basis='ANNUAL',
        consolidation='CONSOLIDATED', period_label=str(year), period_end=date(year, 12, 31),
        source_id=f'statement:{year}') for year, value in [(2021, 100), (2022, 120)]]
    data = market_data.model_copy(update={'financials': FinancialMetrics(),
        'financial_statement_observations': statements})
    result = FundamentalAgent(FinancialToolkit()).run(AgentState(ticker=data.ticker,
        investment_horizon='3 months', user_goal='Analyze', market_data=data), allow_llm_review=False)['fundamental_result']
    assert result.signal is not Signal.UNKNOWN
    assert result.data_mode == 'LATEST_AVAILABLE'
    assert result.metric_periods == {'revenue_growth': date(2022, 12, 31)}
    assert 'reported_revenue_growth' in result.feature_contributions
    assert result.confidence < 0.3


def test_fallback_confidence_decreases_with_age(market_data):
    agent = FundamentalAgent(FinancialToolkit())
    results = []
    for year in [2025, 2023, 2022, 2018]:
        data = market_data.model_copy(update={'financial_observations': [ratio('pe', 12, date(year, 12, 31))]})
        results.append(agent.run(AgentState(ticker=data.ticker, investment_horizon='3 months',
            user_goal='Analyze', market_data=data), allow_llm_review=False)['fundamental_result'])
    assert all(first.confidence > second.confidence for first, second in zip(results, results[1:]))
    assert len({result.score for result in results}) == 1  # Age is not poor business performance.


def test_response_shows_selected_value_period_and_fallback_scope(market_data):
    from stock_multi_agent.chat.response import ResponseComposer
    data = market_data.model_copy(update={'financials': FinancialMetrics(pe=99),
        'financial_observations': [ratio('pe', 12, date(2022, 12, 31))]})
    state = AgentState(ticker=data.ticker, investment_horizon='3 months', user_goal='Analyze',
        market_data=data, allow_llm_planning=False)
    state.fundamental_result = FundamentalAgent(FinancialToolkit()).run(state, allow_llm_review=False)['fundamental_result']
    response = ResponseComposer.fundamental(state)
    assert response.metadata['fundamental_data_mode'] == 'LATEST_AVAILABLE'
    assert response.metadata['fundamental_metric_periods'] == {'pe': '2022-12-31'}
    assert response.metadata['fundamental_fallback_metrics'] == ['pe']
    assert '2022-12-31' in response.message and 'lịch sử' in response.message
    table = next(table for table in response.tables if table.title == 'Chỉ số doanh nghiệp và cách đọc')
    pe = next(row for row in table.rows if row['Chỉ tiêu'] == 'P/E')
    assert pe['Giá trị'] == '12.00' and pe['Kỳ dữ liệu'] == '2022-12-31'
    assert pe['Phạm vi'] == 'Kỳ gần nhất (lịch sử)'
    from stock_multi_agent.services.audit_export import build_audit
    audit = build_audit(response)['audit']
    assert audit['fundamental_metric_periods'] == {'pe': '2022-12-31'}
    assert audit['fundamental_data_mode'] == 'LATEST_AVAILABLE'


def test_llm_review_receives_selected_periods_instead_of_old_history(market_data):
    data = market_data.model_copy(update={'financials': FinancialMetrics(pe=99),
        'financial_observations': [ratio('pe', 99, date(2018, 12, 31)),
            ratio('pe', 12, date(2025, 12, 31))]})
    state = AgentState(ticker=data.ticker, investment_horizon='3 months', user_goal='Analyze', market_data=data)
    agent = FundamentalAgent(FinancialToolkit())
    result = agent.run(state, allow_llm_review=False)['fundamental_result']
    captured = {}
    agent.llm_review = lambda role, facts, sources: captured.update(facts)
    agent.review_result(state, result)
    assert captured['reported_metrics']['pe'] == 12
    assert [(item['value'], item['period_end']) for item in captured['metric_periods']] == [(12, '2025-12-31')]


def test_each_ratio_uses_its_own_period(market_data):
    observations = [FinancialMetricObservation(metric='pe', value=market_data.financials.pe,
        unit='multiple', period_label='2018', period_end=date(2018, 12, 31), source_id='ratio'),
        FinancialMetricObservation(metric='roe', value=market_data.financials.roe,
        unit='ratio', period_label='2025', period_end=date(2025, 12, 31), source_id='ratio')]
    eligible, _ = scoring_financials(market_data.model_copy(update={'financial_observations': observations}))
    assert eligible.pe is None
    assert eligible.roe == market_data.financials.roe


def test_quick_research_checks_web_without_llm_and_offline_never_does(market_data):
    class Store:
        def search(self, *args, **kwargs):
            return EvidenceBundle(query=args[0], citations=[], retrieval_note='fixture')
    class Web:
        calls = 0
        def research(self, *args, **kwargs):
            self.calls += 1
            return SimpleNamespace(external_citations=[], discovery_citations=[])
    web = Web()
    agent = ResearchAgent(Store(), web_research=web)
    state = AgentState(ticker='FPT', investment_horizon='3 months', user_goal='Analyze FPT',
        market_data=market_data, allow_llm_planning=False)
    with execution_scope('QUICK'):
        agent.run(state)
    assert web.calls == 1
    with execution_scope('OFFLINE'):
        agent.run(state)
    assert web.calls == 1
