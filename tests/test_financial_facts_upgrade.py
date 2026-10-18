"""Authored fixtures validate accounting guardrails, not live issuer accuracy."""
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from conftest import conflict_market_data
from stock_multi_agent.models.data import FinancialStatementObservation, FinancialMetricObservation
from stock_multi_agent.models.chart import ChartRequest, ChartType
from stock_multi_agent.providers.market_provider import VnstockMarketDataProvider as Provider
from stock_multi_agent.services.financial_facts import (normalize_observation, build_financial_evidence,
    comparability_reason, ResearchFinancialFactProposal, validate_research_fact)
from stock_multi_agent.services.financial_charts import financial_artifact, render_financial
from stock_multi_agent.services.research_workbench import ResearchWorkbench

NOW = datetime(2026, 10, 17, tzinfo=timezone.utc)


def fact(year=2025, metric='revenue', value=120, **updates):
    payload = dict(statement='income_statement', metric=metric, value=value, unit='tỷ đồng',
        period_label=str(year), period_end=date(year, 12, 31), frequency='annual',
        reporting_basis='ANNUAL', consolidation='CONSOLIDATED', assurance='AUDITED', source_id='fixture:issuer')
    payload.update(updates)
    return FinancialStatementObservation(**payload)


def dataset(*items):
    return conflict_market_data().model_copy(update={'financial_statement_observations': list(items),
        'financial_observations': [], 'fetched_at': NOW, 'as_of': NOW.date()})


def test_explicit_scale_normalizes_without_guessing_unknown_currency():
    result = normalize_observation(fact())
    assert result.normalized_value == 120e9 and result.currency == 'VND'
    unknown = normalize_observation(fact(unit='currency'))
    assert unknown.normalized_value is None and unknown.status == 'PARTIAL'
    conflict = normalize_observation(fact(currency='USD'))
    assert conflict.normalized_value is None
    assert normalize_observation(fact(unit='VND', scale=1_000)).normalized_value == 120_000


def test_derived_yoy_growth_and_cash_conversion_have_input_lineage():
    result = build_financial_evidence(dataset(fact(2024, value=100), fact(),
        fact(metric='net_profit', value=10), fact(metric='operating_cash_flow', value=12, statement='cash_flow')))
    metrics = {metric['metric']: metric for metric in result['derived_metrics']}
    assert metrics['revenue_growth']['value'] == pytest.approx(.2)
    assert metrics['cash_conversion']['value'] == pytest.approx(1.2)
    assert metrics['revenue_growth']['evidence']['source_ids'] == ['fixture:issuer']
    assert len(metrics['revenue_growth']['input_fields']) == 2


@pytest.mark.parametrize('updates', [{'reporting_basis': 'UNKNOWN'}, {'consolidation': 'SEPARATE'},
    {'consolidation': 'UNKNOWN'}, {'assurance': 'UNAUDITED'}, {'unit': 'USD'}, {'unit': 'currency'},
    {'reporting_basis': 'YTD'}, {'frequency': 'quarter'}])
def test_incompatible_bases_units_or_unknown_metadata_do_not_generate_growth(updates):
    result = build_financial_evidence(dataset(fact(2024, value=100), fact(**updates)))
    assert result['derived_metrics'] == [] and result['limitations']


def test_no_yoy_from_adjacent_quarters_or_non_positive_base():
    data = dataset(fact(period_end=date(2025, 3, 31), frequency='quarter', reporting_basis='STANDALONE'),
        fact(period_end=date(2025, 6, 30), frequency='quarter', reporting_basis='STANDALONE'))
    assert build_financial_evidence(data)['derived_metrics'] == []
    assert build_financial_evidence(dataset(fact(2024, value=-100), fact()))['derived_metrics'] == []


def test_future_publications_and_conflicting_values_are_not_selected():
    future = datetime(2027, 1, 1, tzinfo=timezone.utc)
    assert build_financial_evidence(dataset(fact(2024), fact(published_at=future)))['derived_metrics'] == []
    result = build_financial_evidence(dataset(fact(2024), fact(), fact(value=999, source_id='fixture:other')))
    assert not result['derived_metrics'] and any('conflicting' in note for note in result['limitations'])


def test_statement_history_supports_record_orientation_and_explicit_metadata():
    frame = pd.DataFrame({'yearReport': [2024, 2025], 'revenue': [100., 120.], 'net_profit': [10., 15.]})
    frame.attrs = {'unit': 'tỷ đồng', 'consolidation': 'CONSOLIDATED', 'assurance': 'AUDITED'}
    items = Provider._normalize_statement_observations(frame, 'income_statement', 'fixture:issuer')
    assert len(items) == 4 and {item.period_end.year for item in items} == {2024, 2025}
    assert all(item.reporting_basis == 'ANNUAL' and item.unit == 'tỷ đồng' for item in items)
    assert len(build_financial_evidence(dataset(*items))['derived_metrics']) == 2


def test_ratio_history_retains_all_periods_latest_first_and_exact_percentage():
    frame = pd.DataFrame([['roe', 'ROE (%)', .5, 20]], columns=['item_id', 'item_en', '2025', '2024'])
    metrics, items = Provider._normalize_ratios_with_observations(frame, 'fixture:ratios')
    assert metrics.roe == pytest.approx(.005)
    assert len(items) == 2 and items[0].period_label == '2025'
    assert items[1].value == pytest.approx(.2)
    assert all(item.reporting_basis == 'UNKNOWN' for item in items)


def test_ambiguous_ratio_scale_is_not_used_by_financial_chart():
    frame = pd.DataFrame([['roe', 'ROE', 20]], columns=['item_id', 'item_en', '2025'])
    metrics, items = Provider._normalize_ratios_with_observations(frame, 'fixture:ratios')
    assert items[0].unit == 'unknown_ratio' and normalize_observation(items[0]).normalized_value is None
    assert metrics.roe is None


def test_debt_equity_multiple_above_two_is_not_divided_by_one_hundred():
    frame = pd.DataFrame([['debt_to_equity', 'Debt to Equity', 3]], columns=['item_id', 'item_en', '2025'])
    metrics, items = Provider._normalize_ratios_with_observations(frame, 'fixture:ratios')
    assert metrics.debt_to_equity == 3 and items[0].value == 3


def test_record_ratio_history_selects_declared_period_not_row_position():
    frame = pd.DataFrame({'yearReport': [2024, 2025], 'pe_ratio': [12., 14.]})
    metrics, items = Provider._normalize_ratios_with_observations(frame, 'fixture:ratios')
    assert metrics.pe == 14 and len(items) == 2
    assert Provider._period_end('2025-03-31') == date(2025, 3, 31)


def test_financial_charts_normalize_scales_and_explain_interpretation():
    data = dataset(fact(2024, value=100), fact(value=120_000, unit='triệu đồng'))
    artifact = financial_artifact(ChartRequest(ticker='FPT', chart_type=ChartType.REVENUE_PROFIT, period='ALL'), data)
    assert [row['value'] for row in artifact.records] == [100e9, 120e9]
    assert artifact.records[0]['unit'] == 'VND' and 'không tự chứng minh nguyên nhân' in artifact.explanation
    assert len(render_financial(artifact).data) == 1
    growth = financial_artifact(ChartRequest(ticker='FPT', chart_type=ChartType.GROWTH), data)
    assert growth.records[0]['value'] == pytest.approx(.2)


def test_financial_chart_frequency_and_period_controls_do_not_substitute_annual():
    annual = [fact(year) for year in (2022, 2023, 2024, 2025)]
    data = dataset(*annual)
    artifact = financial_artifact(ChartRequest(ticker='FPT', chart_type=ChartType.REVENUE_PROFIT, period='2Y'), data)
    assert len(artifact.records) == 2
    with pytest.raises(ValueError, match='Không có quan sát'):
        financial_artifact(ChartRequest(ticker='FPT', chart_type=ChartType.REVENUE_PROFIT, indicators=['QUARTER']), data)


def test_unknown_metadata_is_displayed_but_comparison_is_withheld():
    data = dataset(fact(2024, reporting_basis='UNKNOWN'), fact(reporting_basis='UNKNOWN'))
    rows = ResearchWorkbench.report_changes(data)
    assert rows[0]['Thay đổi %'] is None and rows[0]['Trạng thái'] == 'NOT_COMPARABLE'
    known = ResearchWorkbench.report_changes(dataset(fact(2024, value=100), fact()))
    assert known[0]['Thay đổi %'] == pytest.approx(20)


def proposal_fixture():
    quote = 'Doanh thu: 120 tỷ đồng cả năm 2025 hợp nhất.'
    citation = SimpleNamespace(chunk_id='fixture:chunk', tickers=['FPT'], quote=quote,
        source_url='https://issuer.invalid/report', published_at=NOW)
    proposal = ResearchFinancialFactProposal(ticker='FPT', statement='income_statement', metric='revenue', value=120,
        unit='tỷ đồng', currency='VND', scale=1e9, reporting_basis='ANNUAL', consolidation='CONSOLIDATED',
        period_label='2025', period_end=date(2025, 12, 31), source_id='fixture:chunk', evidence_quote=quote)
    return proposal, citation


def test_exact_research_fact_requires_metric_issuer_period_unit_and_source():
    proposal, citation = proposal_fixture()
    accepted = validate_research_fact(proposal, citation)
    assert accepted.value == 120 and accepted.source_url == citation.source_url
    for changed in [{'ticker': 'HPG'}, {'metric': 'net_profit'}, {'value': 2025}, {'scale': 1}, {'period_end': date(2024, 12, 31)}]:
        with pytest.raises(ValueError):
            validate_research_fact(proposal.model_copy(update=changed), citation)


def test_exact_research_fact_rejects_wrong_quarter_and_label_value_mapping():
    proposal, citation = proposal_fixture()
    quote = 'Doanh thu: 120 tỷ đồng quý riêng quý 1 năm 2025 hợp nhất.'
    citation.quote = quote
    altered = proposal.model_copy(update={'evidence_quote': quote, 'reporting_basis': 'STANDALONE', 'period_end': date(2025, 6, 30)})
    with pytest.raises(ValueError, match='period'):
        validate_research_fact(altered, citation)
    citation.quote = quote = 'Doanh thu: 120 tỷ đồng, lợi nhuận sau thuế: 20 tỷ đồng cả năm 2025 hợp nhất.'
    with pytest.raises(ValueError, match='immediately'):
        validate_research_fact(proposal.model_copy(update={'metric': 'net_profit', 'evidence_quote': quote}), citation)
    quote = 'Doanh thu: 120 tỷ đồng cả năm 2025 hợp nhất; giá cổ phiếu: 10 VND.'
    citation.quote = quote
    with pytest.raises(ValueError, match='immediately'):
        validate_research_fact(proposal.model_copy(update={'unit': 'VND', 'scale': 1, 'evidence_quote': quote}), citation)
