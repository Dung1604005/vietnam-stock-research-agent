from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from stock_multi_agent.models.chat import AssistantResponse
from stock_multi_agent.models.data import DataKind, SourceReference
from stock_multi_agent.services.interaction_actions import (
    actions_for_ticker, build_interaction_actions, validate_broker_template,
    validate_external_url, validate_ticker,
)


@pytest.mark.parametrize('url', [
    'javascript:alert(1)', 'http://example.com/file.pdf', 'file:///D:/private.txt',
    'https://user:pass@example.com/report.pdf', 'https://localhost/test',
    'https://127.0.0.1/test', 'https://[::1]/', 'https://192.168.1.1/',
    'https://169.254.169.254/', 'https://host.internal/path', 'https://intranet/path',
    'https://2130706433/', 'https://127.1/', 'https://0x7f.0.0.1/',
    'https://example.com:8000/test', 'https://example.com/a%0d%0ab',
    'https://example.com/a%250ab', 'https://example.com/a\\b',
    'https://example.com/?api_key=private', 'https://example.com/#token=private',
    'https://example.com/#/report?access_token=private', 'https://example.com/%22onclick=x',
    'https://[bad/path', 'https://example.com/a b',
])
def test_external_links_reject_unsafe_input(url):
    with pytest.raises(ValueError):
        validate_external_url(url)


def test_public_https_report_is_preserved_not_rewritten():
    url = 'https://fpt.com.vn/reports/2026-Q2.pdf?download=1#page=3'
    assert validate_external_url(url) == url


def test_encoded_spaces_in_actual_report_path_and_query_are_allowed():
    url = 'https://fpt.com.vn/reports/Bao%20cao%20tai%20chinh.pdf?name=Quy%202'
    assert validate_external_url(url) == url


@pytest.mark.parametrize('escaped', ['%0d%0a', '%0A', '%09', '%250d%250a'])
def test_encoded_control_characters_are_not_ordinary_spaces(escaped):
    with pytest.raises(ValueError):
        validate_external_url(f'https://fpt.com.vn/reports/Bao{escaped}cao.pdf')


@pytest.mark.parametrize('ticker', ['', 'FPT/evil', '<script>', 'FPT?buy=1', '123', 'A', 'TOOLONG123456'])
def test_invalid_ticker_rejected(ticker):
    with pytest.raises(ValueError):
        validate_ticker(ticker)


def test_links_for_stock_do_not_guess_exchange_or_fetch_network():
    with patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
        actions = actions_for_ticker(' fpt ')
        assert all(action.ticker == 'FPT' for action in actions)
        assert next(a for a in actions if a.kind == 'CHART').url.endswith('/Stocks/FPT/Chart.aspx')
        assert any(a.url == 'https://dstock.vndirect.com.vn/suc-khoe-co-phieu/FPT' for a in actions)
        assert next(a for a in actions_for_ticker('SHS', 'HNX') if a.kind == 'CHART').url.endswith('/HNX-SHS/')
        assert 'HOSE' not in next(a for a in actions_for_ticker('ZZZZZZ', 'UNKNOWN') if a.kind == 'CHART').url


def citation(**kwargs):
    return {'title': 'FPT báo cáo quý II', 'source_url': 'https://fpt.com.vn/2026-Q2.pdf',
            'document_type': 'financial_statement', 'tickers': ['FPT'],
            'publisher': 'FPT', 'published_at': '2026-07-25T09:00:00+07:00',
            'reporting_period': {'label': 'Q2/2026', 'status': 'DETECTED'}, **kwargs}


def test_exact_report_with_source_publication_and_reporting_period():
    response = AssistantResponse(message='ok', tickers=['FPT'], metadata={'research_citations': [citation(), citation()]})
    actions = build_interaction_actions(response)
    reports = [a for a in actions if a.kind == 'REPORT']
    assert len(reports) == 1
    assert reports[0].url == 'https://fpt.com.vn/2026-Q2.pdf'
    assert reports[0].ticker == 'FPT'
    assert reports[0].period_label == 'Q2/2026'
    assert reports[0].published_at == '2026-07-25T09:00:00+07:00'
    assert 'mới nhất' in reports[0].description  # explicitly NOT claimed to be latest


def test_discovery_pdf_is_not_promoted_to_verified_report():
    response = AssistantResponse(message='ok', tickers=['FPT'], metadata={
        'evidence_bundle': {'web_citations': [citation()]}})
    actions = build_interaction_actions(response)
    assert not [a for a in actions if a.kind == 'REPORT']
    discovery = next(a for a in actions if a.url.endswith('2026-Q2.pdf'))
    assert discovery.kind == 'SOURCE' and 'chưa xác minh' in discovery.description


def test_other_company_report_not_relabelled_and_ambiguous_report_unassigned():
    response = AssistantResponse(message='ok', tickers=['FPT', 'TCB'], metadata={'research_citations': [
        citation(tickers=['HPG']), citation(tickers=[], source_url='https://fpt.com.vn/unassigned.pdf')]})
    reports = [a for a in build_interaction_actions(response) if a.kind == 'REPORT']
    assert len(reports) == 1 and reports[0].ticker is None


@pytest.mark.parametrize('association', [{'ticker': 'TCB'}, {'symbol': 'TCB'}, {'tickers': 'TCB'}])
def test_singular_other_company_association_is_respected(association):
    report = citation(tickers=[])
    report.update(association)
    response = AssistantResponse(message='ok', tickers=['FPT'], metadata={'research_citations': [report]})
    assert not [a for a in build_interaction_actions(response) if a.kind == 'REPORT']


@pytest.mark.parametrize('period, expected', [
    ({'period_end': '2026-06-30', 'reporting_period': {}}, 'Dữ liệu đến kỳ 2026-06-30'),
    ({'reporting_period': {'end': '2026-06-30'}}, 'Dữ liệu đến kỳ 2026-06-30'),
    ({'reporting_period': {'label': 'Chưa xác định', 'status': 'UNKNOWN'}}, 'Chưa xác định'),
])
def test_period_end_fallback_preserves_explicit_unknown_labels(period, expected):
    response = AssistantResponse(message='ok', tickers=['FPT'], metadata={'research_citations': [citation(**period)]})
    report = next(a for a in build_interaction_actions(response) if a.kind == 'REPORT')
    assert report.period_label == expected


def test_legacy_provider_names_and_markdown_urls_are_not_parsed():
    response = AssistantResponse(message='https://example.com/invented.pdf', tickers=['FPT'],
        sources=['vnstock:VCI', '[report](https://example.com/invented.pdf)'])
    assert not [a for a in build_interaction_actions(response) if a.kind in {'REPORT', 'SOURCE'}]


def test_unsafe_and_malformed_metadata_do_not_break_ui():
    response = AssistantResponse(message='ok', tickers=['<bad>', 'FPT'], metadata={'research_citations': [
        None, 'text', citation(source_url='https://[bad/path'), citation(source_url='https://127.0.0.1/private'),
        citation(document_type=[], source_url='https://fpt.com.vn/info')], 'evidence_bundle': ['bad']})
    actions = build_interaction_actions(response)
    assert actions and not [a for a in actions if '127.0' in a.url or '[bad' in a.url]


def test_sources_use_source_reference_schema_not_fake_publication_time():
    response = AssistantResponse(message='ok', tickers=['FPT'], source_details=[SourceReference(
        provider='VCI', source_id='prices', data_kind=DataKind.OHLCV,
        source_url='https://example.com/prices/FPT', request_parameters={'ticker': 'FPT'},
        fetched_at=datetime(2026, 10, 18, tzinfo=timezone.utc))])
    source = next(a for a in build_interaction_actions(response) if a.kind == 'SOURCE')
    assert source.ticker == 'FPT' and source.published_at is None
    assert '2026-10-18' in source.description


@pytest.mark.parametrize('template', [
    'https://trade.example.com/?symbol={ticker}&side=BUY',
    'https://trade.example.com/?quantity=10', 'https://trade.example.com/?price=100',
    'https://trade.example.com/?action=submit', 'https://trade.example.com/?token=secret',
    'https://{ticker}.example.com/', 'https://trade.example.com/{ticker.__class__}',
    'https://trade.example.com/{ticker!r}', 'https://trade.example.com/{ticker:>10}',
    'https://trade.example.com/execute', 'https://trade.example.com/buy/{ticker}',
    'https://trade.example.com/#/order?quantity=10',
])
def test_broker_templates_cannot_be_order_execution_requests(template):
    with pytest.raises(ValueError):
        validate_broker_template(template)


def test_broker_navigation_never_claims_unverified_exact_ticker_selection():
    response = AssistantResponse(message='ok', tickers=['FPT', 'TCB'])
    assert not [a for a in build_interaction_actions(response) if a.kind == 'BROKER']
    home = [a for a in build_interaction_actions(response, broker_url_template='https://trade.vndirect.com.vn/', broker_name='VNDIRECT') if a.kind == 'BROKER']
    assert len(home) == 1 and home[0].ticker is None
    assert 'FPT' not in home[0].label and 'tự kiểm tra mã' in home[0].description.lower()
    exact = [a for a in build_interaction_actions(response, broker_url_template='https://trade.example.com/stocks/{ticker}') if a.kind == 'BROKER']
    assert {a.ticker for a in exact} == {'FPT', 'TCB'}
    assert all('không đặt lệnh' in a.description.lower() for a in exact)


def test_build_is_deterministic_and_never_opens_browser():
    response = AssistantResponse(message='ok', tickers=['FPT'], metadata={'research_citations': [citation()]})
    with patch('urllib.request.urlopen', side_effect=AssertionError('network')), patch('webbrowser.open', side_effect=AssertionError('browser')):
        assert build_interaction_actions(response) == build_interaction_actions(response)
