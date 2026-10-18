from datetime import date

import pytest

from stock_multi_agent.models.chat import AssistantResponse, ChatRole, ConversationMessage, ConversationState, ResponseTable
from stock_multi_agent.services.interaction_actions import build_interaction_actions
from stock_multi_agent.services.navigation_request import navigation_response


@pytest.mark.parametrize('prompt,kind,tickers', [
    ('Mở trang FPT', 'STOCK', ['FPT']),
    ('Mở giá TCB', 'QUOTE', ['TCB']),
    ('Mở biểu đồ FPT', 'CHART', ['FPT']),
    ('Mở DStock mã FPT', 'STOCK', ['FPT']),
    ('Mở VNDIRECT để mua FPT', 'BROKER', ['FPT']),
    ('MO WEB FPT', 'STOCK', ['FPT']),
    ('Hay mo bieu do TCB', 'CHART', ['TCB']),
    ('Mở trang FPT và TCB', 'STOCK', ['FPT', 'TCB']),
    ('Giúp tôi mở bảng giá TCB', 'QUOTE', ['TCB']),
    ('Mở biểu đồ Techcombank', 'CHART', ['TCB']),
    ('Mở trang PNG', 'STOCK', ['PNG']),
])
def test_navigation_is_fast_and_explicit(prompt, kind, tickers, monkeypatch):
    import socket
    import urllib.request
    import webbrowser
    def forbidden(*a, **k):
        raise AssertionError('Navigation parser must remain pure')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(webbrowser, 'open', forbidden)
    result = navigation_response(prompt, ConversationState())
    assert result.tickers == tickers
    assert result.metadata['navigation_request'] == {'kind': kind, 'automatic': True}
    assert not result.source_details


@pytest.mark.parametrize('prompt', [
    'Tôi nên mua mã nào hôm nay?', 'đặt lệnh mua FPT', 'Mở tài khoản VNDIRECT',
    'Phân tích FPT rồi mở biểu đồ', 'Mở biểu đồ FPT và phân tích kỹ thuật',
    'Mở https://example.com/FPT', 'Mở FPT; import os', 'Mở FPT\nprint(1)',
    'Mở javascript:alert(1)', 'Mở example.com FPT', 'Mở giá FPT và đánh giá lợi nhuận',
    'Mở rộng phân tích doanh thu FPT', 'Mở vị thế mua FPT',
    'Mở báo cáo FPT mới nhất', 'Mở báo cáo FPT năm 2024',
])
def test_does_not_intercept_research_advice_or_code(prompt):
    assert navigation_response(prompt, ConversationState()) is None


def test_unambiguous_context_and_broker_clarification():
    single = ConversationState(active_tickers=['FPT'])
    assert navigation_response('Mở giá', single).tickers == ['FPT']
    unclear = navigation_response('Mở bảng giá', ConversationState(active_tickers=['FPT', 'TCB']))
    assert unclear.metadata['navigation_clarification'] and not unclear.tickers
    multiple = navigation_response('Mở VNDIRECT để mua FPT và TCB', single)
    assert multiple.metadata['navigation_clarification'] and 'navigation_request' not in multiple.metadata


def context_with(response):
    return ConversationState(messages=[ConversationMessage(role=ChatRole.ASSISTANT,
                             content=response.message, response=response)])


def citation(ticker, year):
    return {'tickers': [ticker], 'canonical_url': f'https://example.com/{ticker}-{year}.pdf',
            'title': f'{ticker} annual report {year}', 'document_type': 'annual_report',
            'published_at': f'{year + 1}-03-10T10:00:00+00:00',
            'reporting_period': {'label': str(year)}}


def test_report_reuses_only_explicit_matching_sources_and_keeps_old_period():
    previous = AssistantResponse(message='Research', tickers=['FPT', 'TCB'],
        data_timestamp=date(2025, 3, 10), warnings=['Old report warning'],
        metadata={'research_citations': [citation('FPT', 2024), citation('TCB', 2023)],
                  'raw_prompt': 'do not copy', 'llm_output': 'do not copy'},
        tables=[ResponseTable(title='Reports', columns=['ticker'], rows=[
            {'ticker': 'FPT', 'value': 1}, {'ticker': 'TCB', 'value': 2}])])
    result = navigation_response('Mở báo cáo FPT', context_with(previous))
    assert result.metadata['navigation_request']['kind'] == 'REPORT'
    assert result.metadata['research_citations'] == [citation('FPT', 2024)]
    assert result.tables[0].rows == [{'ticker': 'FPT', 'value': 1}]
    assert result.data_timestamp == previous.data_timestamp
    assert result.warnings == previous.warnings
    assert 'raw_prompt' not in result.metadata and 'llm_output' not in result.metadata
    reports = [item for item in build_interaction_actions(result) if item.kind == 'REPORT']
    assert len(reports) == 1 and reports[0].period_label == '2024'
    assert 'không khẳng định' in result.message
    result.metadata['research_citations'][0]['title'] = 'changed'
    assert previous.metadata['research_citations'][0]['title'] != 'changed'


def test_missing_report_falls_through_even_when_other_ticker_has_report():
    previous = AssistantResponse(message='Research', tickers=['TCB'],
                                 metadata={'research_citations': [citation('TCB', 2024)]})
    assert navigation_response('Mở báo cáo FPT', context_with(previous)) is None
    assert navigation_response('Mở báo cáo FPT', ConversationState()) is None


def test_unscoped_multi_company_report_is_not_relabelled():
    previous = AssistantResponse(message='Research', tickers=['FPT', 'TCB'],
        metadata={'research_citations': [{'source_url': 'https://example.com/report.pdf'}]})
    assert navigation_response('Mở báo cáo FPT', context_with(previous)) is None


def test_web_discovery_is_not_verified_report():
    previous = AssistantResponse(message='Research', tickers=['FPT'], metadata={
        'evidence_bundle': {'web_citations': [citation('FPT', 2024)]}})
    assert navigation_response('Mở báo cáo FPT', context_with(previous)) is None


def test_single_company_legacy_pdf_url_is_preserved_not_invented():
    previous = AssistantResponse(message='Research', tickers=['FPT'], sources=['https://example.com/FPT.pdf'])
    result = navigation_response('Mở báo cáo FPT', context_with(previous))
    assert result is not None
    assert result.metadata['research_citations'][0]['source_url'] == previous.sources[0]
    assert 'published_at' not in result.metadata['research_citations'][0]


def test_latest_request_never_reuses_stale_context():
    previous = AssistantResponse(message='Research', tickers=['FPT'],
                                 metadata={'research_citations': [citation('FPT', 2024)]})
    assert navigation_response('Mở báo cáo mới nhất FPT', context_with(previous)) is None


def test_pdf_url_injection_does_not_become_navigation():
    previous = AssistantResponse(message='Research', tickers=['FPT'],
                                 sources=['javascript:evil.pdf', 'http://127.0.0.1/report.pdf'])
    assert navigation_response('Mở báo cáo FPT', context_with(previous)) is None
