"""No real browser launches. Exercise local-only policy and response dedup."""
from types import SimpleNamespace

import pytest

from stock_multi_agent.models.chat import AssistantResponse
from stock_multi_agent.services.browser_automation import is_local_browser_host, open_response_actions


@pytest.mark.parametrize('host', ['localhost:8501', '127.0.0.1:8501', '[::1]:8501', 'localhost'])
def test_loopback_browser_hosts(host):
    assert is_local_browser_host(host)


@pytest.mark.parametrize('host', [None, '', '192.168.1.5:8501', 'example.com', 'localhost.evil.com',
    'user@localhost:8501', 'localhost:8501/path', 'localhost\n.evil'])
def test_remote_or_malformed_hosts_do_not_launch(host):
    opened = []
    result = open_response_actions(AssistantResponse(message='FPT', tickers=['FPT']), 'Phân tích FPT',
        state={}, host=host, opener=lambda url: opened.append(url) or True)
    assert result['status'] == 'NOT_LOCALHOST' and not opened


def test_single_ticker_default_opens_three_categories_once():
    response = AssistantResponse(message='FPT', tickers=['FPT'])
    state, opened = {}, []
    result = open_response_actions(response, 'Phân tích FPT', state=state, host='localhost:8501',
        opener=lambda url: opened.append(url) or True)
    assert result['status'] == 'OPENED' and len(opened) == 3
    assert opened[0] == 'https://dstock.vndirect.com.vn/suc-khoe-co-phieu/FPT'
    assert {x['kind'] for x in result['opened']} == {'STOCK', 'QUOTE', 'CHART'}
    duplicate = open_response_actions(response, 'Phân tích FPT', state=state, host='localhost:8501',
        opener=lambda url: opened.append(url) or True)
    assert duplicate['status'] == 'ALREADY_HANDLED' and len(opened) == 3


def test_explicit_broker_request_opens_portal_and_dstock_no_trade_params():
    response = AssistantResponse(message='navigation', tickers=['TCB'],
        metadata={'navigation_request': {'kind': 'BROKER'}})
    opened = []
    result = open_response_actions(response, 'Mở VNDIRECT để mua TCB', state={}, host='127.0.0.1:8501',
        opener=lambda url: opened.append(url) or True)
    assert opened == ['https://trade.vndirect.com.vn/', 'https://dstock.vndirect.com.vn/suc-khoe-co-phieu/TCB']
    assert result['opened'][0]['ticker'] is None  # Do not claim preselected order ticker.


def test_buy_recommendation_does_not_open_broker():
    opened = []
    open_response_actions(AssistantResponse(message='BUY', tickers=['FPT']), 'Tôi nên mua mã nào hôm nay?',
        state={}, host='localhost', opener=lambda url: opened.append(url) or True)
    assert not any('trade.vndirect' in url for url in opened)


def test_verified_report_opens_exact_pdf_including_encoded_spaces():
    url = 'https://fpt.com.vn/reports/Annual%20Report.pdf'
    response = AssistantResponse(message='Research', tickers=['FPT'], metadata={'research_citations': [
        {'tickers': ['FPT'], 'source_url': url, 'title': 'Annual report', 'document_type': 'annual_report'}]})
    opened = []
    result = open_response_actions(response, 'Mở báo cáo FPT', state={}, host='localhost',
        opener=lambda target: opened.append(target) or True)
    assert opened == [url] and result['status'] == 'OPENED'


def test_unknown_report_domain_is_not_auto_opened():
    response = AssistantResponse(message='Research', tickers=['FPT'], metadata={'research_citations': [
        {'tickers': ['FPT'], 'source_url': 'https://untrusted-report.example/annual.pdf', 'title': 'Unknown', 'document_type': 'annual_report'}]})
    opened = []
    result = open_response_actions(response, 'Mở báo cáo FPT', state={}, host='localhost',
        opener=lambda target: opened.append(target) or True)
    assert not opened and result['status'] == 'NO_SAFE_ACTIONS'
    assert result['skipped'][0]['reason'] == 'UNTRUSTED_AUTOMATIC_DESTINATION'


@pytest.mark.parametrize('mode,metadata,enabled,status', [
    ('OFFLINE', {}, True, 'OFFLINE_BLOCKED'),
    ('STANDARD', {'execution_profile': {'mode': 'OFFLINE'}}, True, 'OFFLINE_BLOCKED'),
    ('STANDARD', {}, False, 'DISABLED'),
])
def test_offline_and_disabled_never_open(mode, metadata, enabled, status):
    response = AssistantResponse(message='fixture', tickers=['FPT'], metadata=metadata)
    def forbidden(url): raise AssertionError('No browser calls allowed')
    assert open_response_actions(response, 'Phân tích FPT', state={}, host='localhost',
        execution_mode=mode, enabled=enabled, opener=forbidden)['status'] == status


def test_list_tab_limit_and_failed_browser_no_automatic_retry():
    response = AssistantResponse(message='list', tickers=['FPT', 'TCB', 'VNM', 'HPG', 'VCB'])
    state, attempts = {}, []
    result = open_response_actions(response, 'Mã đáng chú ý', state=state, host='localhost', max_tabs=2,
        opener=lambda target: attempts.append(target) or False)
    assert len(attempts) == 2 and result['status'] == 'BROWSER_UNAVAILABLE'
    assert result['remaining_count'] > 0
    result = open_response_actions(response, 'Mã đáng chú ý', state=state, host='localhost',
        opener=lambda target: attempts.append(target) or True)
    assert result['status'] == 'ALREADY_HANDLED' and len(attempts) == 2


def test_new_response_can_open_same_ticker_again():
    state, opened = {}, []
    for _ in range(2):
        response = AssistantResponse(message='navigation', tickers=['FPT'], metadata={'navigation_request': {'kind': 'STOCK'}})
        open_response_actions(response, 'Mở trang FPT', state=state, host='localhost',
            opener=lambda target: opened.append(target) or True)
    assert len(opened) == 2


def test_full_app_navigation_does_not_call_provider_and_does_not_repeat_on_rerun(monkeypatch):
    from pathlib import Path
    import stock_multi_agent.config as config
    import stock_multi_agent.services.browser_automation as browser_module
    from streamlit.testing.v1 import AppTest
    import urllib.request
    import webbrowser
    monkeypatch.setattr(config, 'load_workspace_env', lambda: None)
    monkeypatch.setenv('STOCK_AGENT_LLM_PROVIDER', 'none')
    monkeypatch.setenv('STOCK_AGENT_LSTM_ENABLED', 'false')
    def forbidden(*args, **kwargs): raise AssertionError('No provider/browser I/O in UI test')
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(webbrowser, 'open', forbidden)
    opened = []
    actual = browser_module.open_response_actions
    def local_test_hook(response, text, **kwargs):
        # AppTest does not originate a browser request and supplies no Host.
        kwargs.update(host='localhost:8501', opener=lambda url: opened.append(url) or True)
        return actual(response, text, **kwargs)
    monkeypatch.setattr(browser_module, 'open_response_actions', local_test_hook)
    path = Path(__file__).resolve().parents[1]/'app/streamlit_app.py'
    app = AppTest.from_file(str(path), default_timeout=30).run()
    assert not app.exception
    app.chat_input[0].set_value('Mở trang FPT').run()
    assert not app.exception
    assert opened == ['https://dstock.vndirect.com.vn/suc-khoe-co-phieu/FPT']
    responses = [m.response for c in app.session_state['conversations'].values() for m in c.messages if m.response]
    assert len(responses) == 1 and responses[0].metadata['browser_actions']['status'] == 'OPENED'
    app.run()
    assert not app.exception and len(opened) == 1
