"""Isolated Streamlit UI checks: no market, model, browser, or broker calls."""
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from stock_multi_agent.models.chat import AssistantResponse


def action(identity, ticker, kind, *, description=None):
    return SimpleNamespace(id=identity, ticker=ticker, kind=kind, label=f'{kind} {identity}',
                           url=f'https://example.com/{identity}', description=description,
                           published_at='2026-10-18', period_label='Q3/2026')


SCRIPT = '''
import streamlit as st
from stock_multi_agent.models.chat import AssistantResponse
from stock_multi_agent.ui.interaction_panel import render_interaction_actions
response = st.session_state.get('fixture_response', AssistantResponse(message='UI fixture'))
render_interaction_actions(response, 'stable-response')
'''


@pytest.fixture
def ui(monkeypatch):
    import socket
    import urllib.request
    from stock_multi_agent.ui import interaction_panel

    def forbidden(*args, **kwargs):
        raise AssertionError('Interactive UI must not invoke external services or open a browser')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    import webbrowser
    monkeypatch.setattr(webbrowser, 'open', forbidden)
    calls = []
    actions = [action('fpt-stock', 'FPT', 'STOCK'), action('fpt-price', 'FPT', 'QUOTE'),
               action('fpt-report', 'FPT', 'REPORT', description='Chưa xác minh; dữ liệu cũ'),
               action('fpt-broker', 'FPT', 'BROKER'), action('tcb-stock', 'TCB', 'STOCK'),
               action('tcb-report', 'TCB', 'REPORT')]

    def build(response, **kwargs):
        calls.append(kwargs)
        return actions
    monkeypatch.setattr(interaction_panel, 'build_interaction_actions', build)
    return AppTest.from_string(SCRIPT, default_timeout=10), calls, actions


def link_labels(app):
    return [item.proto.label for item in app.get('link_button')]


def test_switch_ticker_never_reuses_other_ticker_report(ui):
    app, calls, _ = ui
    app.run()
    assert not app.exception
    assert 'REPORT fpt-report' in link_labels(app)
    assert 'REPORT tcb-report' not in link_labels(app)
    assert any('2026-10-18' in item.value and 'Q3/2026' in item.value for item in app.caption)
    assert any('Chưa xác minh; dữ liệu cũ' in item.value for item in app.caption)
    app.selectbox[0].select('TCB').run()
    assert not app.exception
    assert 'REPORT tcb-report' in link_labels(app)
    assert 'REPORT fpt-report' not in link_labels(app)
    assert len(calls) == 2


def test_followup_requires_click_and_keeps_selected_ticker(ui):
    app, _, _ = ui
    app.run()
    assert 'pending_prompt' not in app.session_state
    app.selectbox[0].select('TCB').run()
    next(item for item in app.button if item.label == 'Tìm báo cáo mã').click().run()
    assert app.session_state['pending_prompt'] == 'Tìm báo cáo tài chính mới nhất của TCB'
    assert not app.exception


def test_broker_link_disabled_until_manual_confirmation(ui):
    app, _, _ = ui
    app.session_state['interaction_broker_url_template'] = 'https://example.com/trade/{ticker}'
    app.run()
    broker = next(item for item in app.get('link_button') if item.proto.label == 'BROKER fpt-broker')
    assert broker.proto.disabled
    app.checkbox[0].check().run()
    broker = next(item for item in app.get('link_button') if item.proto.label == 'BROKER fpt-broker')
    assert not broker.proto.disabled
    assert 'pending_prompt' not in app.session_state


def test_navigation_cancel_is_per_response_and_never_opens_browser(ui):
    app, _, _ = ui
    response = AssistantResponse(message='Mở trang FPT', tickers=['FPT'], metadata={
        'navigation_request': {'kind': 'STOCK'},
        'browser_actions': {'status': 'CONFIRMATION_REQUIRED', 'request_text': 'Mở trang FPT'}})
    app.session_state['fixture_response'] = response
    app.run()
    assert any(button.label == 'Đồng ý mở trang' for button in app.button)
    next(button for button in app.button if button.label == 'Không mở').click().run()
    assert not app.exception
    assert response.metadata['browser_actions']['status'] == 'CANCELLED'
    app.run()
    assert not any(button.label == 'Đồng ý mở trang' for button in app.button)


@pytest.mark.parametrize('current_offline', [False, True])
def test_offline_has_no_outbound_actions_and_does_not_build_links(ui, current_offline):
    app, calls, _ = ui
    if current_offline:
        app.session_state['interaction_execution_mode'] = 'OFFLINE'
    else:
        app.session_state['fixture_response'] = AssistantResponse(
            message='Offline fixture', metadata={'execution_profile': {'mode': 'OFFLINE'}})
    app.run()
    assert not app.exception and not app.get('link_button') and not app.button
    assert not calls
    assert any('OFFLINE' in item.value for item in app.caption)


def test_sources_without_ticker_and_document_pagination(ui):
    app, _, actions = ui
    actions[:] = [action(f'report-{index}', None, 'REPORT') for index in range(11)]
    app.run()
    assert not app.exception and not app.selectbox
    assert len(app.get('link_button')) == 8
    app.number_input[0].set_value(2).run()
    assert not app.exception
    assert len(app.get('link_button')) == 3
    assert 'REPORT report-10' in link_labels(app)


def test_unique_widgets_when_multiple_messages_render(ui):
    _, _, _ = ui
    app = AppTest.from_string(SCRIPT + "\nrender_interaction_actions(response, 'second-response')\n")
    app.run()
    assert not app.exception
    assert len(app.selectbox) == 2


def test_unscoped_report_not_claimed_as_selected_ticker(ui):
    app, _, actions = ui
    actions.append(action('unscoped-report', None, 'REPORT'))
    app.run()
    assert not app.exception
    assert 'REPORT unscoped-report' not in link_labels(app)
    assert any('Nguồn chưa gắn mã' in item.value for item in app.caption)


def test_static_broker_visible_with_selected_ticker_and_actual_sidebar_key(ui):
    app, calls, actions = ui
    actions[:] = [item for item in actions if item.kind != 'BROKER']
    actions.append(action('vndirect-portal', None, 'BROKER'))
    app.session_state['interaction_broker_template'] = 'https://trade.vndirect.com.vn/'
    app.run()
    assert not app.exception
    assert calls[0]['broker_url_template'] == 'https://trade.vndirect.com.vn/'
    assert 'BROKER vndirect-portal' in link_labels(app)
    assert app.checkbox[0].label.endswith('cần tự chọn mã FPT)')
    broker = next(item for item in app.get('link_button') if item.proto.label == 'BROKER vndirect-portal')
    assert broker.proto.disabled


def test_opened_status_displays_metadata_without_opening_browser(ui):
    app, _, _ = ui
    app.session_state['fixture_response'] = AssistantResponse(message='UI fixture', metadata={
        'browser_actions': {'status': 'OPENED', 'opened': [{'label': 'FPT quote'}]}})
    app.run()
    assert not app.exception
    assert any('Đã mở: FPT quote' in item.value for item in app.caption)


def test_legacy_streamlit_link_button_without_key(ui, monkeypatch):
    from stock_multi_agent.ui import interaction_panel
    original = interaction_panel.st.link_button
    def legacy(label, url, *, help=None, disabled=False, use_container_width=None):
        return original(label, url, help=help, disabled=disabled, use_container_width=use_container_width)
    monkeypatch.setattr(interaction_panel.st, 'link_button', legacy)
    app, _, _ = ui
    app.run()
    assert not app.exception and app.get('link_button')
