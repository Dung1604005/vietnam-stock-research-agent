"""UI mapping and rendering only; no financial policy or tool execution."""
from stock_multi_agent.ui.agent_workspace import AGENTS, agent_states, avatar_svg, workspace_html


def test_studio_only_marks_observed_roles_and_respects_latest_attempt():
    states = agent_states([
        {'node': 'graph.technical', 'status': 'SUCCEEDED'},
        {'node': 'graph.risk_after_technical', 'status': 'STARTED'},
        {'node': 'ChatService.handle', 'status': 'SUCCEEDED'},
        {'node': 'TechnicalAgent.run', 'status': 'STARTED'},
        {'node': 'TechnicalAgent.run', 'status': 'PARTIAL'},
    ])
    assert states['risk'] == 'STARTED'
    assert states['technical'] == 'PARTIAL'
    assert states['fundamental'] == states['manager'] == 'IDLE'
    assert agent_states()["data"] == 'IDLE'


def test_workspace_safely_renders_context_and_six_distinct_characters():
    originals = {avatar_svg(agent.role) for agent in AGENTS}
    assert len(originals) == 6
    rendered = workspace_html(tickers=['<script>alert(1)</script>'], mode='<img onerror=alert(1)>')
    assert '<script>' not in rendered and '<img onerror' not in rendered
    assert '&lt;script&gt;' in rendered
    assert rendered.count('class="studio-avatar"') == 6
    assert '0/6 vai trò có sự kiện' in rendered
    assert 'working' not in rendered


def test_live_risk_event_does_not_animate_technical():
    states = agent_states([{'node': 'graph.risk_after_technical', 'status': 'STARTED'}])
    assert states['technical'] == 'IDLE'
    assert states['risk'] == 'STARTED'
    events = [{'node': 'graph.risk', 'status': 'STARTED'}]
    assert 'Chưa ghi nhận kết thúc' in workspace_html(events)
    assert 'Đang làm việc' not in workspace_html(events)
    assert 'Đang làm việc' in workspace_html(events, running=True)


def test_real_app_renders_and_switches_avatar_without_external_calls(monkeypatch):
    import urllib.request
    from streamlit.testing.v1 import AppTest

    def forbidden(*args, **kwargs):
        raise AssertionError('UI inspection must not call external data or LLM')

    monkeypatch.setenv('STOCK_AGENT_KNOWLEDGE_GRAPH_ENABLED', 'false')
    monkeypatch.setenv('STOCK_AGENT_LLM_PROVIDER', 'mock')
    monkeypatch.setenv('STOCK_AGENT_LSTM_ENABLED', 'false')
    # The real entry point loads private .env defaults. Keep this UI smoke test
    # hermetic so those defaults cannot leak into subsequent backend tests.
    import stock_multi_agent.config as config
    monkeypatch.setattr(config, 'load_workspace_env', lambda: None)
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    from pathlib import Path
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app/streamlit_app.py'), default_timeout=25).run()
    assert not app.exception
    assert any('Shared workspace' in item.value for item in app.markdown)
    app.segmented_control(key='studio_agent_focus').set_value('risk').run()
    assert not app.exception
    assert any('<strong>Shield <span>/ Risk</span>' in item.value for item in app.markdown)
    assert not app.session_state['conversations'][app.session_state['current_conversation_id']].messages
