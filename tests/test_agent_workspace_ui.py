"""UI mapping and rendering only; no financial policy or tool execution."""
from stock_multi_agent.ui.agent_workspace import AGENTS, agent_states, avatar_svg, workspace_html, activity_by_role


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
    assert rendered.count('data-agent=') == 6


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
    renders = []
    monkeypatch.setattr(config, 'load_workspace_env', lambda: renders.append('render'))
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    from pathlib import Path
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app/streamlit_app.py'), default_timeout=25).run()
    assert not app.exception
    # HTML is mounted once in a shadow root, not repeatedly navigated as iframe srcdoc.
    assert not app.get('iframe')
    rendered = '\n'.join(item.proto.body for item in app.get('html'))
    assert 'attachShadow' in rendered
    assert 'Shared workspace' in rendered
    assert 'sessionStorage' in rendered
    assert not app.session_state['conversations'][app.session_state['current_conversation_id']].messages
    from stock_multi_agent.models.chat import AssistantResponse, ConversationMessage, ChatRole
    def handle(prompt, conversation, **kwargs):
        sink = kwargs['progress_sink']
        sink.emit('graph.technical', 'STARTED')
        sink.emit('graph.technical', 'SUCCEEDED')
        response = AssistantResponse(message='UI fixture complete', metadata={'progress_events': sink.snapshot()})
        conversation.add_message(ConversationMessage(role=ChatRole.USER, content=prompt))
        conversation.add_message(ConversationMessage(role=ChatRole.ASSISTANT, content=response.message, response=response))
        return response
    monkeypatch.setattr(app.session_state['chat_service'], 'handle', handle)
    before = len(renders)
    app.chat_input[0].set_value('UI fixture request').run()
    assert not app.exception
    assert len(renders) == before + 1  # No automatic full-page rerun on completion.
    updates = [item.proto.body for item in app.get('html')]
    assert sum('attachShadow' in body for body in updates) == 1
    assert any('CustomEvent' in body and 'attachShadow' not in body for body in updates)


def test_workspace_updates_send_data_only_and_skip_identical_snapshots():
    from stock_multi_agent.ui.agent_workspace import WorkspaceView
    class Slot:
        def __init__(self):
            self.calls = []
        def html(self, body, **kwargs):
            self.calls.append(body)
    slot = Slot()
    view = WorkspaceView(slot, 'test-studio')
    view.update('<section>first</section>')
    view.update('<section>first</section>')
    view.update('<section>second</section>')
    assert len(slot.calls) == 2
    assert all('CustomEvent' in call and 'attachShadow' not in call and 'iframe' not in call for call in slot.calls)
    view.update('</script><script>alert(1)</script>')
    assert '</script><script>' not in slot.calls[-1]


def test_task_history_keeps_attempts_and_distinguishes_received_and_sent_messages():
    events = [
        {'node': 'graph.risk', 'status': 'STARTED', 'attempt': 1},
        {'node': 'graph.risk', 'status': 'PARTIAL', 'attempt': 1, 'latency_ms': 1200},
        {'node': 'graph.risk', 'status': 'STARTED', 'attempt': 2},
    ]
    messages = [{'sender': 'manager', 'receiver': 'risk', 'message_type': 'TASK', 'task': 'Review evidence'}]
    activity = activity_by_role(events, messages)
    assert [item['status'] for item in activity['risk']['tasks']] == ['PARTIAL', 'STARTED']
    assert activity['technical']['tasks'] == []
    assert activity['risk']['messages'] == activity['manager']['messages'] == messages
    html = workspace_html(events, messages=messages, running=True)
    assert 'Đang làm (1)' in html and 'Đã ghi nhận (1)' in html
    assert 'Review evidence' in html
    assert 'Nova → Shield' in html


def test_nested_instrumentation_is_not_shown_as_duplicate_jobs():
    history = activity_by_role([
        {'node': 'graph.technical', 'status': 'STARTED', 'request_id': 'same'},
        {'node': 'TechnicalAgent.run', 'status': 'SUCCEEDED', 'request_id': 'same'},
        {'node': 'graph.technical', 'status': 'PARTIAL', 'request_id': 'same'},
    ])
    assert len(history['technical']['tasks']) == 1
    assert history['technical']['tasks'][0]['status'] == 'PARTIAL'


def test_handoff_animation_requires_actual_event_or_message_and_escapes_details():
    assert 'class="task-transfer ' not in workspace_html()
    messages = [{'sender': 'manager', 'receiver': 'technical', 'message_type': 'TASK',
                 'task': '<img src=x onerror=alert(1)>', 'payload': {'secret': 'NEVER_RENDER'}}]
    html = workspace_html(messages=messages)
    assert 'recorded-transfer' in html and 'live-transfer' not in html
    assert 'data-sender="manager" data-receiver="technical"' in html
    assert '&lt;img src=x onerror=alert(1)&gt;' in html
    assert '<img src=x' not in html and 'NEVER_RENDER' not in html
    live = workspace_html([{'node': 'graph.risk_after_technical', 'status': 'STARTED'}], running=True)
    assert 'live-transfer' in live and 'data-sender="workspace" data-receiver="risk"' in live
    assert 'data-sender="manager"' not in live  # Lifecycle is not proof of Manager messaging.
    finished = workspace_html([
        {'node': 'graph.risk', 'status': 'STARTED'},
        {'node': 'graph.risk', 'status': 'SUCCEEDED'},
    ], running=True)
    assert 'live-transfer' not in finished
