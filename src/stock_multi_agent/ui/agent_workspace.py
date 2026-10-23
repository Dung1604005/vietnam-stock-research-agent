"""Presentation-only agent studio. No providers, orchestration or state mutations."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
import base64
import re
import json
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class StudioAgent:
    role: str
    name: str
    title: str
    color: str
    symbol: str
    task: str


AGENTS = (
    StudioAgent('manager', 'Nova', 'Điều phối', '#8b71e8', '✦', 'Lập kế hoạch · kết nối chuyên gia · tổng hợp'),
    StudioAgent('data', 'Byte', 'Dữ liệu', '#55b6cf', '▤', 'Thu thập · chuẩn hóa · kiểm tra nguồn'),
    StudioAgent('research', 'Scout', 'Nghiên cứu', '#d9a34c', '⌕', 'Tìm tài liệu · kiểm tra kỳ · trích dẫn'),
    StudioAgent('technical', 'Pulse', 'Kỹ thuật', '#77b78b', '⌁', 'Đọc xu hướng · chỉ báo · tín hiệu giá'),
    StudioAgent('fundamental', 'Sage', 'Cơ bản', '#d586ad', '▥', 'Sức khỏe doanh nghiệp · tài chính · định giá'),
    StudioAgent('risk', 'Shield', 'Rủi ro', '#8297dc', '◇', 'Phản biện · kiểm tra rủi ro · bảo vệ kết luận'),
)


def avatar_svg(role: str) -> str:
    """Six original vector characters; role-specific silhouette/accessory."""
    agent = next((a for a in AGENTS if a.role == role), AGENTS[0])
    accessories = {
        'manager': '<path d="M41 29l5-12 10 8 10-8 5 12" fill="#ffe3a4" stroke="#bd9455" stroke-width="2"/>',
        'data': '<path d="M35 53H25v22h10M77 53h10v22H77" fill="#c8f3fa" stroke="#47899e" stroke-width="3"/>',
        'research': '<circle cx="77" cy="77" r="11" fill="#fff6dd" stroke="#a8792d" stroke-width="4"/><path d="M85 85l10 12" stroke="#a8792d" stroke-width="5" stroke-linecap="round"/>',
        'technical': '<path d="M40 95l10-8 9 4 12-15" fill="none" stroke="#d5ffe0" stroke-width="3"/>',
        'fundamental': '<path d="M30 87l25 6V75l-25-5zm50 0l-25 6V75l25-5z" fill="#ffe7f0" stroke="#b26c8c" stroke-width="2"/>',
        'risk': '<path d="M75 72l17 6v13q-3 12-17 18-14-6-17-18V78z" fill="#c9d8ff" stroke="#647bb7" stroke-width="2"/><path d="M68 89l5 5 10-12" fill="none" stroke="#647bb7" stroke-width="3"/>',
    }
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 112 120" role="img" aria-label="{agent.name} · {agent.role}">
    <ellipse cx="56" cy="109" rx="34" ry="7" fill="#282247" opacity=".1"/>
    <path d="M41 92v13m30-13v13" stroke="{agent.color}" stroke-width="11" stroke-linecap="round"/>
    <rect x="32" y="69" width="48" height="31" rx="14" fill="{agent.color}"/>
    <path d="M33 79l-8 11m54-11l8 11" stroke="{agent.color}" stroke-width="10" stroke-linecap="round"/>
    <path d="M56 35V23" stroke="{agent.color}" stroke-width="4"/><circle cx="56" cy="20" r="5" fill="{agent.color}"/>
    <rect x="23" y="35" width="66" height="43" rx="19" fill="{agent.color}"/>
    <rect x="29" y="38" width="54" height="33" rx="14" fill="#f9fbff"/>
    <rect x="35" y="46" width="42" height="19" rx="9" fill="#32364d"/>
    <path d="M44 53v5m23-5v5" stroke="#edfcff" stroke-width="4" stroke-linecap="round"/>
    <path d="M52 61q4 3 8 0" stroke="#b1b9cc" stroke-width="1.5" fill="none"/>
    <circle cx="35" cy="42" r="3" fill="white" opacity=".6"/>{accessories[agent.role]}</svg>'''


def avatar_uri(role: str) -> str:
    return 'data:image/svg+xml;base64,' + base64.b64encode(avatar_svg(role).encode()).decode()


def agent_states(events: list[dict] | None = None) -> dict[str, str]:
    """Only match role-owned nodes, never generic request completion to all roles."""
    states = {agent.role: 'IDLE' for agent in AGENTS}
    for event in events or []:
        node = str(event.get('node', '')).casefold()
        # risk_after_technical belongs to Risk, not both roles. Class-based and
        # graph-based instrumentation may nest; the latest observed owner wins.
        owner = re.split(r'[^a-z]+', node.removeprefix('graph.'))[0]
        for agent in AGENTS:
            if owner in {agent.role, agent.role + 'agent'}:
                status = event.get('status')
                if status in {'STARTED', 'SUCCEEDED', 'PARTIAL', 'FAILED', 'SKIPPED', 'FALLBACK'}:
                    states[agent.role] = status
    return states


def event_role(node):
    owner = re.split(r'[^a-z]+', str(node).casefold().removeprefix('graph.'))[0]
    return next((agent.role for agent in AGENTS if owner in {agent.role, agent.role + 'agent'}), None)


def task_title(node):
    stage = str(node).split('.')[-1]
    titles = {'manager_plan': 'Lập kế hoạch phân tích', 'plan': 'Lập kế hoạch phân tích',
        'data': 'Lấy và kiểm tra dữ liệu', 'research': 'Tìm bằng chứng doanh nghiệp',
        'technical': 'Phân tích xu hướng giá', 'fundamental': 'Đánh giá tài chính doanh nghiệp',
        'risk': 'Đánh giá rủi ro và phản biện', 'manager_review': 'Đối chiếu kết quả chuyên gia',
        'review': 'Đối chiếu bằng chứng', 'finalize': 'Tổng hợp kết luận',
        'technical_reanalysis': 'Phân tích lại kỹ thuật', 'data_reanalysis': 'Làm mới dữ liệu',
        'research_reanalysis': 'Mở rộng nghiên cứu', 'risk_after_technical': 'Kiểm tra rủi ro sau phân tích lại',
        'fundamental_after_research': 'Đánh giá lại sau bổ sung bằng chứng',
        'risk_after_research': 'Đối chiếu rủi ro với bằng chứng mới',
        'technical_after_data': 'Tính lại kỹ thuật trên dữ liệu mới',
        'fundamental_after_data': 'Tính lại tài chính trên dữ liệu mới',
        'risk_after_data': 'Kiểm tra rủi ro trên dữ liệu mới', 'research_after_data': 'Nghiên cứu trên dữ liệu mới'}
    if stage == 'run':
        role = event_role(node)
        return {'data': 'Lấy và kiểm tra dữ liệu', 'research': 'Tìm bằng chứng doanh nghiệp',
            'technical': 'Tính chỉ báo và phân tích giá', 'fundamental': 'Đánh giá tài chính doanh nghiệp',
            'risk': 'Đo rủi ro và phản biện'}.get(role, 'Thực hiện tác vụ')
    return titles.get(stage, stage.replace('_', ' '))


def activity_by_role(events=None, messages=None, reports=None, trace=None):
    """Read-only projection of existing telemetry; no invented task completion."""
    activity = {agent.role: {'tasks': [], 'messages': [], 'reports': [], 'trace': []} for agent in AGENTS}
    latest = {}
    for event in events or []:
        role = event_role(event.get('node', ''))
        if not role:
            continue
        identity = (event.get('request_id', ''), event['node'], event.get('attempt', 1))
        latest[identity] = {**event, 'role': role, 'title': task_title(event['node'])}
    for event in latest.values():
        activity[event['role']]['tasks'].append(event)
    for detail in activity.values():
        # Outer graph node and inner Agent.run describe one operation, not two jobs.
        graph_requests = {item.get('request_id', '') for item in detail['tasks'] if item['node'].startswith('graph.')}
        detail['tasks'] = [item for item in detail['tasks'] if item['node'].startswith('graph.')
            or item.get('request_id', '') not in graph_requests]
    for message in messages or []:
        message = message.model_dump(mode='json') if hasattr(message, 'model_dump') else message
        for role in {message.get('sender'), message.get('receiver')}:
            if role in activity:
                activity[role]['messages'].append(message)
    for report in reports or []:
        report = report.model_dump(mode='json') if hasattr(report, 'model_dump') else report
        if report.get('agent') in activity:
            activity[report['agent']]['reports'].append(report)
    for event in trace or []:
        event = event.model_dump(mode='json') if hasattr(event, 'model_dump') else event
        if event.get('agent') in activity:
            activity[event['agent']]['trace'].append(event)
    return activity


def _time(value):
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).astimezone(
            timezone(timedelta(hours=7))).strftime('%H:%M:%S')
    except (ValueError, TypeError):
        return '—'


def inspector_html(activity, states, running):
    panels = []
    for agent in AGENTS:
        detail = activity[agent.role]
        current, past = [], []
        for item in reversed(detail['tasks']):
            status = item.get('status', 'PENDING')
            label = STATUS_LABELS.get(status, str(status))
            if status == 'STARTED' and not running:
                label = STATUS_LABELS['PENDING']
            latency = item.get('latency_ms')
            duration = f' · {latency / 1000:.2f}s' if isinstance(latency, (int, float)) else ''
            row = f'<li class="task-row"><span class="task-dot state-{escape(str(status).lower())}"></span><div><strong>{escape(item["title"])}</strong><span>{escape(label)} · Lượt {escape(str(item.get("attempt", 1)))} · {_time(item.get("timestamp"))}{duration}</span><small>{escape(str(item["node"]))}</small></div></li>'
            (current if status == 'STARTED' and running else past).append(row)
        messages = []
        for message in reversed(detail['messages'][-100:]):
            sender, receiver = str(message.get('sender', '?')), str(message.get('receiver', '?'))
            kind = str(message.get('message_type', 'MESSAGE'))
            names = {a.role: a.name for a in AGENTS}
            messages.append(f'<li class="task-row"><span class="message-icon">↗</span><div><strong>{escape(str(message.get("task", "")))}</strong><span>{escape(names.get(sender, sender))} → {escape(names.get(receiver, receiver))} · {escape(kind)} · {_time(message.get("timestamp"))}</span></div></li>')
        traces = ''.join(f'<li class="task-row"><span class="task-dot"></span><div><strong>{escape(str(item.get("action", "")))}</strong><span>{escape(str(item.get("detail", "")))}</span><small>{escape(str(item.get("tool") or item.get("phase", "")))} · {_time(item.get("timestamp"))}</small></div></li>' for item in reversed(detail['trace'][-100:]))
        report = detail['reports'][-1] if detail['reports'] else None
        summary = f'<div class="agent-summary"><strong>Kết quả báo cáo gần nhất</strong><p>{escape(str(report.get("summary", "")))}</p></div>' if report else ''
        panels.append(f'''<section class="agent-detail" data-detail="{agent.role}" hidden>
        <header><img src="{avatar_uri(agent.role)}" alt=""/><div><h3>{agent.name} <span>{agent.role.title()} · {agent.title}</span></h3><p>{agent.task}</p></div><button type="button" class="replay-task">↗ Xem lại giao task</button></header>
        <div class="detail-tabs" role="group" aria-label="Lọc công việc"><button type="button" data-tab="current">Đang làm ({len(current)})</button><button type="button" data-tab="history">Đã ghi nhận ({len(past)})</button><button type="button" data-tab="messages">Task trao đổi ({len(messages)})</button><button type="button" data-tab="trace">Chi tiết hành động ({len(detail['trace'])})</button></div>
        <div class="task-content" data-content="current"><ul>{''.join(current) or '<li class="empty-tasks">Chưa có tác vụ đang chạy được ghi nhận cho agent này.</li>'}</ul></div>
        <div class="task-content" data-content="history" hidden>{summary}<ul>{''.join(past[:100]) or '<li class="empty-tasks">Chưa có lịch sử tác vụ. Giao yêu cầu để bắt đầu.</li>'}</ul></div>
        <div class="task-content" data-content="messages" hidden><ul>{''.join(messages) or '<li class="empty-tasks">Chưa có message trao đổi được cung cấp. Message đầy đủ hiển thị sau khi request trả kết quả.</li>'}</ul></div>
        <div class="task-content" data-content="trace" hidden><ul>{traces or '<li class="empty-tasks">Chưa có trace hành động chi tiết. Sự kiện tác vụ vẫn cập nhật ở các tab bên cạnh.</li>'}</ul></div></section>''')
    return '<div class="studio-details">' + ''.join(panels) + '</div>'


def handoffs_html(events, messages, running):
    positions = {'manager': (500, 145), 'data': (180, 225), 'research': (820, 225),
                 'technical': (180, 465), 'fundamental': (500, 465), 'risk': (820, 465)}
    transfers = []
    if running:
        for receiver, detail in activity_by_role(events).items():
            for event in detail['tasks']:
                if event.get('status') == 'STARTED':
                    # Lifecycle proves a task began, not that Manager sent a message.
                    transfers.append(('workspace', receiver, task_title(event['node']), 'Bắt đầu tác vụ'))
    else:
        for message in messages or []:
            message = message.model_dump(mode='json') if hasattr(message, 'model_dump') else message
            sender, receiver = message.get('sender'), message.get('receiver')
            if sender in positions and receiver in positions and sender != receiver:
                transfers.append((sender, receiver, str(message.get('task', '')), str(message.get('message_type', 'MESSAGE'))))
    edges = []
    for index, (sender, receiver, task, kind) in enumerate(transfers[-6:] if running else transfers[-60:]):
        x1, y1 = positions.get(sender, (500, 320))
        x2, y2 = positions[receiver]
        curve = f'M{x1},{y1} Q500,320 {x2},{y2}'
        edges.append(f'''<g class="task-transfer {'live-transfer' if running else 'recorded-transfer'}" data-sender="{sender}" data-receiver="{receiver}"><title>{escape(kind + ': ' + task)}</title><path class="transfer-route" d="{curve}" marker-end="url(#task-arrow)"/><g class="task-packet"><rect x="-10" y="-7" width="20" height="14" rx="4"/><path d="M-7,-4L0,1 7,-4"/><animateMotion dur="1.6s" begin="{index * .15}s" repeatCount="2" path="{curve}"/></g><circle class="receive-ring" cx="{x2}" cy="{y2}" r="23"/></g>''')
    caption = 'Task bắt đầu · cập nhật từ sự kiện thực tế' if running and transfers else 'Bấm avatar để xem công việc · chọn “Xem lại giao task” để phát lại message' if transfers else 'Bấm trực tiếp vào avatar để xem công việc'
    return f'<svg class="studio-handoffs" viewBox="0 0 1000 620" preserveAspectRatio="none" aria-label="Luồng trao đổi task"><defs><marker id="task-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5 0 10z" fill="#8b71e8"/></marker></defs>{"".join(edges)}</svg><div class="studio-handoff-caption">{caption}</div>'


STATUS_LABELS = {'IDLE': 'Chưa được gọi', 'STARTED': 'Đang làm việc', 'SUCCEEDED': 'Đã xử lý',
                 'PARTIAL': 'Kết quả một phần', 'FAILED': 'Gặp lỗi', 'SKIPPED': 'Đã bỏ qua', 'FALLBACK': 'Dùng fallback',
                 'PENDING': 'Chưa ghi nhận kết thúc'}


def workspace_html(events=None, *, tickers=(), mode='STANDARD', selected='manager', running=False,
                   messages=None, reports=None, trace=None) -> str:
    states = agent_states(events)
    if not running:
        states = {role: 'PENDING' if status == 'STARTED' else status for role, status in states.items()}
    active = sum(status == 'STARTED' for status in states.values())
    observed = sum(status != 'IDLE' for status in states.values())
    label = 'Đang phối hợp' if active else 'Đã có hoạt động' if observed else 'Sẵn sàng nhận nhiệm vụ'
    cards = []
    activity = activity_by_role(events, messages, reports, trace)
    for index, agent in enumerate(AGENTS):
        status = states[agent.role]
        current = next((item['title'] for item in reversed(activity[agent.role]['tasks']) if item.get('status') == 'STARTED'), '') if status == 'STARTED' else ''
        cards.append(f'''<button type="button" class="studio-station station-{index} {'working' if status == 'STARTED' else ''} {'selected' if selected == agent.role else ''}" data-agent="{agent.role}" aria-label="Xem công việc của {agent.name}, {agent.title}" aria-pressed="{'true' if selected == agent.role else 'false'}" style="--agent:{agent.color}">
        <div class="station-tag">{agent.symbol} {agent.title}</div>
        <div class="station-bubble">{escape(current) if current else 'Bấm để xem công việc'}<span class="typing-dots">•••</span></div>
        <div class="station-scene"><div class="station-monitor"><span></span><i></i><b></b></div>
        <img class="studio-avatar" src="{avatar_uri(agent.role)}" alt="{agent.name}, {agent.title}"/>
        <div class="station-desk"></div></div>
        <div class="station-name">{agent.name} <span>{agent.role.title()}</span></div>
        <div class="station-status state-{status.lower()}"><i></i>{STATUS_LABELS[status]}</div></button>''')
    context = ', '.join(str(ticker) for ticker in tickers) or 'Chưa chọn mã cổ phiếu'
    return f'''<section class="agent-studio" aria-label="Không gian cộng tác của sáu AI agent">
    <div class="studio-toolbar"><div><span class="studio-live-dot"></span> AGENT WORKSPACE <span class="studio-divider">/</span> Phòng nghiên cứu</div><span class="studio-mode">{escape(str(mode))}</span></div>
    <div class="studio-heading"><div><h2>Một đội ngũ. Nhiều góc nhìn.</h2><p>Sáu chuyên gia cùng kết nối dữ liệu thành nhận định.</p></div><span class="studio-team-state">{label}</span></div>
    <div class="studio-floor"><div class="studio-window window-one"></div><div class="studio-window window-two"></div><div class="studio-wall-label">RESEARCH LAB <span>01</span></div>
    <div class="studio-rug"></div><svg class="studio-connections" viewBox="0 0 1000 620" preserveAspectRatio="none" aria-hidden="true"><path d="M180 225Q500 240 500 320M500 145V320M820 225Q500 240 500 320M180 465Q500 380 500 320M500 465V320M820 465Q500 380 500 320"/></svg>
    {handoffs_html(events, messages, running)}
    <div class="studio-table"><span>✦</span><strong>Shared workspace</strong><small>{escape(context)}</small></div>
    {''.join(cards)}<div class="studio-plant plant-left">✳</div><div class="studio-plant plant-right">✳</div>
    <div class="studio-floor-note">{active} đang làm việc <span>·</span> {observed}/6 vai trò có sự kiện</div></div>
    {inspector_html(activity, states, running)}
    <div class="studio-footnote">Trạng thái và lịch sử từ request đang xem · Task bắt đầu lấy từ lifecycle; trao đổi lấy từ message đã ghi nhận · Giờ hiển thị UTC+7</div></section>'''


def _script_json(value):
    return json.dumps(value, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


class WorkspaceView:
    """One mounted scene; subsequent deltas carry data only, never a new document."""
    def __init__(self, slot, identity):
        self.slot, self.identity = slot, identity
        self.previous = None

    def update(self, markup):
        if markup == self.previous:
            return
        self.previous = markup
        channel = _script_json(self.identity)
        data = _script_json(markup)
        self.slot.html(f'''<script>(() => {{
            const id={channel}, markup={data};
            window.__stockStudioPending ??= {{}};
            window.__stockStudioPending[id]=markup;
            window.dispatchEvent(new CustomEvent(id, {{detail:markup}}));
        }})();</script>''', unsafe_allow_javascript=True)


def render_workspace(events=None, *, tickers=(), mode='STANDARD', selected='manager', running=False, target=None,
                     messages=None, reports=None, trace=None):
    import streamlit as st
    from pathlib import Path
    from uuid import uuid4
    from stock_multi_agent.ui.styles import APP_CSS
    folder = Path(__file__).parent
    css = (folder / 'agent_workspace.css').read_text(encoding='utf-8')
    js = (folder / 'agent_workspace.js').read_text(encoding='utf-8')
    markup = workspace_html(events, tickers=tickers, mode=mode, selected=selected, running=running,
        messages=messages, reports=reports, trace=trace)
    if isinstance(target, WorkspaceView):
        target.update(markup)
        return target
    identity = 'stock-studio-' + uuid4().hex
    scoped_css = APP_CSS.split('<style>', 1)[1].split('</style>', 1)[0] + css
    # Stable shadow DOM isolates CSS without an iframe navigation/white flash.
    # Only trusted local JS is executable; dynamic text is escaped HTML + JSON.
    boot = f'''<div id="{identity}" style="height:1170px;overflow:auto;overflow-anchor:none"></div>
    <script>(() => {{
        const host=document.getElementById({_script_json(identity)});
        const root=host.attachShadow({{mode:'open'}});
        const style=document.createElement('style');
        style.textContent={_script_json(scoped_css)};
        root.appendChild(style);
        const container=document.createElement('div');
        container.innerHTML=window.__stockStudioPending?.[{_script_json(identity)}] || {_script_json(markup)};
        root.appendChild(container);
        const studioDocument={{querySelector:s=>root.querySelector(s),querySelectorAll:s=>root.querySelectorAll(s),
            body: {{dataset: {{}}}} }};
        studioDocument.body.dataset.defaultAgent={_script_json(selected)};
        (function(document, studioRoot, channel) {{ {js} }})(studioDocument, root, {_script_json(identity)});
    }})();</script>'''
    with (target.container() if target is not None else st.container()):
        st.html(boot, unsafe_allow_javascript=True)
        view = WorkspaceView(st.empty(), identity)
        view.previous = markup
    return view
