"""Presentation-only agent studio. No providers, orchestration or state mutations."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
import base64
import re


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


STATUS_LABELS = {'IDLE': 'Chưa được gọi', 'STARTED': 'Đang làm việc', 'SUCCEEDED': 'Đã xử lý',
                 'PARTIAL': 'Kết quả một phần', 'FAILED': 'Gặp lỗi', 'SKIPPED': 'Đã bỏ qua', 'FALLBACK': 'Dùng fallback',
                 'PENDING': 'Chưa ghi nhận kết thúc'}


def workspace_html(events=None, *, tickers=(), mode='STANDARD', selected='manager', running=False) -> str:
    states = agent_states(events)
    if not running:
        states = {role: 'PENDING' if status == 'STARTED' else status for role, status in states.items()}
    active = sum(status == 'STARTED' for status in states.values())
    observed = sum(status != 'IDLE' for status in states.values())
    label = 'Đang phối hợp' if active else 'Đã có hoạt động' if observed else 'Sẵn sàng nhận nhiệm vụ'
    cards = []
    for index, agent in enumerate(AGENTS):
        status = states[agent.role]
        cards.append(f'''<div class="studio-station station-{index} {'working' if status == 'STARTED' else ''} {'selected' if selected == agent.role else ''}" style="--agent:{agent.color}">
        <div class="station-tag">{agent.symbol} {agent.title}</div>
        <div class="station-scene"><div class="station-monitor"><span></span><i></i><b></b></div>
        <img class="studio-avatar" src="{avatar_uri(agent.role)}" alt="{agent.name}, {agent.title}"/>
        <div class="station-desk"></div></div>
        <div class="station-name">{agent.name} <span>{agent.role.title()}</span></div>
        <div class="station-status state-{status.lower()}"><i></i>{STATUS_LABELS[status]}</div></div>''')
    chosen = next((a for a in AGENTS if a.role == selected), AGENTS[0])
    context = ', '.join(str(ticker) for ticker in tickers) or 'Chưa chọn mã cổ phiếu'
    return f'''<section class="agent-studio" aria-label="Không gian cộng tác của sáu AI agent">
    <div class="studio-toolbar"><div><span class="studio-live-dot"></span> AGENT WORKSPACE <span class="studio-divider">/</span> Phòng nghiên cứu</div><span class="studio-mode">{escape(str(mode))}</span></div>
    <div class="studio-heading"><div><h2>Một đội ngũ. Nhiều góc nhìn.</h2><p>Sáu chuyên gia cùng kết nối dữ liệu thành nhận định.</p></div><span class="studio-team-state">{label}</span></div>
    <div class="studio-floor"><div class="studio-window window-one"></div><div class="studio-window window-two"></div><div class="studio-wall-label">RESEARCH LAB <span>01</span></div>
    <div class="studio-rug"></div><svg class="studio-connections" viewBox="0 0 1000 480" preserveAspectRatio="none" aria-hidden="true"><path d="M170 145Q500 190 500 240M500 135V240M830 145Q500 190 500 240M170 360Q500 290 500 240M500 360V240M830 360Q500 290 500 240"/></svg>
    <div class="studio-table"><span>✦</span><strong>Shared workspace</strong><small>{escape(context)}</small></div>
    {''.join(cards)}<div class="studio-plant plant-left">✳</div><div class="studio-plant plant-right">✳</div>
    <div class="studio-floor-note">{active} đang làm việc <span>·</span> {observed}/6 vai trò có sự kiện</div></div>
    <div class="studio-inspector"><img src="{avatar_uri(chosen.role)}" alt=""/><div><strong>{chosen.name} <span>/ {chosen.role.title()}</span></strong><p>{chosen.task}</p></div><span class="studio-inspector-status">{STATUS_LABELS[states[chosen.role]]}</span></div>
    <div class="studio-footnote">Avatar minh họa vai trò · Trạng thái từ sự kiện thực thi thực tế · Không phải sáu mô hình riêng biệt</div></section>'''


def render_workspace(events=None, *, tickers=(), mode='STANDARD', selected='manager', running=False, target=None):
    import streamlit as st
    (target if target is not None else st).markdown(
        workspace_html(events, tickers=tickers, mode=mode, selected=selected, running=running), unsafe_allow_html=True)
