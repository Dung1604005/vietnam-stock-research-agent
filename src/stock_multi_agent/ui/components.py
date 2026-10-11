from __future__ import annotations

import pandas as pd
import streamlit as st

from stock_multi_agent.models.chat import AssistantResponse, ResponseType
from stock_multi_agent.services.chart_service import ChartService


def render_user_message(content: str) -> None:
    with st.chat_message("user"):
        st.markdown(content)


def render_assistant_message(response: AssistantResponse, chart_service: ChartService) -> None:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response.message)
        if response.response_type == ResponseType.ANALYSIS:
            render_analysis_card(response)
        if response.response_type == ResponseType.MARKET_OVERVIEW:
            render_market_overview(response)
        for table in response.tables:
            if response.response_type == ResponseType.COMPARISON:
                render_comparison_table(table.title, table.rows, table.note)
            else:
                render_stock_table(table.title, table.rows, table.note)
        for chart in response.charts:
            render_chart(chart_service.render_figure(chart))
        render_knowledge_citations(response)
        render_data_freshness(response)
        render_response_warnings(response)
        render_semantic_debug(response)
        if response.agent_trace:
            render_agent_status(response)
            render_agent_trace(response)
        render_disclaimer(response)


def render_response_warnings(response: AssistantResponse) -> None:
    if not response.warnings:
        return
    if response.response_type == ResponseType.ANALYSIS:
        # Keep diagnostics available for audit without dominating the answer.
        with st.expander(f"Chẩn đoán dữ liệu/mô hình ({len(response.warnings)})", expanded=False):
            for warning in response.warnings:
                st.caption(f"• {warning}")
        return
    for warning in response.warnings:
        st.warning(warning, icon="⚠️")


def render_knowledge_citations(response: AssistantResponse) -> None:
    citations = response.metadata.get("knowledge_citations", [])
    if not citations:
        return
    st.caption("Evidence origin: Local KB · deterministic retrieval")
    with st.expander("Knowledge citations", expanded=False):
        for citation in citations:
            published = citation.get("published_at") or "not supplied by source"
            effective = citation.get("effective_at") or "not supplied by source"
            st.markdown(
                f"**{citation['title']}** · {citation['publisher']}  \n"
                f"[{citation['source_url']}]({citation['source_url']})  \n"
                f"Published: `{published}` · Effective: `{effective}` · "
                f"Retrieved: `{citation['retrieved_at']}` · Verification: `{citation['verification_status']}`"
            )
            if citation.get("freshness_warning"):
                st.warning(citation["freshness_warning"])


def render_semantic_debug(response: AssistantResponse) -> None:
    semantic = response.metadata.get("semantic_understanding")
    if not semantic:
        return
    with st.expander("View semantic interpretation", expanded=False):
        first, second = st.columns(2)
        first.markdown(f"**Intent:** `{semantic['intent']}`")
        first.markdown(f"**Tickers:** `{', '.join(semantic['tickers']) or '—'}`")
        first.markdown(f"**Period:** `{semantic.get('period') or '—'}`")
        second.markdown(f"**Confidence:** `{semantic['confidence']:.0%}`")
        second.markdown(f"**Parser:** `{semantic['parser_source']}`")
        second.caption(f"Latency: {semantic.get('parser_latency_ms', 0)} ms · fallback: {semantic.get('fallback_reason') or '—'}")
        secondary = ", ".join(semantic.get("secondary_intents", [])) or "—"
        second.markdown(f"**Secondary:** `{secondary}`")
        st.caption(semantic.get("interpretation", ""))
        if semantic.get("resolved_context"):
            st.markdown("**Resolved context**")
            st.json(semantic["resolved_context"], expanded=False)
        if semantic.get("ambiguities"):
            st.warning(" · ".join(semantic["ambiguities"]))


def render_analysis_card(response: AssistantResponse) -> None:
    metadata = response.metadata
    items: list[tuple[str, str]] = []
    if "technical_score" in metadata:
        items.append(("Technical", f"{metadata['technical_score']:.1f}/10 · {metadata.get('technical_signal', '')}"))
    if "fundamental_score" in metadata:
        items.append(("Fundamental", f"{metadata['fundamental_score']:.1f}/10 · {metadata.get('fundamental_signal', '')}"))
    if "risk_score" in metadata:
        items.append(("Risk", f"{metadata['risk_score']:.1f}/10 · {metadata.get('risk_level', '')}"))
    if "recommendation" in metadata:
        items.append(("Decision", f"{metadata['recommendation']} · {metadata.get('confidence', 0):.0%}"))
    if not items:
        return
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.markdown(
            f'<div class="score-card"><div class="score-label">{label}</div><div class="score-value">{value}</div></div>',
            unsafe_allow_html=True,
        )


def render_market_overview(response: AssistantResponse) -> None:
    metadata = response.metadata
    if not metadata:
        return
    first, second, third = st.columns(3)
    first.metric("VN-Index", f"{metadata.get('index', 0):,.2f}", f"{metadata.get('day_change', 0):+.2%}")
    second.metric("20 phiên", f"{metadata.get('change_20', 0):+.2%}")
    third.metric("Volume / TB20", f"{metadata.get('volume_ratio', 0):.2f}x")


def render_stock_table(title: str, rows: list[dict], note: str | None = None) -> None:
    st.markdown(f"#### {title}")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if note:
        st.caption(note)


def render_comparison_table(title: str, rows: list[dict], note: str | None = None) -> None:
    render_stock_table(title, rows, note)


def render_chart(figure) -> None:
    st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})


def render_agent_status(response: AssistantResponse) -> None:
    agents = []
    for event in response.agent_trace:
        name = event.agent.value.title()
        if name not in agents and name != "User":
            agents.append(name)
    if agents:
        st.markdown(" ".join(f'<span class="route-pill">● {name} · Done</span>' for name in agents), unsafe_allow_html=True)


def render_agent_trace(response: AssistantResponse) -> None:
    with st.expander("View agent activity", expanded=False):
        route = response.metadata.get("routing_plan", [])
        if route:
            st.caption("Route: " + " → ".join(route))
        task_plan = response.metadata.get("task_plan", [])
        if task_plan:
            st.caption(f"Task budget: {response.metadata.get('task_plan_budget_units', 0)} units")
            st.json(task_plan, expanded=False)
        for event in response.agent_trace:
            tool = f" · Tool: `{event.tool}`" if event.tool else ""
            st.markdown(
                f'<div class="trace-row"><span class="trace-agent">{event.agent.value.title()}</span> '
                f'· {event.phase} · {event.action}{tool}<br><small>{event.detail}</small></div>',
                unsafe_allow_html=True,
            )
        messages = response.metadata.get("agent_messages", [])
        if messages:
            st.markdown("##### Agent communication")
            for message in messages:
                st.caption(
                    f"{message['sender'].title()} → {message['receiver'].title()} · "
                    f"{message['type']} · {message['task']}"
                )


def render_data_freshness(response: AssistantResponse) -> None:
    if response.data_timestamp or response.data_fetched_at:
        source = ", ".join(response.sources) if response.sources else "unknown"
        as_of = response.data_timestamp.strftime("%d/%m/%Y") if response.data_timestamp else "unknown"
        fetched = response.data_fetched_at.astimezone().strftime("%d/%m/%Y %H:%M:%S %Z") if response.data_fetched_at else "unknown"
        st.markdown(
            f'<div class="freshness">Data as of: {as_of} · Fetched: {fetched} · Source: {source}</div>',
            unsafe_allow_html=True,
        )
    if response.source_details:
        with st.expander("Data provenance", expanded=False):
            for source_ref in response.source_details:
                as_of = source_ref.as_of.isoformat() if source_ref.as_of else "not supplied"
                checksum = source_ref.content_sha256[:12] if source_ref.content_sha256 else "not captured"
                st.markdown(
                    f"**{source_ref.data_kind.value}** · `{source_ref.source_id}`  \n"
                    f"As-of: `{as_of}` · fetched UTC: `{source_ref.fetched_at.isoformat()}` · "
                    f"schema: `{source_ref.schema_version}` · snapshot: `{checksum}`"
                )


def render_disclaimer(response: AssistantResponse) -> None:
    if response.response_type != ResponseType.TEXT:
        st.markdown(
            '<div class="small-disclaimer">Phân tích phục vụ học tập và minh họa hệ Multi-Agent, '
            'không phải khuyến nghị đầu tư hay lệnh giao dịch.</div>',
            unsafe_allow_html=True,
        )
