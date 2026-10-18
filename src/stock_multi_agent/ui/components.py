from __future__ import annotations

import pandas as pd
import streamlit as st
import html

from stock_multi_agent.models.chat import AssistantResponse, ResponseType
from stock_multi_agent.services.chart_service import ChartService
from stock_multi_agent.ui.interaction_panel import render_interaction_actions


def render_user_message(content: str) -> None:
    with st.chat_message("user"):
        st.markdown(content)


def render_assistant_message(response: AssistantResponse, chart_service: ChartService) -> None:
    from uuid import uuid4
    render_key = response.metadata.get('request_metrics', {}).get('request_id') or response.metadata.setdefault('ui_render_id', uuid4().hex)
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response.message)
        profile = response.metadata.get('execution_profile', {})
        if profile:
            st.caption(f'Chế độ: {profile.get("mode", "STANDARD")} · Ngân sách là giới hạn, không phải cam kết tốc độ.')
        for origin in response.metadata.get('data_origins', []):
            st.caption(f'{origin.get("group")}: {origin.get("mode")} · dữ liệu đến {origin.get("as_of", "chưa rõ")}')
        if response.response_type == ResponseType.ANALYSIS:
            render_analysis_card(response)
        if response.response_type == ResponseType.MARKET_OVERVIEW:
            render_market_overview(response)
        for table in response.tables:
            if response.response_type == ResponseType.COMPARISON:
                render_comparison_table(table.title, table.rows, table.note)
            else:
                render_stock_table(table.title, table.rows, table.note)
        if response.charts:
            targets = st.tabs([f'{chart.request.ticker} · {chart.request.chart_type.value}' for chart in response.charts]) if len(response.charts) > 1 else [st.container()]
            for index, (target, chart) in enumerate(zip(targets, response.charts)):
                with target:
                    render_chart(chart_service.render_figure(chart), key=f'chart-{render_key}-{index}')
                    st.caption(f'{chart.explanation} Dữ liệu đến {chart.data_timestamp} · Nguồn: {chart.source}')
                    for warning in chart.warnings:
                        st.caption(warning)
        render_knowledge_citations(response)
        render_claim_evidence(response)
        render_data_freshness(response)
        render_response_warnings(response)
        render_semantic_debug(response)
        if response.agent_trace:
            render_agent_status(response)
            render_agent_trace(response)
        render_disclaimer(response)
        render_interaction_actions(response, render_key)
        if response.source_details or response.tables or response.metadata.get('research_brief'):
            from stock_multi_agent.services.research_workbench import ResearchWorkbench
            key = render_key
            first, second, third = st.columns(3)
            first.download_button('Tải báo cáo Markdown', ResearchWorkbench.export(response),
                file_name='research-report.md', mime='text/markdown', key=f'export-md-{key}')
            second.download_button('Tải báo cáo HTML', ResearchWorkbench.export(response, 'html'),
                file_name='research-report.html', mime='text/html', key=f'export-html-{key}')
            from stock_multi_agent.services.audit_export import export_audit_json
            third.download_button('Tải audit JSON đã lọc', export_audit_json(response),
                file_name='research-audit.json', mime='application/json', key=f'export-audit-{key}')
            st.caption('Export gồm dữ liệu phân tích, nguồn và audit đã lọc; không gồm khóa, raw prompt hoặc danh mục riêng mặc định. Kiểm tra trước khi chia sẻ.')


def render_claim_evidence(response):
    reviews = response.metadata.get('llm_specialist_reviews', [])
    if not reviews:
        return
    with st.expander('Đối chiếu nhận định → trích đoạn → nguồn', expanded=False):
        for review in reviews:
            st.caption(f'{review.get("agent")} · {review.get("status")} · {review.get("grounding_status", "NOT_AUDITED")}')
            for claim in review.get('claims', []):
                st.write(claim.get('claim_vi', ''))
                st.code(claim.get('exact_quote', ''), language=None)
                st.caption(f'Nguồn {claim.get("source_id")} · mã {claim.get("ticker")} · chỉ tiêu {claim.get("metric")} · kỳ {claim.get("period_end")} · đơn vị {claim.get("unit")}')
        st.caption('Đối chiếu anchor không chứng minh suy luận đúng hoặc bảo đảm hiệu quả đầu tư.')


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
    if "data_uncertainty_score" in metadata:
        items.append((
            "Data uncertainty",
            f"{metadata['data_uncertainty_score']:.1f}/10 · {metadata.get('data_uncertainty_level', '')}",
        ))
    if "recommendation" in metadata:
        items.append(("Decision", metadata["recommendation"]))
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


def render_chart(figure, key=None) -> None:
    st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False}, key=key)


def render_agent_status(response: AssistantResponse) -> None:
    reports = response.metadata.get('agent_reports', [])
    if reports:
        latest = {item['agent']: item for item in reports}
        st.markdown(' '.join(f'<span class="route-pill">● {html.escape(str(name).title())} · {html.escape(item["status"])}</span>'
            for name, item in latest.items()), unsafe_allow_html=True)
        return
    agents = []
    for event in response.agent_trace:
        name = event.agent.value.title()
        if name not in agents and name != "User":
            agents.append(name)
    if agents:
        st.caption('Vai trò có trace: ' + ', '.join(agents) + ' — xem trạng thái tác vụ bên dưới.')


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
            tool = f" · Tool: `{html.escape(event.tool)}`" if event.tool else ""
            st.markdown(
                f'<div class="trace-row"><span class="trace-agent">{event.agent.value.title()}</span> '
                f'· {html.escape(event.phase)} · {html.escape(event.action)}{tool}<br><small>{html.escape(event.detail)}</small></div>',
                unsafe_allow_html=True,
            )
        events = response.metadata.get('progress_events', [])
        if events:
            st.dataframe(pd.DataFrame([{key: event.get(key) for key in ('sequence', 'node', 'attempt', 'status', 'latency_ms')}
                for event in events]), hide_index=True, use_container_width=True)
        messages = response.metadata.get("agent_messages", [])
        if messages:
            st.markdown("##### Agent communication")
            for message in messages:
                st.caption(
                    f"{message['sender'].title()} → {message['receiver'].title()} · "
                    f"{message['type']} · {message['task']}"
                )
        evaluation = response.metadata.get("agent_evaluation")
        if evaluation:
            st.markdown("##### Agent contract evaluation")
            st.caption(
                f"Overall {evaluation['overall_score']:.0%} · "
                f"contracts {evaluation['contract_coverage']:.0%} · "
                f"outputs {evaluation['output_coverage']:.0%} · "
                f"evidence {evaluation['evidence_coverage']:.0%}"
            )
            reports = response.metadata.get("agent_reports", [])
            if reports:
                st.dataframe(pd.DataFrame([
                    {
                        "Agent": item["agent"],
                        "Stage": item["stage"],
                        "Status": item["status"],
                        "Evidence": item["evidence_count"],
                    }
                    for item in reports
                ]), hide_index=True, use_container_width=True)


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
                source_link = (
                    f"[Open source]({source_ref.source_url})"
                    if source_ref.source_url else "Direct URL not supplied"
                )
                st.markdown(
                    f"**{source_ref.data_kind.value}** · `{source_ref.source_id}`  \n"
                    f"As-of: `{as_of}` · fetched UTC: `{source_ref.fetched_at.isoformat()}` · "
                    f"schema: `{source_ref.schema_version}` · snapshot: `{checksum}` · {source_link}"
                )
    citations = response.metadata.get("research_citations", [])
    if citations:
        with st.expander("Research citations", expanded=False):
            for citation in citations:
                published = citation.get("published_at", "unknown")
                period = citation.get("reporting_period", {})
                period_label = period.get("label", "Chưa xác định")
                st.markdown(
                    f"**[{citation['title']}]({citation['source_url']})** · {citation['publisher']}  \n"
                    f"Dữ liệu đến kỳ: **{period_label}** · công bố: `{published}` · retrieval score: `{citation['retrieval_score']:.3f}` · "
                    f"chunk: `{citation['chunk_id']}`"
                )


def render_disclaimer(response: AssistantResponse) -> None:
    if response.response_type != ResponseType.TEXT:
        st.markdown(
            '<div class="small-disclaimer">Phân tích phục vụ học tập và minh họa hệ Multi-Agent, '
            'không phải khuyến nghị đầu tư hay lệnh giao dịch.</div>',
            unsafe_allow_html=True,
        )
