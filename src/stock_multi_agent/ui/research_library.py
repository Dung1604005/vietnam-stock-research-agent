"""Small local library review view, independent of a live market/LLM provider."""
from datetime import datetime

import streamlit as st

from stock_multi_agent.models.research import ResearchDocumentType
from stock_multi_agent.services.research_sync import ResearchSyncService


def render_research_library(store):
    sync = ResearchSyncService(store)
    st.subheader("Thư viện báo cáo")
    st.caption("URL tìm thấy trong chat và crawler được lưu tại đây. Duyệt tài liệu để đưa vào RAG.")
    health = sync.repository.health()
    st.write(health["candidate_counts"])
    selected_source = st.selectbox("Nguồn cập nhật", [None] + [source.source_id for source in sync.registry.sources if source.fetch_allowed],
                                   format_func=lambda value: value or "Tất cả nguồn")
    if st.button("Kiểm tra báo cáo mới"):
        with st.spinner("Đang kiểm tra các nguồn đến hạn…"):
            report = sync.sync(source_id=selected_source, seconds=60)
        st.json(report.model_dump(mode="json"))
    if st.button("Ingest các tài liệu đã duyệt"):
        with st.spinner("Đang cập nhật RAG…"):
            report = sync.sync(source_id=selected_source, queue_only=True, seconds=60)
        st.json(report.model_dump(mode="json"))
    status = st.selectbox("Trạng thái", ["QUARANTINED", "DISCOVERED", "RETRYABLE_ERROR", "INDEXED", "UNCHANGED", "PERMANENT_ERROR", "REJECTED"])
    items = sync.repository.list(status=status, limit=50)
    if not items:
        st.info("Chưa có tài liệu ở trạng thái này.")
    for item in items:
        with st.expander(item.title or str(item.source_url)):
            st.write(str(item.source_url))
            st.caption(f"Nguồn: {item.publisher} · Lý do: {item.reason or 'Đang chờ xử lý'} · Chính sách: {item.storage_mode.value}")
            st.caption(f"Dữ liệu đến kỳ: {item.reporting_period.label} · Ngày công bố: {item.published_at.isoformat() if item.published_at else 'Chưa xác định'}")
            if item.excerpt:
                st.text(item.excerpt)
            with st.form("review-" + item.candidate_id):
                title = st.text_input("Tiêu đề", item.title)
                tickers = st.text_input("Mã cổ phiếu (cách nhau bằng dấu phẩy)", ",".join(item.tickers))
                kinds = [kind.value for kind in ResearchDocumentType]
                kind = st.selectbox("Loại tài liệu", kinds, index=kinds.index(item.document_type.value))
                published = st.text_input("Ngày công bố từ nguồn (ISO-8601 có múi giờ, để trống nếu chưa biết)", item.published_at.isoformat() if item.published_at else "")
                note = st.text_input("Cơ sở cho phép sử dụng / ghi chú duyệt", item.approval_note)
                action = st.selectbox("Thao tác", ["approve", "retry", "reject"], format_func=lambda value: {"approve": "Duyệt lưu toàn văn tài liệu này", "retry": "Thử lại", "reject": "Loại"}[value])
                if st.form_submit_button("Lưu"):
                    try:
                        sync.repository.review(item.candidate_id, action, note=note, title=title,
                            tickers=[ticker.strip() for ticker in tickers.split(",") if ticker.strip()],
                            document_type=ResearchDocumentType(kind),
                            published_at=datetime.fromisoformat(published.replace("Z", "+00:00")) if published else None)
                        st.success("Đã lưu. Tài liệu được duyệt sẽ được xử lý ở lần đồng bộ tiếp theo.")
                    except ValueError as exc:
                        st.error(str(exc))
    with st.expander("Tình trạng cập nhật nguồn"):
        st.json(health["sources"])
