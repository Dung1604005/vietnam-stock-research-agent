"""Personal sidebar interactions; no browser execution or trade execution here."""

from __future__ import annotations

import streamlit as st

from stock_multi_agent.services.interaction_actions import actions_for_ticker, validate_broker_template
from stock_multi_agent.services.local_workspace import LocalWorkspace, WorkspaceError, normalize_ticker


_DEFAULT_BROKER_NAME = "VNDIRECT"
_DEFAULT_BROKER_URL = "https://trade.vndirect.com.vn/"


def _queue(prompt: str) -> None:
    st.session_state.pending_prompt = prompt
    st.rerun()


def render_local_workspace_panel(active_tickers: list[str], execution_mode: str) -> None:
    """Render inside sidebar, after the existing execution-mode/replay widgets."""
    offline = execution_mode == "OFFLINE"
    # Migrate sessions created when automatic browsing was enabled by default.
    # A fresh explicit checkbox choice is required under the consent policy.
    if st.session_state.get('interaction_consent_version') != 'opt-in-v2':
        st.session_state['interaction_auto_open'] = False
        st.session_state['interaction_consent_version'] = 'opt-in-v2'
    st.session_state.interaction_execution_mode = execution_mode
    workspace = LocalWorkspace()
    try:
        watchlist = workspace.list_tickers()
        preferences = workspace.get_preferences()
    except WorkspaceError as exc:
        st.error(str(exc))
        st.session_state.interaction_broker_name = ""
        st.session_state.interaction_broker_template = ""
        return
    broker_name = preferences.get("broker_name", _DEFAULT_BROKER_NAME)
    broker_template = preferences.get("broker_url_template", _DEFAULT_BROKER_URL)
    try:
        validate_broker_template(broker_template)
    except ValueError:
        st.warning("Liên kết công ty chứng khoán đã lưu không an toàn; hãy lưu cấu hình mới.")
        broker_name, broker_template = "", ""
    st.session_state.interaction_broker_name = broker_name
    st.session_state.interaction_broker_template = broker_template
    with st.expander("Tương tác trực tiếp / Theo dõi cá nhân", expanded=False):
        st.checkbox("Tự mở trang trên máy này", value=False, key="interaction_auto_open", disabled=offline,
                    help="Chỉ bật khi bạn đồng ý cho yêu cầu mới tự mở tab. Nếu không bật, yêu cầu mở trang cần xác nhận riêng. Không đăng nhập hoặc đặt lệnh thay bạn.")
        st.slider("Số tab tự mở tối đa", min_value=1, max_value=5, value=3,
                  key="interaction_max_tabs", disabled=offline)
        st.caption("Tự mở trang đang bật." if st.session_state.interaction_auto_open and not offline else
                   "Tự mở trang đang tắt; bạn có thể mở liên kết thủ công khi online.")
        if offline:
            st.caption("OFFLINE: không mở website hoặc gọi tìm kiếm Internet. Ghi chú/theo dõi vẫn hoạt động.")
        picked = st.selectbox("Mã trong danh sách theo dõi", [""] + watchlist,
                              format_func=lambda item: item or "— Chọn mã —", key="interaction_watchlist_pick")
        fallback = next((item for item in active_tickers if isinstance(item, str) and item), "FPT")
        ticker_text = st.text_input("Mã cổ phiếu tương tác", value=fallback, key="interaction_ticker")
        try:
            ticker = normalize_ticker(picked or ticker_text)
        except WorkspaceError as exc:
            st.warning(str(exc))
            ticker = None
        if picked:
            st.caption(f"Đang thao tác với {picked} từ danh sách theo dõi.")
        add, remove = st.columns(2)
        if add.button("Thêm theo dõi", key="interaction_watchlist_add", disabled=ticker is None):
            try:
                workspace.add_ticker(ticker)
                st.rerun()
            except WorkspaceError as exc:
                st.error(str(exc))
        if remove.button("Bỏ theo dõi", key="interaction_watchlist_remove", disabled=ticker not in watchlist):
            try:
                workspace.remove_ticker(ticker)
                st.rerun()
            except WorkspaceError as exc:
                st.error(str(exc))
        if ticker:
            try:
                note = workspace.get_note(ticker)
            except WorkspaceError as exc:
                st.error(str(exc))
                note = ""
            note_text = st.text_area(f"Ghi chú cá nhân · {ticker}", value=note, max_chars=10_000,
                                     key=f"interaction_note_{ticker}")
            if st.button("Lưu ghi chú", key="interaction_note_save"):
                try:
                    workspace.save_note(ticker, note_text)
                    st.success("Đã lưu ghi chú trên máy này.")
                except WorkspaceError as exc:
                    st.error(str(exc))
            first, second = st.columns(2)
            if first.button("Phân tích mã", key="interaction_analyze"):
                _queue(f"Phân tích {ticker}")
            if second.button("Vẽ biểu đồ", key="interaction_chart"):
                _queue(f"Cho tôi biểu đồ {ticker}")
            if st.button("Tìm báo cáo mới nhất", key="interaction_report", disabled=offline):
                _queue(f"Tìm báo cáo mới nhất của {ticker}")
            if not offline:
                st.caption("Liên kết mở thủ công — không bảo đảm giá realtime hoặc mã còn niêm yết.")
                for action in actions_for_ticker(ticker):
                    st.link_button(action.label, action.url)
        comparison = st.multiselect("Mã muốn so sánh (2–5 mã)", watchlist,
                                   default=watchlist[:2], max_selections=5, key="interaction_compare_tickers")
        if st.button("So sánh các mã đã chọn", key="interaction_compare", disabled=len(comparison) < 2):
            _queue("So sánh " + " và ".join(comparison))
        st.divider()
        st.caption("Cổng giao dịch chỉ để mở website. Không lưu tài khoản, mật khẩu hay gửi lệnh.")
        use_custom = st.checkbox("Dùng liên kết công ty chứng khoán tùy chỉnh",
                                 value=broker_template not in ("", _DEFAULT_BROKER_URL), key="interaction_broker_custom")
        custom_name = st.text_input("Tên công ty chứng khoán", value=broker_name or _DEFAULT_BROKER_NAME,
                                    key="interaction_broker_name_input", disabled=not use_custom)
        custom_url = st.text_input("URL HTTPS; có thể dùng {ticker}", value=broker_template or _DEFAULT_BROKER_URL,
                                   key="interaction_broker_url_input", disabled=not use_custom)
        if st.button("Lưu cổng giao dịch", key="interaction_broker_save"):
            name = custom_name if use_custom else _DEFAULT_BROKER_NAME
            template = custom_url if use_custom else _DEFAULT_BROKER_URL
            try:
                validate_broker_template(template)
                workspace.save_preferences({"broker_name": name, "broker_url_template": template})
                st.session_state.interaction_broker_name = name
                st.session_state.interaction_broker_template = template
                st.success("Đã lưu liên kết. Bạn tự kiểm tra mã và xác nhận mọi thao tác trên website.")
            except (ValueError, WorkspaceError) as exc:
                st.error(str(exc))
        st.caption("VNDIRECT mặc định mở trang chủ giao dịch, không giả định đã chọn đúng mã. "
                   "Chỉ URL deep link do bạn xác minh mới có thể điều hướng theo {ticker}.")
