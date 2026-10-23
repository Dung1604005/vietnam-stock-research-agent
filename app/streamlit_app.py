from __future__ import annotations

from stock_multi_agent.config import load_workspace_env


load_workspace_env()

import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import time

from stock_multi_agent.chat.conversation import SQLiteConversationStore
from stock_multi_agent.models.chat import ChatRole
from stock_multi_agent.services.chat_service import ChatService
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.price_forecast import PriceForecastService
from stock_multi_agent.ui.components import render_assistant_message, render_user_message
from stock_multi_agent.ui.styles import APP_CSS
from stock_multi_agent.ui.agent_workspace import AGENTS, render_workspace
from stock_multi_agent.services.execution_profiles import execution_scope
from stock_multi_agent.services.progress_events import ProgressSink
from stock_multi_agent.ui.local_workspace_panel import render_local_workspace_panel
from stock_multi_agent.services.browser_automation import open_response_actions
from stock_multi_agent.services.navigation_request import navigation_response
from stock_multi_agent.models.chat import ConversationMessage


st.set_page_config(page_title="Agent Studio · Stock Research", page_icon="✦", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

SUGGESTIONS = [
    "Tôi nên mua cổ phiếu nào hôm nay?",
    "Phân tích FPT",
    "Thị trường Việt Nam hôm nay như thế nào?",
    "Tìm một số cổ phiếu đáng chú ý",
    "So sánh FPT và CMG",
    "Cho tôi biểu đồ VN-Index",
    "RSI FPT hiện tại như thế nào?",
    "P/E là gì?",
    "ATO khác ATC thế nào?",
]

# Increment when provider construction defaults change. Streamlit preserves
# session_state across source reloads, so this prevents an old provider object
# from retaining a bursty rate-limit policy after an application upgrade.
from stock_multi_agent.services.runtime_identity import SERVICE_RUNTIME_PROFILE


with st.sidebar:
    st.markdown('<div class="studio-brand"><b>✦</b><div>agent studio<small>STOCK RESEARCH TEAM</small></div></div>', unsafe_allow_html=True)
    selected_lstm = st.radio('Mô hình dự báo LSTM', ['v2', 'v4'], horizontal=True,
        format_func=lambda value: 'V2' if value == 'v2' else 'V4 · thử nghiệm',
        key='selected_lstm_version',
        help='Mô hình được chọn đóng góp 20% vào điểm Technical; chỉ báo kỹ thuật đóng góp 80%. Áp dụng cho yêu cầu mới.')
    st.caption('Technical = 80% chỉ báo + 20% tín hiệu LSTM. Không phải độ chính xác dự báo.')
    if selected_lstm == 'v4':
        st.info('V4 đang thử nghiệm, chưa qua forward validation. Bạn đang chọn dùng trọng số demo 20%.')


def initialize_session() -> tuple[ChatService, SQLiteConversationStore]:
    runtime_profile = f'{SERVICE_RUNTIME_PROFILE}:{selected_lstm}'
    if st.session_state.get("chat_service_runtime_profile") != runtime_profile:
        forecast_service = PriceForecastService.from_environment(default_enabled=True,
            model_version=selected_lstm, experimental_blend=selected_lstm == 'v4', defer_load=True)
        st.session_state.chat_service = ChatService(
            analysis_service=AnalysisService(price_forecast_service=forecast_service)
        )
        st.session_state.chat_service_runtime_profile = runtime_profile
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    store = SQLiteConversationStore(st.session_state.conversations)
    if not st.session_state.get("current_conversation_id"):
        conversation = next(iter(store.list()), None) or store.create()
        st.session_state.current_conversation_id = conversation.conversation_id
        st.session_state.conversations = store.items
    return st.session_state.chat_service, store


def new_chat(store: SQLiteConversationStore) -> None:
    conversation = store.create()
    st.session_state.current_conversation_id = conversation.conversation_id
    st.session_state.conversations = store.items


service, store = initialize_session()
if service.analysis.price_forecast_service.load_warning:
    st.sidebar.warning(service.analysis.price_forecast_service.load_warning)
current_id = st.session_state.current_conversation_id
conversation = store.get(current_id)

with st.sidebar:
    st.caption('KHÔNG GIAN CỦA BẠN')
    st.button("＋ New chat", use_container_width=True, on_click=new_chat, args=(store,))
    st.caption("CONVERSATIONS")
    for item in store.list():
        if st.button(item.title, key=f"conversation-{item.conversation_id}", use_container_width=True):
            st.session_state.current_conversation_id = item.conversation_id
            st.rerun()
    st.divider()
    if st.button("Delete current chat", use_container_width=True):
        store.delete(current_id)
        replacement = store.create()
        st.session_state.current_conversation_id = replacement.conversation_id
        st.session_state.conversations = store.items
        st.rerun()
    st.caption("Real vnstock data · No trade execution")
    execution_mode = st.selectbox('Chế độ chạy', ['STANDARD', 'QUICK', 'DEEP', 'OFFLINE'],
        help='STANDARD giữ cấu hình hiện hành. QUICK không gọi LLM. OFFLINE chỉ dùng KB/replay đã lưu, không gọi mạng.')
    offline_bundle_path = st.text_input('Replay bundle JSON trên ổ D', value='',
        disabled=execution_mode != 'OFFLINE', help='Tạo bằng scripts/capture_demo_bundle.py; không dùng snapshot chỉ có last bar.')
    render_local_workspace_panel(conversation.active_tickers, execution_mode)
    with st.expander('Tình trạng hệ thống — kiểm tra local', expanded=False):
        from stock_multi_agent.services.runtime_health import runtime_health
        provider = service.analysis.data_agent.market_tool.provider
        gateway = service.analysis.manager_agent.llm_provider
        st.json(runtime_health(provider=provider, gateway=gateway), expanded=False)
        st.caption('Không gọi API giá hoặc generate LLM để hiển thị bảng này.')
    library_open = st.checkbox("Thư viện báo cáo / Crawler", key="research_library_open")

if library_open:
    from stock_multi_agent.ui.research_library import render_research_library
    if execution_mode == 'OFFLINE':
        st.info('Offline: tác vụ crawl/ingest từ Internet bị tắt. Chuyển STANDARD nếu muốn cập nhật nguồn.')
        st.stop()
    with execution_scope(execution_mode):
        render_research_library(service.research)
    st.stop()

st.markdown('<div class="studio-page-head"><div><h1>Không gian nghiên cứu</h1>'
    '<p>Bạn đặt câu hỏi. Đội ngũ AI cùng tìm góc nhìn.</p></div>'
    '<span class="studio-page-badge">✦ &nbsp; 6 chuyên gia · 1 đội ngũ</span></div>', unsafe_allow_html=True)
studio_focus = st.segmented_control('Gặp đội ngũ của bạn', options=[agent.role for agent in AGENTS],
    default='manager', format_func=lambda role: next(f'{agent.symbol} {agent.name}' for agent in AGENTS if agent.role == role),
    key='studio_agent_focus') or 'manager'
workspace_view = st.empty()
latest_response = next((message.response for message in reversed(conversation.messages) if message.response), None)
render_workspace(latest_response.metadata.get('progress_events', []) if latest_response else [],
    tickers=conversation.active_tickers, mode=execution_mode, selected=studio_focus, target=workspace_view)

if not conversation.messages:
    st.markdown(
        '<div class="empty-state"><h1>Bắt đầu một ý tưởng nghiên cứu</h1>'
        '<p>Chọn gợi ý bên dưới hoặc giao nhiệm vụ cho đội ngũ của bạn.</p></div>',
        unsafe_allow_html=True,
    )
    suggestion_columns = st.columns(2)
    for index, suggestion in enumerate(SUGGESTIONS):
        if suggestion_columns[index % 2].button(suggestion, key=f"suggestion-{index}", use_container_width=True):
            st.session_state.pending_prompt = suggestion
            st.rerun()

for message in conversation.messages:
    if message.role == ChatRole.USER:
        render_user_message(message.content)
    elif message.response:
        render_assistant_message(message.response, service.charts)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)

prompt = st.chat_input("Giao nhiệm vụ cho đội ngũ… ví dụ: Phân tích FPT")
if st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    render_user_message(prompt)
    sink = ProgressSink()
    render_workspace([], tickers=conversation.active_tickers, mode=execution_mode,
        selected=studio_focus, target=workspace_view)
    def handle_request():
        # Opening a known page is a navigation task, not financial analysis:
        # no LLM or provider round trip is needed for these explicit requests.
        direct = navigation_response(prompt, conversation)
        if direct is not None:
            direct.metadata['execution_profile'] = {'mode': execution_mode, 'external_allowed': execution_mode != 'OFFLINE'}
            conversation.add_message(ConversationMessage(role=ChatRole.USER, content=prompt))
            conversation.add_message(ConversationMessage(role=ChatRole.ASSISTANT, content=direct.message, response=direct))
            if direct.tickers:
                conversation.active_tickers = direct.tickers
            if conversation.title == 'New conversation':
                conversation.title = prompt[:60]
            return direct
        return service.handle(prompt, conversation, execution_mode=execution_mode,
            progress_sink=sink, offline_bundle=offline_bundle_path.strip() or None)
    with st.status('Đang thực thi — hiển thị sự kiện từ tác vụ thực…', expanded=True) as status:
        live_progress = st.empty()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='stock-request') as executor:
            future = executor.submit(copy_context().run, handle_request)
            last_event_count = 0
            while not future.done():
                events = sink.snapshot()
                if events and len(events) != last_event_count:
                    last_event_count = len(events)
                    render_workspace(events, tickers=conversation.active_tickers, mode=execution_mode,
                        selected=studio_focus, running=True, target=workspace_view)
                    latest = {}
                    for event in events:
                        latest[event['node']] = event
                    live_progress.write('\n\n'.join(f'{event["node"]} · {event["status"]}'
                        for event in list(latest.values())[-8:]))
                time.sleep(0.15)
            try:
                response = future.result()
            except Exception as exc:
                render_workspace(sink.snapshot(), tickers=conversation.active_tickers, mode=execution_mode,
                    selected=studio_focus, target=workspace_view)
                st.error(f'Không hoàn thành request ({type(exc).__name__}). Không tự thay dữ liệu giả.')
                status.update(label='Request không hoàn thành', state='error')
                st.stop()
        failed = [event for event in sink.snapshot() if event['status'] in {'FAILED', 'PARTIAL'}]
        status.update(label='Đã trả kết quả có giới hạn/fallback' if failed else 'Đã hoàn thành tác vụ',
            state='complete', expanded=False)
    # Local OS navigation happens once after a new response, never while
    # rendering old conversation messages or creating analysis plans.
    context = getattr(st, 'context', None)
    browser_host = context.headers.get('Host') if context is not None else None
    response.metadata['browser_actions'] = open_response_actions(
        response, prompt, state=st.session_state, host=browser_host,
        execution_mode=execution_mode,
        enabled=st.session_state.get('interaction_auto_open', False),
        max_tabs=st.session_state.get('interaction_max_tabs', 3),
        broker_url_template=st.session_state.get('interaction_broker_template') or None,
        broker_name=st.session_state.get('interaction_broker_name') or 'VNDIRECT')
    store.save(conversation)
    st.session_state.conversations = store.items
    st.rerun()
