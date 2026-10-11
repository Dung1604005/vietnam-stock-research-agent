from __future__ import annotations

import streamlit as st

from stock_multi_agent.chat.conversation import InMemoryConversationStore
from stock_multi_agent.models.chat import ChatRole
from stock_multi_agent.services.chat_service import ChatService
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.price_forecast import PriceForecastService
from stock_multi_agent.ui.components import render_assistant_message, render_user_message
from stock_multi_agent.ui.styles import APP_CSS


st.set_page_config(page_title="Stock Multi-Agent Assistant", page_icon="🤖", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

SUGGESTIONS = [
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
SERVICE_RUNTIME_PROFILE = "free-vnstock-guard-lstm-v2-kb-v1"


def initialize_session() -> tuple[ChatService, InMemoryConversationStore]:
    if st.session_state.get("chat_service_runtime_profile") != SERVICE_RUNTIME_PROFILE:
        forecast_service = PriceForecastService.from_environment(default_enabled=True)
        st.session_state.chat_service = ChatService(
            analysis_service=AnalysisService(price_forecast_service=forecast_service)
        )
        st.session_state.chat_service_runtime_profile = SERVICE_RUNTIME_PROFILE
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    store = InMemoryConversationStore(st.session_state.conversations)
    if not st.session_state.get("current_conversation_id"):
        conversation = store.create()
        st.session_state.current_conversation_id = conversation.conversation_id
        st.session_state.conversations = store.items
    return st.session_state.chat_service, store


def new_chat(store: InMemoryConversationStore) -> None:
    conversation = store.create()
    st.session_state.current_conversation_id = conversation.conversation_id
    st.session_state.conversations = store.items


service, store = initialize_session()
current_id = st.session_state.current_conversation_id
conversation = store.get(current_id)

with st.sidebar:
    st.markdown('<div class="app-kicker">🤖 Stock Multi-Agent</div>', unsafe_allow_html=True)
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

st.markdown('<div class="app-kicker">Stock Multi-Agent Assistant</div>', unsafe_allow_html=True)

if not conversation.messages:
    st.markdown(
        '<div class="empty-state"><h1>Hôm nay bạn muốn phân tích gì?</h1>'
        '<p>Trò chuyện tự nhiên với hệ thống 5 Agent chuyên biệt.</p></div>',
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

prompt = st.chat_input("Message Stock Agent…")
if st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    render_user_message(prompt)
    parsed = service.parser.parse(prompt, conversation)
    plan = service.router.plan(parsed, conversation)
    with st.status("Analyzing with specialized agents…", expanded=True) as status:
        for step in plan.steps:
            labels = {
                "DATA": "Data Agent is fetching market data…",
                "TECHNICAL": "Technical Agent is selecting indicators…",
                "FUNDAMENTAL": "Fundamental Agent is analyzing financials…",
                "RISK": "Risk Agent is reviewing uncertainty…",
                "MANAGER": "Manager is synthesizing the response…",
                "CHART": "Chart service is rendering selected data…",
                "MARKET": "Manager is calculating market movement…",
                "SCREENING": "Manager is screening the configured watchlist…",
                "COMPARISON": "Manager is coordinating comparable analyses…",
                "RESEARCH": "Research service is retrieving verified citations…",
                "KNOWLEDGE": "Knowledge service is retrieving grounded theory…",
                "CHAT": "Understanding your request…",
            }
            status.write(labels.get(step.value, f"Running {step.value}…"))
        service.handle(prompt, conversation, parsed_query=parsed)
        status.update(label="Analysis completed", state="complete", expanded=False)
    store.save(conversation)
    st.session_state.conversations = store.items
    st.rerun()
