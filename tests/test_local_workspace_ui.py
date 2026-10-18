from streamlit.testing.v1 import AppTest

from stock_multi_agent.services.local_workspace import LocalWorkspace


def panel(mode="STANDARD"):
    return AppTest.from_string(f"""
import streamlit as st
from stock_multi_agent.ui.local_workspace_panel import render_local_workspace_panel
with st.sidebar:
    render_local_workspace_panel(['FPT'], {mode!r})
""").run()


def test_sidebar_persists_watchlist_notes_and_preferences(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_AGENT_RUNTIME_DIR", str(tmp_path))
    app = panel()
    assert not app.exception
    assert app.session_state["interaction_broker_template"] == "https://trade.vndirect.com.vn/"
    assert app.session_state["interaction_auto_open"] is True
    app.button(key="interaction_watchlist_add").click().run()
    assert LocalWorkspace().list_tickers() == ["FPT"]
    app.text_area(key="interaction_note_FPT").set_value("Theo dõi doanh thu").run()
    app.button(key="interaction_note_save").click().run()
    assert LocalWorkspace().get_note("FPT") == "Theo dõi doanh thu"
    app.button(key="interaction_broker_save").click().run()
    assert LocalWorkspace().get_preferences()["broker_name"] == "VNDIRECT"
    assert not app.exception


def test_sidebar_queues_analysis_and_chart_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_AGENT_RUNTIME_DIR", str(tmp_path))
    app = panel()
    app.button(key="interaction_analyze").click().run()
    assert app.session_state["pending_prompt"] == "Phân tích FPT"
    app.button(key="interaction_chart").click().run()
    assert app.session_state["pending_prompt"] == "Cho tôi biểu đồ FPT"
    app.button(key="interaction_report").click().run()
    assert app.session_state["pending_prompt"] == "Tìm báo cáo mới nhất của FPT"
    assert not app.exception


def test_sidebar_compare_and_pick_watchlist(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_AGENT_RUNTIME_DIR", str(tmp_path))
    LocalWorkspace().add_ticker("FPT")
    LocalWorkspace().add_ticker("VCB")
    app = panel()
    app.button(key="interaction_compare").click().run()
    assert app.session_state["pending_prompt"] == "So sánh FPT và VCB"
    app.selectbox(key="interaction_watchlist_pick").select("VCB").run()
    app.button(key="interaction_analyze").click().run()
    assert app.session_state["pending_prompt"] == "Phân tích VCB"
    app.button(key="interaction_watchlist_remove").click().run()
    assert LocalWorkspace().list_tickers() == ["FPT"]
    assert not app.exception


def test_sidebar_offline_has_no_external_links_but_notes_work(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_AGENT_RUNTIME_DIR", str(tmp_path))
    app = panel("OFFLINE")
    assert not app.exception
    assert not app.get("link_button")
    assert app.button(key="interaction_report").disabled
    assert app.checkbox(key="interaction_auto_open").disabled
    assert app.session_state["interaction_execution_mode"] == "OFFLINE"
    app.button(key="interaction_watchlist_add").click().run()
    assert LocalWorkspace().list_tickers() == ["FPT"]


def test_sidebar_corrupt_store_does_not_crash_or_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_AGENT_RUNTIME_DIR", str(tmp_path))
    path = tmp_path / "interaction" / "workspace.json"
    path.parent.mkdir()
    path.write_text("broken", encoding="utf-8")
    app = panel()
    assert not app.exception
    assert app.error
    assert path.read_text(encoding="utf-8") == "broken"


def test_sidebar_custom_broker_rejects_unsafe_urls(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_AGENT_RUNTIME_DIR", str(tmp_path))
    app = panel()
    app.checkbox(key="interaction_broker_custom").check().run()
    app.text_input(key="interaction_broker_url_input").set_value("https://example.com/execute?symbol={ticker}").run()
    app.button(key="interaction_broker_save").click().run()
    assert app.error
    assert LocalWorkspace().get_preferences() == {}
    app.text_input(key="interaction_broker_url_input").set_value("https://example.com/quote/{ticker}").run()
    app.button(key="interaction_broker_save").click().run()
    assert LocalWorkspace().get_preferences()["broker_url_template"] == "https://example.com/quote/{ticker}"
    assert not app.exception
