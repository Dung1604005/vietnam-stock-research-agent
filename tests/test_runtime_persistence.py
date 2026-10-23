"""Durable local chat state, lazy model startup, and partial-result presentation."""
from pathlib import Path

import pytest

from stock_multi_agent.chat.conversation import SQLiteConversationStore
from stock_multi_agent.chat.response import ResponseComposer
from stock_multi_agent.models.chat import ChatRole, ConversationMessage
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.execution_profiles import execution_scope
from stock_multi_agent.services.price_forecast import PriceForecastService


def test_conversation_survives_store_recreation_and_delete(tmp_path):
    path = tmp_path / 'conversations.sqlite3'
    session = {}
    first = SQLiteConversationStore(session, path)
    conversation = first.create()
    conversation.active_tickers = ['FPT']
    conversation.add_message(ConversationMessage(role=ChatRole.USER, content='Phân tích FPT'))
    first.save(conversation)
    assert first.items is session
    restored = SQLiteConversationStore(path=path)
    assert restored.get(conversation.conversation_id).active_tickers == ['FPT']
    assert restored.get(conversation.conversation_id).messages[0].content == 'Phân tích FPT'
    restored.delete(conversation.conversation_id)
    assert SQLiteConversationStore(path=path).list() == []


def test_conversation_store_preserves_explicit_session_copy(tmp_path):
    path = tmp_path / 'conversations.sqlite3'
    first = SQLiteConversationStore(path=path)
    conversation = first.create()
    conversation.title = 'Pending session edit'
    second = SQLiteConversationStore({conversation.conversation_id: conversation}, path=path)
    assert second.get(conversation.conversation_id).title == 'Pending session edit'


def test_corrupt_conversation_is_not_overwritten(tmp_path):
    import sqlite3
    path = tmp_path / 'conversations.sqlite3'
    SQLiteConversationStore(path=path)
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO conversations VALUES (?,?)', ('broken', '{broken'))
    with pytest.raises(ValueError):
        SQLiteConversationStore(path=path)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT payload FROM conversations').fetchone()[0] == '{broken'


def test_forecast_model_not_loaded_at_ui_construction(monkeypatch, market_data):
    calls = []
    deferred = PriceForecastService.from_environment(default_enabled=True, model_version='v2', defer_load=True)
    def load(**configuration):
        calls.append(configuration)
        return PriceForecastService()
    monkeypatch.setattr(PriceForecastService, 'from_environment', load)
    assert deferred.load_warning is None and deferred.trained_model is None and calls == []
    with execution_scope('QUICK'):
        forecast = deferred.forecast_next_session(market_data)
    assert forecast.technical_blend_weight in {None, 0} and calls == []
    with execution_scope('STANDARD'):
        deferred.forecast_next_session(market_data)
        deferred.forecast_next_session(market_data)
    assert len(calls) == 1


def test_partial_data_response_does_not_invent_specialist_scores(market_data):
    state = AgentState(ticker='FPT', investment_horizon='3 months', user_goal='Analyze', market_data=market_data)
    response = ResponseComposer.partial_analysis(state)
    assert response.metadata['analysis_status'] == 'PARTIAL'
    assert 'technical_score' not in response.metadata and 'recommendation' not in response.metadata
    assert response.data_timestamp == market_data.as_of
    assert response.source_details == market_data.source_refs


def test_partial_response_requires_verified_data():
    state = AgentState(ticker='FPT', investment_horizon='3 months', user_goal='Analyze')
    with pytest.raises(ValueError, match='verified'):
        ResponseComposer.partial_analysis(state)


def test_unavailable_risk_is_not_presented_as_a_safe_recommendation():
    from test_dual_recommendation import _state
    from stock_multi_agent.models.screening import RecommendationBucket
    from stock_multi_agent.services.recommendation_service import RecommendationService
    state = _state('FPT', RecommendationBucket.SAFE_MODERATE_RETURN)
    state.risk_result.assessment_status = 'UNAVAILABLE'
    state.risk_result.risk_score = 0
    assert RecommendationService._classify('FPT', state) is None
    assert RecommendationService._reference_candidate('FPT', state) is None
    response = ResponseComposer.full_analysis(state)
    assert response.metadata['risk_level'] == 'UNKNOWN'
    assert response.metadata['risk_score'] is None
    assert '**Rủi ro đo lường:** CHƯA ĐÁNH GIÁ' in response.message


def test_cli_resume_uses_saved_run_without_starting_new_analysis(monkeypatch, capsys):
    import sys
    from stock_multi_agent import cli
    from test_dual_recommendation import _state
    from stock_multi_agent.models.screening import RecommendationBucket
    state = _state('FPT', RecommendationBucket.SAFE_MODERATE_RETURN)
    calls = []
    class Service:
        def resume_analysis(self, run_id):
            calls.append(run_id)
            return state
        def analyze(self, *args):
            raise AssertionError('A resumed run must not create a new analysis')
    monkeypatch.setattr(cli, 'load_workspace_env', lambda: None)
    monkeypatch.setattr(cli, 'AnalysisService', Service)
    monkeypatch.setattr(sys, 'argv', ['stock-agents', '--resume-run', state.run_id])
    cli.main()
    assert calls == [state.run_id]
    assert state.run_id in capsys.readouterr().out
