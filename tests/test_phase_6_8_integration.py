"""Cross-phase offline fixtures; none of these tests measures investment accuracy."""
from types import SimpleNamespace
from pathlib import Path
import json

import pytest
from pydantic import BaseModel, ValidationError

from conftest import conflict_market_data
from stock_multi_agent.models.chat import ConversationState
from stock_multi_agent.models.forecast import PriceForecast, ForecastStatus
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.agent_llm import AgentLLMAdvisor
from stock_multi_agent.services.chat_service import ChatService
from stock_multi_agent.services.demo_bundle import DemoBundleService
from stock_multi_agent.services.llm_gateway import LLMGateway
from stock_multi_agent.services.progress_events import ProgressSink
from stock_multi_agent.services.finetuning_dataset import content_hash
from stock_multi_agent.providers.llm_provider import ColabQwenProvider


class FixtureProvider:
    def get_market_data(self, ticker, include_financials=True):
        return conflict_market_data(ticker)


class NeverGenerate:
    calls = 0
    def parse_structured(self, *args):
        self.calls += 1
        raise AssertionError('This execution profile must not generate')


def service_with_forbidden_model(monkeypatch):
    monkeypatch.setenv('STOCK_AGENT_LLM_PROVIDER', 'mock')
    monkeypatch.setenv('STOCK_AGENT_KNOWLEDGE_GRAPH_ENABLED', 'false')
    model = NeverGenerate()
    analysis = AnalysisService(provider=FixtureProvider(), llm_advisor=AgentLLMAdvisor(
        provider=LLMGateway(model), enabled=True))
    return ChatService(analysis_service=analysis, universe={'default': ['FPT']}), model


def test_quick_profile_is_no_llm_and_progress_reflects_graph_order(monkeypatch):
    service, model = service_with_forbidden_model(monkeypatch)
    sink = ProgressSink()
    response = service.handle('Phân tích FPT', ConversationState(), execution_mode='QUICK', progress_sink=sink)
    assert response.metadata['request_metrics']['llm_calls'] == 0 and model.calls == 0
    assert 'technical_score' in response.metadata and response.metadata['execution_profile']['mode'] == 'QUICK'
    events = response.metadata['progress_events']
    data_done = next(item['sequence'] for item in events if item['node'] == 'graph.data' and item['status'] != 'STARTED')
    technical_start = next(item['sequence'] for item in events if item['node'] == 'graph.technical' and item['status'] == 'STARTED')
    assert data_done < technical_start
    assert all(any(done['node'] == event['node'] and done['attempt'] == event['attempt'] and done['status'] != 'STARTED'
        for done in events) for event in events if event['status'] == 'STARTED')


def test_offline_chat_uses_real_replay_boundary_and_original_timestamps(monkeypatch, tmp_path):
    service, model = service_with_forbidden_model(monkeypatch)
    data = conflict_market_data()
    bundle = tmp_path / 'test-replay.json'
    DemoBundleService.capture([data], bundle, rights_note='Explicit offline synthetic test fixture only', allow_test_fixture=True)
    attempted = []
    def forbidden(*args, **kwargs):
        attempted.append(True)
        raise AssertionError('No network is permitted')
    import urllib.request
    import socket
    import stock_multi_agent.providers.llm_provider as llm_module
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(llm_module, 'urlopen', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    response = service.handle('Phân tích FPT', ConversationState(), execution_mode='OFFLINE', offline_bundle=bundle)
    assert not attempted and model.calls == 0
    assert response.metadata['execution_profile']['mode'] == 'OFFLINE'
    assert response.data_timestamp == data.as_of and response.data_fetched_at == data.fetched_at
    assert response.metadata['data_origins'][0]['mode'] == 'SNAPSHOT_REPLAY'
    assert 'technical_score' in response.metadata and 'không phải giá hiện tại' in response.message
    missing = service.handle('Phân tích HPG', ConversationState(), execution_mode='OFFLINE', offline_bundle=bundle)
    assert 'technical_score' not in missing.metadata and not attempted
    theory = service.handle('P/E là gì?', ConversationState(), execution_mode='OFFLINE')
    assert theory.message and not attempted


def test_bad_replay_is_friendly_not_live_fallback(monkeypatch, tmp_path):
    service, _ = service_with_forbidden_model(monkeypatch)
    path = tmp_path / 'bad.json'
    path.write_text('{"bundle_sha256":"wrong","payload":{}}', encoding='utf-8')
    conversation = ConversationState()
    response = service.handle('Phân tích FPT', conversation, execution_mode='OFFLINE', offline_bundle=path)
    assert 'Không mở được replay' in response.message
    assert len(conversation.messages) == 2 and 'technical_score' not in response.metadata


class TypedOutput(BaseModel):
    answer: str


def test_adapter_only_applies_to_exact_runtime_contract():
    provider = ColabQwenProvider(model=ColabQwenProvider.DEFAULT_MODEL, base_url='https://example.invalid', api_token='fixture')
    system = 'Chỉ trả lời bằng JSON.'
    provider.adapter_manifest = SimpleNamespace(prompt_hash=content_hash([system]),
        output_schema_hash=content_hash(TypedOutput.model_json_schema()))
    provider.adapter_identity = 'approved-test-identity'
    assert provider._adapter_for_contract(system, TypedOutput.model_json_schema()) == 'approved-test-identity'
    assert provider._adapter_for_contract(system + ' khác', TypedOutput.model_json_schema()) == ''
    assert provider._adapter_for_contract(system, {'type': 'object'}) == ''


def test_server_cannot_silently_ignore_requested_adapter(monkeypatch):
    import stock_multi_agent.providers.llm_provider as module
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({'output': {'answer': 'Đã trả lời'}}).encode()
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: Response())
    provider = ColabQwenProvider(base_url='https://example.invalid', api_token='fixture')
    with pytest.raises(RuntimeError, match='attest'):
        provider._post('/generate-structured', {'adapter_identity': 'approved-identity'})


@pytest.mark.parametrize('eligible,weight', [(None, None), (False, .2), (True, 0)])
def test_v3_does_not_inherit_legacy_twenty_percent_without_eligibility(eligible, weight):
    analysis = AnalysisService(provider=FixtureProvider())
    data = conflict_market_data()
    state = AgentState(ticker='FPT', investment_horizon='3 months', user_goal='Test', market_data=data)
    baseline = analysis.technical_agent.run(state, allow_llm_planning=False)['technical_result']
    state.price_forecast = PriceForecast(ticker='FPT', as_of=data.as_of, current_price=data.bars[-1].close,
        forecast_price=data.bars[-1].close * 1.03, status=ForecastStatus.TRAINED_MODEL,
        method='Fixture only', model_version='lstm-single-ticker-v3', confidence=.1,
        policy_eligibility=eligible, technical_blend_weight=weight)
    result = analysis.technical_agent.run(state, allow_llm_planning=False)['technical_result']
    assert result.score == baseline.score and 'lstm_weight' not in result.diagnostics


def test_forecast_rejects_nonfinite_prices():
    data = conflict_market_data()
    with pytest.raises(ValidationError):
        PriceForecast(ticker='FPT', as_of=data.as_of, current_price=100, forecast_price=float('inf'),
            status=ForecastStatus.TRAINED_MODEL, method='Fixture', model_version='bad', confidence=.1)


def test_streamlit_offline_worker_rerun_and_exports(monkeypatch, tmp_path):
    """Exercise actual UI/worker boundary without market/model/network services."""
    from streamlit.testing.v1 import AppTest
    import stock_multi_agent.config as config
    import urllib.request
    import socket
    service, _ = service_with_forbidden_model(monkeypatch)
    replay = tmp_path / 'ui-fixture.json'
    DemoBundleService.capture([conflict_market_data()], replay,
        rights_note='Explicit synthetic UI test fixture only', allow_test_fixture=True)
    monkeypatch.setattr(config, 'load_workspace_env', lambda: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('UI smoke must not make external calls')
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app/streamlit_app.py'), default_timeout=30)
    app.session_state['chat_service'] = service
    from stock_multi_agent.services.runtime_identity import SERVICE_RUNTIME_PROFILE
    app.session_state['chat_service_runtime_profile'] = f'{SERVICE_RUNTIME_PROFILE}:v2'
    app.run()
    assert not app.exception
    next(widget for widget in app.selectbox if widget.label == 'Chế độ chạy').select('OFFLINE').run()
    next(widget for widget in app.text_input if widget.label == 'Replay bundle JSON trên ổ D').set_value(str(replay)).run()
    app.chat_input[0].set_value('Phân tích FPT').run()
    assert not app.exception
    responses = [message.response for item in app.session_state['conversations'].values()
                 for message in item.messages if message.response is not None]
    assert len(responses) == 1
    assert responses[0].metadata['execution_profile']['mode'] == 'OFFLINE'
    assert len(app.get('download_button')) >= 3
    assert len(app.get('plotly_chart')) >= 1
    app.run()
    assert not app.exception
    assert sum(len(item.messages) for item in app.session_state['conversations'].values()) == 2
