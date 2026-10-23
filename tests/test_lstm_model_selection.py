"""Offline inference/selection contracts, not model-quality validation."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

from stock_multi_agent.agents.technical_agent import TechnicalAgent
from stock_multi_agent.chat.response import ResponseComposer
from stock_multi_agent.models.forecast import ForecastStatus
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.lstm_forecast_v4 import LstmV4ForecastModel, LstmV4ArtifactError
from stock_multi_agent.services.ml_v4 import FEATURES, FEATURE_VERSION, feature_frame, origin_scales
from stock_multi_agent.services.price_forecast import PriceForecastService
from stock_multi_agent.tools.technical_indicators import TechnicalIndicatorToolkit


class NamedModel:
    def __init__(self, probability=.1):
        self.probability = probability
        self.calls = []

    def predict_on_batch(self, values):
        self.calls.append(values.copy())
        return {'return_1': np.array([[.5]]), 'return_5': np.array([[.2]]),
                'direction_1': np.array([[.1, self.probability, .9-self.probability]])}


@pytest.fixture
def archive(isolated_runtime_root):
    directory = isolated_runtime_root / 'v4-selection'
    directory.mkdir(exist_ok=True)
    spec = {'schema_version': 'lstm-spec-v4', 'feature_version': FEATURE_VERSION,
        'features': FEATURES, 'model_version': 'lstm-one-session-priority-v4',
        'protocol_id': 'test-only', 'window': 30, 'max_gap_business_days': 5,
        'output_heads': ['return_1', 'return_5', 'direction_1'], 'class_order': ['DOWN', 'FLAT', 'UP'],
        'scaler': {'features': FEATURES, 'fit_scope': 'TRAIN_ONLY',
                   'mean': [0.]*25, 'std': [1.]*25, 'lower': [-100.]*25, 'upper': [100.]*25},
        'origin_scale': {'feature': 'volatility_20', 'floor': .005, 'ceiling': .1},
        'calibration': {'1': {'alpha': .8, 'bias': .001, 'flat_threshold': .5,
                              'absolute_residual_quantile_90': .04}}}
    content = {'model.keras': b'fixture-never-load-as-keras', 'spec.json': json.dumps(spec).encode(),
               'evaluation.json': json.dumps({'protocol_id': 'test-only'}).encode(),
               'protocol.json': json.dumps({'protocol_id': 'test-only'}).encode()}
    files = {name: hashlib.sha256(value).hexdigest() for name, value in content.items()}
    manifest = {'schema_version': 'lstm-artifact-v4', 'files': files,
        'artifact_identity': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        'predictor_identity': hashlib.sha256(json.dumps({n: files[n] for n in ('model.keras', 'spec.json', 'protocol.json')}, sort_keys=True).encode()).hexdigest()}
    path = directory / 'fixture.zip'
    with zipfile.ZipFile(path, 'w') as z:
        for name, value in {**content, 'manifest.json': json.dumps(manifest).encode()}.items():
            z.writestr('run/' + name, value)
    return path


def test_v4_runtime_matches_training_transformation_and_calibration(archive, market_data):
    keras = NamedModel()
    model = LstmV4ForecastModel(archive, experimental_blend=True, keras_model=keras)
    forecast = model.predict_next_session(market_data)
    frame = feature_frame(pd.DataFrame([{'ticker': market_data.ticker, **bar.model_dump()} for bar in market_data.bars]))
    np.testing.assert_allclose(keras.calls[0][0], frame.iloc[-30:][FEATURES].to_numpy(dtype=np.float32))
    expected = .8 * .5 * origin_scales(frame)[-1, 0] + .001
    assert forecast.forecast_price == pytest.approx(market_data.bars[-1].close*(1+expected))
    assert forecast.policy_eligibility is False
    assert forecast.validation_status == 'PENDING_FORWARD_VALIDATION'
    assert forecast.experimental_blend_enabled and forecast.technical_blend_weight == .2


def test_v4_flat_gate_uses_probability_and_keeps_interval(archive, market_data):
    model = LstmV4ForecastModel(archive, keras_model=NamedModel(probability=.8))
    result = model.predict_next_session(market_data)
    assert result.forecast_price == result.current_price
    assert result.lower_bound < result.forecast_price < result.upper_bound
    assert result.technical_blend_weight == 0 and not result.experimental_blend_enabled


@pytest.mark.parametrize('explicit', [False, True])
def test_v4_only_explicit_demo_can_blend_twenty_percent(archive, market_data, explicit):
    model = LstmV4ForecastModel(archive, experimental_blend=explicit, keras_model=NamedModel())
    forecast = model.predict_next_session(market_data)
    agent = TechnicalAgent(TechnicalIndicatorToolkit())
    state = AgentState(ticker=market_data.ticker, market_data=market_data,
                       user_goal='fixture', investment_horizon='3 months')
    original = agent.run(state, allow_llm_planning=False)['technical_result']
    state.price_forecast = forecast
    after = agent.run(state, allow_llm_planning=False)['technical_result']
    if explicit:
        signal_score = 5+5*np.clip((forecast.forecast_price/forecast.current_price-1)/.03, -1, 1)
        assert after.score == pytest.approx(round(.8*original.score+.2*signal_score, 2))
        assert after.diagnostics['lstm_weight'] == .2
        assert after.diagnostics['indicator_weight'] == .8
        assert after.diagnostics['lstm_model_version'] == model.version
    else:
        assert after.score == original.score and 'lstm_weight' not in after.diagnostics
    state.technical_result = after
    summary = ResponseComposer._forecast_summary(state)
    assert ('đóng góp 20%' in summary) is explicit
    if explicit:
        assert 'V4 thử nghiệm' in summary


def test_tampered_archive_is_rejected(archive):
    damaged = archive.with_name('tampered.zip')
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(damaged, 'w') as dst:
        for name in src.namelist():
            dst.writestr(name, b'changed' if name.endswith('model.keras') else src.read(name))
    with pytest.raises(LstmV4ArtifactError, match='checksum'):
        LstmV4ForecastModel(damaged, keras_model=NamedModel())


@pytest.mark.parametrize('bad', ['shape', 'probability', 'nan', 'positional'])
def test_invalid_outputs_fall_back_without_model_weight(archive, market_data, bad):
    class Invalid(NamedModel):
        def predict_on_batch(self, values):
            result = super().predict_on_batch(values)
            if bad == 'shape': result['return_1'] = np.zeros((2, 1))
            if bad == 'probability': result['direction_1'][:] = 2.
            if bad == 'nan': result['return_1'][:] = np.nan
            if bad == 'positional': return list(result.values())
            return result
    service = PriceForecastService(trained_model=LstmV4ForecastModel(archive,
        experimental_blend=True, keras_model=Invalid()))
    result = service.forecast_next_session(market_data)
    assert result.status == ForecastStatus.BASELINE_UNTRAINED
    assert not result.experimental_blend_enabled
    assert result.technical_blend_weight in (0, None)


def test_short_history_does_not_run_model(archive, market_data):
    keras = NamedModel()
    model = LstmV4ForecastModel(archive, experimental_blend=True, keras_model=keras)
    with pytest.raises(LstmV4ArtifactError, match='window'):
        model.predict_next_session(market_data.model_copy(update={'bars': market_data.bars[-40:]}))
    assert not keras.calls


def test_ui_selection_overrides_env_without_mutating_it(archive, monkeypatch):
    monkeypatch.setenv('STOCK_AGENT_LSTM_ENABLED', 'true')
    monkeypatch.setenv('STOCK_AGENT_FORECAST_MODEL_VERSION', 'v3')
    monkeypatch.setenv('STOCK_AGENT_LSTM_V4_ARCHIVE', str(archive))
    service = PriceForecastService.from_environment(model_version='v4', experimental_blend=True)
    assert isinstance(service.trained_model, LstmV4ForecastModel)
    assert service.trained_model.experimental_blend
    import os
    assert os.environ['STOCK_AGENT_FORECAST_MODEL_VERSION'] == 'v3'


def test_missing_selected_v4_does_not_silently_use_v2(monkeypatch, isolated_runtime_root):
    monkeypatch.setenv('STOCK_AGENT_LSTM_ENABLED', 'true')
    monkeypatch.setenv('STOCK_AGENT_LSTM_V4_ARCHIVE', str(isolated_runtime_root/'absent.zip'))
    service = PriceForecastService.from_environment(model_version='v4', experimental_blend=True)
    assert service.trained_model is None and 'V4' in service.load_warning


def test_v2_selection_uses_v2_loader_and_existing_artifact_path(monkeypatch, isolated_runtime_root):
    from stock_multi_agent.services import lstm_forecast
    path = isolated_runtime_root/'v2-existing'
    class V2:
        def __init__(self, artifact_dir): self.artifact_dir = artifact_dir
    monkeypatch.setattr(lstm_forecast, 'LstmV2ForecastModel', V2)
    monkeypatch.setenv('STOCK_AGENT_LSTM_ENABLED', 'true')
    monkeypatch.setenv('STOCK_AGENT_LSTM_ARTIFACT_DIR', str(path))
    monkeypatch.setenv('STOCK_AGENT_LSTM_V3_SHADOW_DIR', '')
    monkeypatch.setenv('STOCK_AGENT_FORECAST_MODEL_VERSION', 'v4')
    service = PriceForecastService.from_environment(model_version='v2')
    assert isinstance(service.trained_model, V2)
    assert service.trained_model.artifact_dir == path


def test_global_disable_is_respected_even_with_ui_choice(monkeypatch):
    monkeypatch.setenv('STOCK_AGENT_LSTM_ENABLED', 'false')
    service = PriceForecastService.from_environment(model_version='v4', experimental_blend=True)
    assert service.trained_model is None


def test_v4_demo_flag_does_not_bypass_v3_gate(archive, market_data):
    forecast = LstmV4ForecastModel(archive, experimental_blend=True,
        keras_model=NamedModel()).predict_next_session(market_data)
    forecast = forecast.model_copy(update={'model_version': 'lstm-v3'})
    state = AgentState(ticker=market_data.ticker, market_data=market_data,
        price_forecast=forecast, user_goal='fixture', investment_horizon='3 months')
    result = TechnicalAgent(TechnicalIndicatorToolkit()).run(state, allow_llm_planning=False)['technical_result']
    assert 'lstm_weight' not in result.diagnostics


def test_session_rebuilds_on_model_switch_and_preserves_conversations():
    # Execute the actual initialization function in isolation, without startup
    # provider calls, secrets, TensorFlow, or a Streamlit browser.
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root/'app/streamlit_app.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'initialize_session')
    class Session(dict):
        def __getattr__(self, key): return self[key]
        def __setattr__(self, key, value): self[key] = value
    session = Session(conversations={'kept': 'history'}, current_conversation_id='kept')
    constructed = []
    def forecast(**kwargs):
        constructed.append(kwargs)
        return kwargs
    namespace = {'st': SimpleNamespace(session_state=session), 'SERVICE_RUNTIME_PROFILE': 'fixture',
        'selected_lstm': 'v2', 'ChatService': lambda **kwargs: SimpleNamespace(**kwargs),
        'AnalysisService': lambda **kwargs: SimpleNamespace(**kwargs),
        'PriceForecastService': SimpleNamespace(from_environment=forecast),
        'SQLiteConversationStore': lambda values: values}
    # Future annotations avoid evaluating production return types in this fixture.
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])), '<ui-initialize>', 'exec'), namespace)
    first, _ = namespace['initialize_session']()
    assert namespace['initialize_session']()[0] is first
    namespace['selected_lstm'] = 'v4'
    second, _ = namespace['initialize_session']()
    assert second is not first and session.conversations == {'kept': 'history'}
    assert constructed == [{'default_enabled': True, 'model_version': 'v2', 'experimental_blend': False, 'defer_load': True},
                           {'default_enabled': True, 'model_version': 'v4', 'experimental_blend': True, 'defer_load': True}]
