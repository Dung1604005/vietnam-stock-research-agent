"""Server/client contracts without downloading Qwen, GPU, or public publication."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from stock_multi_agent.providers.llm_provider import ColabQwenProvider
import stock_multi_agent.providers.llm_provider as provider_module

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qwen3_serving', ROOT/'notebooks/serve_qwen3_4b_colab.py')
serving = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serving)


class Generator:
    def __init__(self, output='{"status":"OK","language":"vi"}'):
        self.output, self.calls = output, []

    def generate(self, system, user, tokens):
        self.calls.append((system, user, tokens))
        return self.output


class Probe(BaseModel):
    status: str
    language: str


def request(**changes):
    return {'model': serving.MODEL_ID, 'system': 'test', 'payload': {'query': 'fixture'},
            'json_schema': Probe.model_json_schema(), 'max_new_tokens': 128, **changes}


def test_health_requires_auth_and_exposes_actual_model_identity():
    engine = Generator()
    api = serving.QwenAPI(engine, token='fixture')
    with pytest.raises(serving.APIError) as exc:
        api.dispatch('/health', None, None)
    assert exc.value.status == 401
    result = api.dispatch('/health', None, 'Bearer fixture')
    assert result['model'] == serving.MODEL_ID
    assert result['model_revision'] == serving.MODEL_REVISION
    assert result['adapter_mode'] == 'BASE_ONLY'
    assert 'token' not in result and not engine.calls


def test_structured_server_matches_actual_client(monkeypatch):
    engine = Generator()
    api = serving.QwenAPI(engine, token='fixture')
    class Response:
        def __init__(self, output): self.output = output
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(self.output).encode()
    def urlopen(req, timeout):
        path = '/health' if req.data is None else '/generate-structured'
        output = api.dispatch(path, None if req.data is None else json.loads(req.data), req.get_header('Authorization'))
        return Response(output)
    monkeypatch.setattr(provider_module, 'urlopen', urlopen)
    monkeypatch.setenv('COLAB_QWEN_MODEL_REVISION', serving.MODEL_REVISION)
    provider = ColabQwenProvider(model=serving.MODEL_ID, base_url='https://fixture.invalid', api_token='fixture')
    assert provider.check_health()['adapter_mode'] == 'BASE_ONLY'
    result = provider.parse_structured('test', {'query': 'fixture'}, Probe)
    assert result.status == 'OK' and len(engine.calls) == 1
    assert 'Required JSON Schema' in engine.calls[0][1]


def test_real_loopback_http_client_roundtrip(monkeypatch):
    """Real HTTP, fixture generation only. No public URL or GPU/model download."""
    monkeypatch.setenv('COLAB_QWEN_MODEL_REVISION', serving.MODEL_REVISION)
    api = serving.QwenAPI(Generator(), token='private-test-token')
    server = serving.start_server(api, port=0)
    provider = ColabQwenProvider(model=serving.MODEL_ID,
        base_url=f'http://127.0.0.1:{server.server_address[1]}', api_token=api.token)
    try:
        assert provider.check_health()['model_revision'] == serving.MODEL_REVISION
        assert provider.parse_structured('fixture', {}, Probe).status == 'OK'
        provider.api_token = 'wrong'
        with pytest.raises(RuntimeError, match='HTTP 401'):
            provider.check_health()
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize('changes,status', [
    ({'model': 'Qwen/Qwen2.5-3B-Instruct'}, 400),
    ({'model_revision': 'bad'}, 409),
    ({'adapter_identity': 'unapproved'}, 409),
    ({'max_new_tokens': 9999}, 400), ({'max_new_tokens': True}, 400),
    ({'payload': 'not object'}, 400), ({'system': ''}, 400),
    ({'json_schema': {'$ref': 'https://example.invalid/schema'}}, 400),
    ({'json_schema': {'$ref': 'file:///etc/passwd'}}, 400),
])
def test_invalid_requests_rejected_before_generation(changes, status):
    engine = Generator()
    api = serving.QwenAPI(engine, token='fixture')
    with pytest.raises(serving.APIError) as exc:
        api.dispatch('/generate-structured', request(**changes), 'Bearer fixture')
    assert exc.value.status == status and not engine.calls


@pytest.mark.parametrize('output', ['not JSON', '{"wrong":1}', '{"status":NaN}', '', 'explanation {"status":"OK","language":"vi"}'])
def test_invalid_model_output_is_not_faked_or_retried(output):
    engine = Generator(output)
    api = serving.QwenAPI(engine, token='fixture')
    with pytest.raises(serving.APIError) as exc:
        api.dispatch('/generate-structured', request(), 'Bearer fixture')
    assert exc.value.status == 422 and len(engine.calls) == 1
    assert api.lock.acquire(blocking=False)
    api.lock.release()


def test_schema_with_local_defs_is_supported():
    api = serving.QwenAPI(Generator('{"value":"OK"}'), token='fixture')
    schema = {'type': 'object', '$defs': {'word': {'const': 'OK'}},
              'properties': {'value': {'$ref': '#/$defs/word'}}, 'required': ['value']}
    result = api.dispatch('/generate-structured', request(json_schema=schema), 'Bearer fixture')
    assert result['output'] == {'value': 'OK'}


def test_busy_server_returns_429_without_another_generation():
    engine = Generator()
    api = serving.QwenAPI(engine, token='fixture')
    api.lock.acquire()
    try:
        with pytest.raises(serving.APIError) as exc:
            api.dispatch('/generate-structured', request(), 'Bearer fixture')
        assert exc.value.status == 429 and not engine.calls
    finally:
        api.lock.release()


def test_text_route_preserves_text_and_rejects_wrong_model():
    api = serving.QwenAPI(Generator('Nhận định fixture.'), token='fixture')
    assert api.dispatch('/generate', request(), 'Bearer fixture')['output'] == 'Nhận định fixture.'
    with pytest.raises(serving.APIError):
        api.dispatch('/generate', request(model='wrong'), 'Bearer fixture')


def test_client_rejects_reported_wrong_model_and_revision():
    provider = ColabQwenProvider(model=serving.MODEL_ID, base_url='https://fixture.invalid', api_token='fixture')
    provider.model_revision = serving.MODEL_REVISION
    with pytest.raises(RuntimeError, match='different model'):
        provider._check_serving_identity({'model': 'Qwen2.5'})
    with pytest.raises(RuntimeError, match='revision mismatch'):
        provider._check_serving_identity({'model': serving.MODEL_ID, 'model_revision': 'wrong'})


def test_check_command_unwraps_gateway(monkeypatch, capsys):
    check_spec = importlib.util.spec_from_file_location('check_qwen_test', ROOT/'scripts/check_colab_qwen.py')
    check = importlib.util.module_from_spec(check_spec)
    check_spec.loader.exec_module(check)
    provider = ColabQwenProvider(model=serving.MODEL_ID, base_url='https://fixture.invalid', api_token='fixture')
    monkeypatch.setattr(provider, 'check_health', lambda: {'model': serving.MODEL_ID, 'model_revision': serving.MODEL_REVISION, 'adapter_mode': 'BASE_ONLY'})
    monkeypatch.setattr(provider, 'parse_structured', lambda *args: check.ProbeOutput(status='OK', language='vi'))
    from types import SimpleNamespace
    monkeypatch.setattr(check, 'configured_semantic_provider', lambda: SimpleNamespace(provider=provider))
    monkeypatch.setattr(check, 'load_workspace_env', lambda: None)
    check.main()
    output = capsys.readouterr().out
    assert serving.MODEL_ID in output and 'BASE_ONLY' in output
    assert 'fixture' not in output


def test_notebook_self_contained_and_safe_defaults():
    notebook = json.loads((ROOT/'notebooks/serve_qwen3_4b_colab.ipynb').read_text(encoding='utf-8'))
    sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    for source in sources:
        ast.parse(source)
    assert (ROOT/'notebooks/serve_qwen3_4b_colab.py').read_text(encoding='utf-8') in sources
    joined = '\n'.join(sources)
    assert 'PUBLISH_ENDPOINT = False' in joined and 'HOST_POLICY_REVIEWED = False' in joined
    assert 'STOP_SERVER = False' in joined
    assert 'adapter_model.safetensors' not in joined
    assert 'bitsandbytes==0.48.2' in joined
    assert not any(c.get('outputs') for c in notebook['cells'])
