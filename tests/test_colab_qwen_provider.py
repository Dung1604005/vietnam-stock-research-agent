from __future__ import annotations

import json

from pydantic import BaseModel

import stock_multi_agent.providers.llm_provider as provider_module
from stock_multi_agent.providers.llm_provider import ColabQwenProvider, configured_semantic_provider


class _Output(BaseModel):
    intent: str
    confidence: float


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_colab_qwen_provider_posts_authenticated_schema_request(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"output": {"intent": "STOCK_ANALYSIS", "confidence": 0.91}})

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)
    provider = ColabQwenProvider(
        base_url="https://demo.trycloudflare.com/",
        api_token="test-token",
        model="Qwen/Qwen2.5-3B-Instruct",
    )
    result = provider.parse_structured(
        "Return typed output.",
        {"query": "Phân tích FPT", "ticker": "FPT"},
        _Output,
    )

    assert result.intent == "STOCK_ANALYSIS"
    assert captured["url"] == "https://demo.trycloudflare.com/generate-structured"
    assert captured["authorization"] == "Bearer test-token"
    assert captured["timeout"] == 120
    assert captured["body"]["payload"] == {"query": "Phân tích FPT", "ticker": "FPT"}
    assert "api_token" not in captured["body"]
    assert captured["body"]["json_schema"]["type"] == "object"


def test_configured_provider_selects_colab_qwen(monkeypatch) -> None:
    monkeypatch.setenv("STOCK_AGENT_LLM_PROVIDER", "colab_qwen")
    monkeypatch.setenv("COLAB_QWEN_BASE_URL", "https://demo.trycloudflare.com")
    monkeypatch.setenv("COLAB_QWEN_API_TOKEN", "test-token")
    provider = configured_semantic_provider()
    from stock_multi_agent.services.llm_gateway import LLMGateway
    assert isinstance(provider, LLMGateway)
    assert isinstance(provider.provider, ColabQwenProvider)


def test_missing_colab_credentials_fail_only_when_called() -> None:
    provider = ColabQwenProvider(base_url="", api_token="")
    try:
        provider.parse_structured("system", {"query": "FPT"}, _Output)
    except RuntimeError as exc:
        assert str(exc) == "Colab Qwen URL/token is not configured"
    else:
        raise AssertionError("Expected missing credentials to reject the call")
