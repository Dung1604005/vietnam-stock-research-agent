from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from stock_multi_agent.services.request_telemetry import check_deadline, observed, remaining_seconds


def _transport_timeout(configured: float) -> float:
    check_deadline()
    remaining = remaining_seconds()
    return min(configured, remaining) if remaining is not None else configured


class LLMProvider(Protocol):
    def summarize(self, system: str, facts: dict[str, object]) -> str: ...


T = TypeVar("T")


class SemanticLLMProvider(Protocol):
    def parse_structured(self, system: str, payload: dict[str, object], output_model: type[T]) -> T: ...


class MockLLMProvider:
    """Deterministic presenter; analysis policy remains inside agents."""

    def summarize(self, system: str, facts: dict[str, object]) -> str:
        del system
        return "; ".join(f"{key}={value}" for key, value in facts.items())


class OpenAIProvider:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.timeout_seconds = 45

    @observed("llm.openai.summarize", "llm")
    def summarize(self, system: str, facts: dict[str, object]) -> str:
        from stock_multi_agent.services.execution_profiles import assert_external_allowed
        assert_external_allowed('llm.openai')
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'llm' extra to use OpenAI") from exc
        response = OpenAI(timeout=_transport_timeout(self.timeout_seconds), max_retries=0).responses.create(
            model=self.model,
            max_output_tokens=getattr(self, 'max_output_tokens', 640),
            instructions=system,
            input=json.dumps(facts, ensure_ascii=False),
        )
        return response.output_text

    @observed("llm.openai.structured", "llm")
    def parse_structured(self, system: str, payload: dict[str, object], output_model: type[T]) -> T:
        from stock_multi_agent.services.execution_profiles import assert_external_allowed
        assert_external_allowed('llm.openai')
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'llm' extra to use OpenAI") from exc
        response = OpenAI(timeout=_transport_timeout(self.timeout_seconds), max_retries=0).responses.parse(
            model=self.model,
            max_output_tokens=getattr(self, 'max_output_tokens', 640),
            instructions=system,
            input=json.dumps(payload, ensure_ascii=False),
            text_format=output_model,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned no structured semantic output")
        return parsed


class OllamaProvider:
    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout_seconds = self._env_int("OLLAMA_TIMEOUT_SECONDS", 90, 5, 120)
        self.num_predict = self._env_int("OLLAMA_NUM_PREDICT", 512, 64, 1024)
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip() or "30m"

    @staticmethod
    def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except ValueError:
            value = default
        return max(minimum, min(maximum, value))

    def _generation_settings(self) -> dict[str, object]:
        return {
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "num_predict": min(self.num_predict, getattr(self, 'max_output_tokens', self.num_predict)),
            },
        }

    @observed("llm.ollama.summarize", "llm")
    def summarize(self, system: str, facts: dict[str, object]) -> str:
        from stock_multi_agent.services.execution_profiles import assert_external_allowed
        assert_external_allowed('llm.ollama')
        body = json.dumps({
            "model": self.model,
            "prompt": f"{system}\nFacts: {json.dumps(facts, ensure_ascii=False)}",
            "stream": False,
            **self._generation_settings(),
        }).encode()
        request = Request(f"{self.base_url}/api/generate", data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=_transport_timeout(self.timeout_seconds)) as response:  # noqa: S310 - configured local provider
            return str(json.loads(response.read())["response"])

    @observed("llm.ollama.structured", "llm")
    def parse_structured(self, system: str, payload: dict[str, object], output_model: type[T]) -> T:
        from stock_multi_agent.services.execution_profiles import assert_external_allowed
        assert_external_allowed('llm.ollama')
        schema = output_model.model_json_schema()  # type: ignore[attr-defined]
        body = json.dumps({
            "model": self.model,
            "prompt": f"{system}\nInput: {json.dumps(payload, ensure_ascii=False)}",
            "format": schema,
            "stream": False,
            **self._generation_settings(),
        }).encode()
        request = Request(f"{self.base_url}/api/generate", data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=_transport_timeout(self.timeout_seconds)) as response:  # noqa: S310 - configured local provider
            content = str(json.loads(response.read())["response"])
        return output_model.model_validate_json(content)  # type: ignore[attr-defined,no-any-return]


class ColabQwenProvider:
    """Authenticated Qwen client; optional adapters are exact-contract, fail-closed."""

    DEFAULT_MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
    DEFAULT_REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_token: str | None = None,
    ) -> None:
        self.model = model or os.getenv("COLAB_QWEN_MODEL", self.DEFAULT_MODEL)
        self.model_revision = os.getenv('COLAB_QWEN_MODEL_REVISION',
            self.DEFAULT_REVISION if self.model == self.DEFAULT_MODEL else '').strip()
        self.tokenizer_revision = os.getenv('COLAB_QWEN_TOKENIZER_REVISION', self.model_revision).strip()
        self.adapter_identity = ''
        self.adapter_manifest = None
        self.adapter_status = 'DISABLED'
        manifest_path = os.getenv('STOCK_AGENT_LLM_ADAPTER_MANIFEST', '').strip()
        if manifest_path:
            try:
                from pathlib import Path
                from stock_multi_agent.services.llm_adapter_registry import load_approved_adapter
                if os.name == 'nt' and Path(manifest_path).resolve().drive.upper() != 'D:':
                    raise ValueError('Adapter artifacts must stay on D')
                self.adapter_manifest = load_approved_adapter(manifest_path,
                    expected_base_model=self.model, expected_base_revision=self.model_revision,
                    expected_tokenizer_revision=self.tokenizer_revision,
                    expected_schema_hash=os.getenv('STOCK_AGENT_LLM_ADAPTER_SCHEMA_HASH', '').strip(),
                    expected_prompt_hash=os.getenv('STOCK_AGENT_LLM_ADAPTER_PROMPT_HASH', '').strip())
                self.adapter_identity = self.adapter_manifest.identity
                self.adapter_status = 'APPROVED_CONTRACT_ONLY'
            except (ValueError, OSError):
                self.adapter_status = 'REJECTED_BASE_MODEL_ONLY'
        self.base_url = (base_url or os.getenv("COLAB_QWEN_BASE_URL", "")).strip().rstrip("/")
        self.api_token = (api_token or os.getenv("COLAB_QWEN_API_TOKEN", "")).strip()
        self.timeout_seconds = self._env_int("COLAB_QWEN_TIMEOUT_SECONDS", 120, 10, 300)

    @staticmethod
    def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except ValueError:
            value = default
        return max(minimum, min(maximum, value))

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        from stock_multi_agent.services.execution_profiles import assert_external_allowed
        assert_external_allowed('llm.colab')
        if not self.base_url or not self.api_token:
            raise RuntimeError("Colab Qwen URL/token is not configured")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=_transport_timeout(self.timeout_seconds)) as response:  # noqa: S310 - user-configured endpoint
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"Colab Qwen returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError("Colab Qwen endpoint is unavailable") from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("Colab Qwen returned an invalid or timed-out response") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("Colab Qwen returned a non-object response")
        self._check_serving_identity(decoded)
        if payload.get('adapter_identity') and (
            decoded.get('adapter_identity') != payload['adapter_identity']
            or decoded.get('model_revision') != self.model_revision
        ):
            raise RuntimeError('Qwen server did not attest the requested approved adapter/revision')
        return decoded

    def _check_serving_identity(self, decoded, *, required=False):
        if required or 'model' in decoded:
            if decoded.get('model') != self.model:
                raise RuntimeError('Qwen endpoint serves a different model; check URL and model configuration')
        if self.model_revision and (required or 'model_revision' in decoded):
            if decoded.get('model_revision') != self.model_revision:
                raise RuntimeError('Qwen endpoint model revision mismatch')

    def check_health(self):
        """Explicit authenticated model-identity check; no generation or token logging."""
        from stock_multi_agent.services.execution_profiles import assert_external_allowed
        assert_external_allowed('llm.colab.health')
        if not self.base_url or not self.api_token:
            raise RuntimeError('Colab Qwen URL/token is not configured')
        request = Request(self.base_url + '/health', headers={'Authorization': 'Bearer ' + self.api_token})
        try:
            with urlopen(request, timeout=_transport_timeout(min(self.timeout_seconds, 10))) as response:
                decoded = json.loads(response.read().decode('utf-8'))
        except HTTPError as exc:
            raise RuntimeError(f'Colab Qwen health returned HTTP {exc.code}') from exc
        except (URLError, TimeoutError, ValueError) as exc:
            raise RuntimeError('Colab Qwen health is unavailable or invalid') from exc
        if not isinstance(decoded, dict) or decoded.get('status') != 'ok':
            raise RuntimeError('Colab Qwen is not ready')
        self._check_serving_identity(decoded, required=True)
        return decoded

    def _adapter_for_contract(self, system, schema):
        if self.adapter_manifest is None:
            return ''
        from stock_multi_agent.services.finetuning_dataset import content_hash
        if (content_hash([system]) == self.adapter_manifest.prompt_hash
                and content_hash(schema) == self.adapter_manifest.output_schema_hash):
            return self.adapter_identity
        return ''  # Never apply a routing adapter to a different specialist schema.

    @observed("llm.colab.summarize", "llm")
    def summarize(self, system: str, facts: dict[str, object]) -> str:
        response = self._post("/generate", {
            "model": self.model,
            "system": system,
            "payload": facts,
            "max_new_tokens": getattr(self, 'max_output_tokens', 640),
        })
        output = response.get("output")
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError("Colab Qwen returned no text output")
        return output.strip()

    @observed("llm.colab.structured", "llm")
    def parse_structured(self, system: str, payload: dict[str, object], output_model: type[T]) -> T:
        schema = output_model.model_json_schema()
        adapter = self._adapter_for_contract(system, schema)
        request_payload = {
            "model": self.model,
            "system": system,
            "payload": payload,
            "json_schema": schema,
            "max_new_tokens": getattr(self, 'max_output_tokens', 640),
        }
        if adapter:
            request_payload.update(adapter_identity=adapter, model_revision=self.model_revision)
        response = self._post("/generate-structured", request_payload)
        output = response.get("output")
        if output is None:
            raise RuntimeError("Colab Qwen returned no structured output")
        return output_model.model_validate(output)  # type: ignore[attr-defined,no-any-return]


def configured_semantic_provider() -> SemanticLLMProvider | None:
    provider = os.getenv("STOCK_AGENT_LLM_PROVIDER", "mock").strip().lower()
    if provider == "openai":
        return _gateway(OpenAIProvider())
    if provider == "ollama":
        return _gateway(OllamaProvider())
    if provider in {"colab", "colab_qwen", "remote_qwen"}:
        return _gateway(ColabQwenProvider())
    return None


def _gateway(provider):
    # Credentials/URL are hashed for identity and never included in logs/cache keys.
    import hashlib
    identity = hashlib.sha256(repr(sorted(vars(provider).items())).encode()).hexdigest()
    return _shared_gateway(type(provider).__name__, identity, provider.model)


@lru_cache(maxsize=4)
def _shared_gateway(kind, identity, model):
    del identity
    from stock_multi_agent.services.llm_gateway import LLMGateway
    provider = {'OpenAIProvider': OpenAIProvider, 'OllamaProvider': OllamaProvider,
                'ColabQwenProvider': ColabQwenProvider}[kind](model=model)
    return LLMGateway(provider)
