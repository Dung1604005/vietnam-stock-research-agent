from __future__ import annotations

import json
import os
from typing import Protocol, TypeVar
from urllib.request import Request, urlopen


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

    def summarize(self, system: str, facts: dict[str, object]) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'llm' extra to use OpenAI") from exc
        response = OpenAI().responses.create(
            model=self.model,
            instructions=system,
            input=json.dumps(facts, ensure_ascii=False),
        )
        return response.output_text

    def parse_structured(self, system: str, payload: dict[str, object], output_model: type[T]) -> T:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'llm' extra to use OpenAI") from exc
        response = OpenAI().responses.parse(
            model=self.model,
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
                "num_predict": self.num_predict,
            },
        }

    def summarize(self, system: str, facts: dict[str, object]) -> str:
        body = json.dumps({
            "model": self.model,
            "prompt": f"{system}\nFacts: {json.dumps(facts, ensure_ascii=False)}",
            "stream": False,
            **self._generation_settings(),
        }).encode()
        request = Request(f"{self.base_url}/api/generate", data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - configured local provider
            return str(json.loads(response.read())["response"])

    def parse_structured(self, system: str, payload: dict[str, object], output_model: type[T]) -> T:
        schema = output_model.model_json_schema()  # type: ignore[attr-defined]
        body = json.dumps({
            "model": self.model,
            "prompt": f"{system}\nInput: {json.dumps(payload, ensure_ascii=False)}",
            "format": schema,
            "stream": False,
            **self._generation_settings(),
        }).encode()
        request = Request(f"{self.base_url}/api/generate", data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - configured local provider
            content = str(json.loads(response.read())["response"])
        return output_model.model_validate_json(content)  # type: ignore[attr-defined,no-any-return]


def configured_semantic_provider() -> SemanticLLMProvider | None:
    provider = os.getenv("STOCK_AGENT_LLM_PROVIDER", "mock").strip().lower()
    if provider == "openai":
        return OpenAIProvider()
    if provider == "ollama":
        return OllamaProvider()
    return None
