from __future__ import annotations

from pydantic import BaseModel, Field

from stock_multi_agent.config import load_workspace_env
from stock_multi_agent.providers.llm_provider import ColabQwenProvider, configured_semantic_provider


class ProbeOutput(BaseModel):
    status: str = Field(pattern=r"^OK$")
    language: str = Field(pattern=r"^vi$")


def main() -> None:
    load_workspace_env()
    provider = configured_semantic_provider()
    if not isinstance(provider, ColabQwenProvider):
        raise SystemExit("STOCK_AGENT_LLM_PROVIDER phải là colab_qwen.")
    result = provider.parse_structured(
        "Trả đúng JSON schema, không thêm nội dung khác.",
        {"task": "Xác nhận endpoint hoạt động", "required_status": "OK", "required_language": "vi"},
        ProbeOutput,
    )
    print(f"Colab Qwen hoạt động: status={result.status}, language={result.language}, model={provider.model}")


if __name__ == "__main__":
    main()
