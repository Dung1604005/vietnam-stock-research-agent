from __future__ import annotations

from datetime import date, datetime, timezone

from stock_multi_agent.agents.manager_agent import ManagerAgent
from stock_multi_agent.models.chat import AssistantResponse, ParsedQuery, QueryIntent, ResponseType
from stock_multi_agent.models.data import DataKind, SourceReference


class _ManagerSynthesisProvider:
    def __init__(self) -> None:
        self.payload = None

    def parse_structured(self, system, payload, output_model):
        assert "4 đến 6 câu tiếng Việt" in system
        self.payload = payload
        return output_model.model_validate({
            "lines": [
                "Người dùng yêu cầu phân tích cổ phiếu FPT trong ba tháng.",
                "Dữ liệu định lượng hiện có cho thấy tín hiệu kỹ thuật cần được theo dõi.",
                "Kết quả cơ bản và rủi ro được giữ nguyên theo các agent chuyên trách.",
                "Dự báo còn bất định nên không được xem là cam kết lợi nhuận.",
            ],
            "cited_source_ids": ["test:ohlcv:FPT", "invented:source"],
        })


def test_manager_synthesis_is_grounded_and_filters_sources() -> None:
    provider = _ManagerSynthesisProvider()
    manager = ManagerAgent(
        llm_provider=provider,
        routing_enabled=False,
        synthesis_enabled=True,
    )
    query = ParsedQuery(
        raw_query="Phân tích FPT",
        intent=QueryIntent.STOCK_ANALYSIS,
        tickers=["FPT"],
        confidence=0.9,
    )
    fetched_at = datetime(2026, 10, 12, tzinfo=timezone.utc)
    response = AssistantResponse(
        message="Kết luận deterministic: THEO DÕI.",
        response_type=ResponseType.ANALYSIS,
        tickers=["FPT"],
        source_details=[SourceReference(
            provider="test",
            source_id="test:ohlcv:FPT",
            data_kind=DataKind.OHLCV,
            fetched_at=fetched_at,
            as_of=date(2026, 10, 9),
        )],
        metadata={"recommendation": "WATCH", "private_blob": "must-not-be-sent"},
    )

    result = manager.summarize_response(query, response)

    assert result.status == "GROUNDED"
    assert len(result.lines) == 4
    assert result.cited_source_ids == ["test:ohlcv:FPT"]
    assert provider.payload["decision_facts"] == {"recommendation": "WATCH"}
    assert "private_blob" not in str(provider.payload)


def test_manager_synthesis_rejects_cjk_output() -> None:
    class CjkProvider:
        def parse_structured(self, system, payload, output_model):
            del system, payload
            return output_model.model_validate({
                "lines": ["这是中文输出。"] * 4,
                "cited_source_ids": [],
            })

    manager = ManagerAgent(llm_provider=CjkProvider(), routing_enabled=False, synthesis_enabled=True)
    query = ParsedQuery(raw_query="Phân tích FPT", intent=QueryIntent.STOCK_ANALYSIS, tickers=["FPT"])
    response = AssistantResponse(message="Kết quả", response_type=ResponseType.ANALYSIS, tickers=["FPT"])
    result = manager.summarize_response(query, response)
    assert result.status == "UNAVAILABLE"
    assert result.fallback_reason == "non_vietnamese_output"
