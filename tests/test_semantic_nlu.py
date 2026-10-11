from __future__ import annotations

import json
from pathlib import Path

import pytest

from stock_multi_agent.chat.query_parser import QueryParser
from stock_multi_agent.models.chat import ConversationState, QueryIntent
from stock_multi_agent.nlu.models import SemanticQuery


DATASET = json.loads(
    (Path(__file__).parent / "data" / "semantic_queries.json").read_text(encoding="utf-8")
)


def build_context(case: dict) -> ConversationState | None:
    if "context" not in case:
        return None
    values = dict(case["context"])
    if "last_intent" in values:
        values["last_intent"] = QueryIntent(values["last_intent"])
    return ConversationState(**values)


@pytest.mark.parametrize("case", DATASET, ids=lambda case: case["query"])
def test_semantic_dataset(case: dict) -> None:
    parsed = QueryParser(use_configured_provider=False).parse(case["query"], build_context(case))
    assert parsed.intent.value == case["expected_intent"]
    assert parsed.tickers == case.get("expected_tickers", [])
    if "expected_period" in case:
        assert parsed.chart_period == case["expected_period"]
    if "expected_horizon" in case:
        assert parsed.investment_horizon.startswith(case["expected_horizon"].removesuffix("M"))
    if "expected_indicators" in case:
        assert set(case["expected_indicators"]).issubset(parsed.requested_indicators)
    if "expected_requested_result_count" in case:
        assert parsed.requested_result_count == case["expected_requested_result_count"]
    if "expected_requested_actions" in case:
        assert parsed.requested_actions == case["expected_requested_actions"]
    if "expected_sector" in case:
        assert parsed.sector == case["expected_sector"]
    if "expected_follow_up" in case:
        assert parsed.follow_up is case["expected_follow_up"]
    if "expected_needs_clarification" in case:
        assert parsed.needs_clarification is case["expected_needs_clarification"]
    if "expected_secondary_intents" in case:
        assert set(case["expected_secondary_intents"]).issubset(
            intent.value for intent in parsed.secondary_intents
        )


def test_ticker_is_not_fuzzy_corrected() -> None:
    parsed = QueryParser(use_configured_provider=False).parse("phân tích FPX")
    assert parsed.tickers == ["FPX"]
    assert "FPT" not in parsed.tickers


def test_exchange_name_selects_current_universe_screen_not_a_ticker() -> None:
    # Exchange terms select a universe, not an equity ticker.
    parsed = QueryParser(use_configured_provider=False).parse("Sàng lọc sàn HOSE")
    assert parsed.intent is QueryIntent.STOCK_DISCOVERY
    assert parsed.sector == "hose_current"
    assert parsed.tickers == []


def test_discovery_parser_preserves_requested_count_and_buy_sell_direction() -> None:
    parser = QueryParser(use_configured_provider=False)
    buy = parser.parse("Tìm cho tôi 10 mã tốt nhất để mua")
    assert buy.intent is QueryIntent.STOCK_DISCOVERY
    assert buy.requested_result_count == 10
    assert buy.requested_actions == ["BUY"]

    ambiguous = parser.parse("Tìm mã tốt nhất để buy hoặc sell")
    assert ambiguous.requested_result_count == 1
    assert ambiguous.requested_actions == ["BUY", "SELL"]
    assert ambiguous.needs_clarification is True


def test_semantic_result_exposes_auditable_fields() -> None:
    context = ConversationState(active_tickers=["FPT"])
    parsed = QueryParser(use_configured_provider=False).parse("con này rủi ro không", context)
    assert parsed.confidence >= 0.60
    assert parsed.references_previous_context is True
    assert parsed.resolved_references
    assert parsed.interpretation
    assert parsed.parser_source == "semantic_similarity"
    assert parsed.llm_attempted is False
    assert parsed.fallback_reason == "no_configured_llm"
    assert parsed.parser_latency_ms >= 0
    assert parsed.prompt_version == "semantic-system-v2-kb"


def test_structured_llm_path_uses_schema_and_merges_safe_entities() -> None:
    class FakeStructuredProvider:
        def parse_structured(self, system, payload, output_model):
            assert "Never invent" in system
            assert payload["deterministic_entities"]["tickers"] == ["FPT"]
            assert output_model is SemanticQuery
            return SemanticQuery(
                original_query="",
                normalized_query="",
                intent=QueryIntent.STOCK_ANALYSIS,
                tickers=[],
                confidence=0.92,
                interpretation="Người dùng muốn đánh giá FPT.",
            )

    parsed = QueryParser(llm_provider=FakeStructuredProvider()).parse("FPT ổn không?")
    assert parsed.parser_source == "llm_structured"
    assert parsed.tickers == ["FPT"]
    assert parsed.confidence == pytest.approx(0.92)
    assert parsed.llm_attempted is True
    assert parsed.fallback_reason is None


def test_structured_llm_cannot_bypass_buy_sell_clarification() -> None:
    class FakeStructuredProvider:
        def parse_structured(self, system, payload, output_model):
            del system, payload, output_model
            return SemanticQuery(
                original_query="", normalized_query="", intent=QueryIntent.STOCK_DISCOVERY,
                confidence=0.9, interpretation="screen",
            )

    parsed = QueryParser(llm_provider=FakeStructuredProvider()).parse("Tim ma tot nhat de buy hoac sell")
    assert parsed.requested_actions == ["BUY", "SELL"]
    assert parsed.needs_clarification is True
    assert "buy_or_sell_request" in parsed.ambiguities


def test_structured_llm_cannot_bypass_missing_ticker_clarification() -> None:
    class FakeStructuredProvider:
        def parse_structured(self, system, payload, output_model):
            del system, payload, output_model
            return SemanticQuery(
                original_query="", normalized_query="", intent=QueryIntent.STOCK_ANALYSIS,
                confidence=0.9, interpretation="analyze an unspecified ticker",
                needs_clarification=False,
            )

    parsed = QueryParser(llm_provider=FakeStructuredProvider()).parse("Phan tich ma")
    assert parsed.tickers == []
    assert parsed.needs_clarification is True
    assert parsed.clarification_question


def test_structured_llm_failure_is_logged_as_safe_fallback_reason() -> None:
    class FailingProvider:
        def parse_structured(self, system, payload, output_model):
            raise RuntimeError("network unavailable")

    parsed = QueryParser(llm_provider=FailingProvider()).parse("Phân tích FPT")
    assert parsed.parser_source == "semantic_similarity"
    assert parsed.llm_attempted is True
    assert parsed.fallback_reason == "llm_error:RuntimeError"


def test_semantic_llm_can_be_disabled_independently(monkeypatch) -> None:
    monkeypatch.setenv("STOCK_AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("STOCK_AGENT_LLM_SEMANTIC_ENABLED", "false")
    parser = QueryParser()
    assert parser.semantic.llm_provider is None

    parsed = parser.parse("Phan tich FPT")
    assert parsed.tickers == ["FPT"]
    assert parsed.llm_attempted is False
