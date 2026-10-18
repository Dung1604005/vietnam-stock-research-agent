from __future__ import annotations

import pytest

from stock_multi_agent.chat.query_parser import QueryParser
from stock_multi_agent.chat.router import IntentRouter, RouteStep, RoutingPlanValidator
from stock_multi_agent.models.chat import ConversationState, QueryIntent
from stock_multi_agent.services.request_telemetry import request_scope


@pytest.mark.parametrize("query,intent", [
    ("Dự báo giá FPT phiên tới", QueryIntent.FORECAST),
    ("Tóm tắt báo cáo FPT", QueryIntent.DOCUMENT_QA),
    ("Nguồn dữ liệu FPT", QueryIntent.SOURCE_REQUEST),
    ("Kiểm tra luận điểm đầu tư FPT", QueryIntent.THESIS_REVIEW),
])
def test_extended_intents_have_valid_typed_plans(query, intent):
    parsed = QueryParser(use_configured_provider=False).parse(query)
    assert parsed.intent == intent
    assert not parsed.needs_clarification
    plan = IntentRouter().plan(parsed)
    assert not RoutingPlanValidator.validate(plan, parsed)
    assert all(task.completion_condition for task in plan.tasks)
    assert RouteStep.FORECAST in plan.steps if intent == QueryIntent.FORECAST else RouteStep.RESEARCH in plan.steps


def test_composed_research_and_technical_have_executable_dependencies():
    parser = QueryParser(use_configured_provider=False)
    query = parser.parse("Tóm tắt báo cáo FPT và phân tích kỹ thuật")
    assert QueryIntent.TECHNICAL_ANALYSIS in query.secondary_intents
    plan = IntentRouter().plan(query)
    assert {RouteStep.RESEARCH, RouteStep.DATA, RouteStep.TECHNICAL}.issubset(plan.steps)
    assert not RoutingPlanValidator.validate(plan, query)


def test_cache_is_context_sensitive_and_does_not_return_mutable_shared_result(monkeypatch):
    monkeypatch.setenv("STOCK_AGENT_REQUEST_TELEMETRY_ENABLED", "false")
    parser = QueryParser(use_configured_provider=False)
    context = ConversationState(active_tickers=["FPT"], last_intent=QueryIntent.STOCK_ANALYSIS)
    first = parser.parse("Đổi thành 6 tháng", context)
    assert first.tickers == ["FPT"] and first.investment_horizon == "6 months"
    first.tickers.append("ABC")
    with request_scope() as metrics:
        second = parser.parse("Đổi thành 6 tháng", context)
        assert second.cache_hit and metrics.cache_hits == 1
        assert second.tickers == ["FPT"]
    context.active_tickers = ["VNM"]
    changed = parser.parse("Đổi thành 6 tháng", context)
    assert not changed.cache_hit and changed.tickers == ["VNM"]


def test_clear_configured_query_skips_llm_but_ambiguous_query_can_call_it(monkeypatch):
    import stock_multi_agent.nlu.semantic_parser as module
    calls = []
    class Provider:
        def parse_structured(self, *args):
            calls.append(1)
            raise RuntimeError("offline")
    monkeypatch.setattr(module, "configured_semantic_provider", lambda: Provider())
    parser = QueryParser(use_configured_provider=True)
    parser.parse("Tóm tắt báo cáo FPT")
    assert calls == []
    parser.parse("xyz qwerty")
    assert len(calls) == 1


def test_word_cardinality_alias_and_as_of():
    parser = QueryParser(use_configured_provider=False)
    assert parser.parse("Tìm mười mã cổ phiếu đáng chú ý").requested_result_count == 10
    parsed = parser.parse("Tìm báo cáo Techcombank trước 2025-12-31 trên HOSE")
    assert parsed.tickers == ["TCB"]
    assert parsed.as_of.isoformat() == "2025-12-31"
    assert parsed.exchange == "HOSE"
    assert parser.parse("Báo cáo FPT ngày 2025-02-31").needs_clarification


def test_theory_uses_only_knowledge_and_portfolio_requires_input():
    parser = QueryParser(use_configured_provider=False)
    theory = parser.parse("RSI là gì?")
    assert IntentRouter().plan(theory).steps == [RouteStep.KNOWLEDGE]
    portfolio = parser.parse("Đánh giá rủi ro danh mục của tôi")
    assert portfolio.intent == QueryIntent.PORTFOLIO_RISK
    assert portfolio.needs_clarification
    assert IntentRouter().plan(portfolio).steps == [RouteStep.CHAT]
