from stock_multi_agent.chat.query_parser import QueryParser
from stock_multi_agent.chat.router import IntentRouter, RouteStep, RoutingPlan, RoutingPlanValidator, RoutingTask
from stock_multi_agent.models.chat import (
    ChatRole,
    ConversationMessage,
    ConversationState,
    ParsedQuery,
    QueryIntent,
)


def test_intent_ticker_and_period_extraction() -> None:
    parser = QueryParser()
    analysis = parser.parse("FPT 3 tháng tới có ổn không?")
    assert analysis.intent is QueryIntent.STOCK_ANALYSIS
    assert analysis.tickers == ["FPT"]
    assert analysis.investment_horizon == "3 months"

    chart = parser.parse("Cho tôi biểu đồ giá FPT 1 năm")
    assert chart.intent is QueryIntent.STOCK_CHART
    assert chart.tickers == ["FPT"]
    assert chart.chart_period == "1Y"


def test_common_intents() -> None:
    parser = QueryParser()
    cases = {
        "RSI FPT hiện thế nào?": QueryIntent.INDICATOR_QUERY,
        "FPT định giá có đắt không?": QueryIntent.FUNDAMENTAL_ANALYSIS,
        "FPT rủi ro không?": QueryIntent.RISK_ANALYSIS,
        "Tình hình chứng khoán Việt Nam hiện tại thế nào?": QueryIntent.MARKET_OVERVIEW,
        "Có mã ngân hàng nào đáng chú ý?": QueryIntent.STOCK_DISCOVERY,
        "So sánh FPT và CMG": QueryIntent.STOCK_COMPARISON,
    }
    for query, expected in cases.items():
        assert parser.parse(query).intent is expected


def test_follow_up_resolves_active_ticker_and_comparison() -> None:
    context = ConversationState(active_tickers=["FPT"], last_intent=QueryIntent.STOCK_ANALYSIS)
    parser = QueryParser()
    chart = parser.parse("Cho tôi biểu đồ 6 tháng", context)
    assert chart.tickers == ["FPT"]
    assert chart.chart_period == "6M"
    assert chart.follow_up is True

    technical = parser.parse("RSI thì sao?", context)
    assert technical.intent is QueryIntent.INDICATOR_QUERY
    assert technical.tickers == ["FPT"]

    comparison = parser.parse("So với CMG?", context)
    assert comparison.tickers == ["FPT", "CMG"]


def test_first_stock_reference_uses_previous_discovery() -> None:
    context = ConversationState(last_stock_list=["FPT", "CMG"], active_tickers=["FPT", "CMG"])
    parsed = QueryParser().parse("Cho tôi biểu đồ RSI của mã đầu tiên", context)
    assert parsed.intent is QueryIntent.STOCK_CHART
    assert parsed.tickers == ["FPT"]
    assert parsed.requested_indicators == ["RSI"]


def test_natural_macd_chart_request() -> None:
    context = ConversationState(active_tickers=["FPT"])
    parsed = QueryParser().parse("Cho tôi MACD", context)
    assert parsed.intent is QueryIntent.INDICATOR_QUERY
    assert parsed.tickers == ["FPT"]


def test_router_selects_only_needed_agents() -> None:
    parser = QueryParser()
    router = IntentRouter()
    chart_plan = router.plan(parser.parse("Biểu đồ FPT"))
    assert chart_plan.steps == [RouteStep.DATA, RouteStep.CHART]
    rsi_plan = router.plan(parser.parse("RSI FPT?"))
    assert rsi_plan.steps == [RouteStep.DATA, RouteStep.TECHNICAL]
    full_plan = router.plan(parser.parse("Phân tích FPT"))
    assert RouteStep.FUNDAMENTAL in full_plan.steps
    assert RouteStep.RISK in full_plan.steps
    recommendation_plan = router.plan(parser.parse("Tìm 10 mã tốt nhất để mua"))
    assert recommendation_plan.steps == [RouteStep.SCREENING, RouteStep.RECOMMENDATION]
    assert RoutingPlanValidator.validate(
        recommendation_plan, parser.parse("Tìm 10 mã tốt nhất để mua")
    ) == []


def test_routing_plan_has_typed_dependencies_and_bounded_budget() -> None:
    parsed = QueryParser().parse("Phân tích FPT")
    plan = IntentRouter().plan(parsed)
    tasks = {task.id: task for task in plan.tasks}
    assert tasks["technical"].depends_on == ["data"]
    assert set(tasks["risk"].depends_on) == {"data", "technical", "fundamental"}
    assert tasks["manager"].expected_outputs == ["final_decision"]
    assert plan.total_budget_units <= RoutingPlanValidator.max_total_budget_units
    assert RoutingPlanValidator.validate(plan, parsed) == []


def test_routing_plan_validator_rejects_forward_dependency_and_missing_input() -> None:
    parsed = QueryParser().parse("Phân tích FPT")
    invalid = RoutingPlan(
        intent=QueryIntent.STOCK_ANALYSIS,
        reason="unsafe test",
        tasks=[RoutingTask(
            id="technical",
            step=RouteStep.TECHNICAL,
            depends_on=["data"],
            required_inputs=["market_data"],
            expected_outputs=["technical_result"],
            timeout_seconds=15,
            budget_units=2,
        )],
    )
    errors = RoutingPlanValidator.validate(invalid, parsed)
    assert any("forward dependencies" in error for error in errors)
    assert any("missing required inputs" in error for error in errors)


def test_router_defensively_asks_for_clarification_when_ticker_is_missing() -> None:
    parsed = ParsedQuery(
        raw_query="Phan tich ma",
        intent=QueryIntent.STOCK_ANALYSIS,
        needs_clarification=False,
    )
    plan = IntentRouter().plan(parsed)
    assert plan.steps == [RouteStep.CHAT]
    assert RoutingPlanValidator.validate(plan, parsed) == []


def test_risk_secondary_expands_its_deterministic_dependencies() -> None:
    query = QueryParser().parse("Phân tích kỹ thuật FPT và rủi ro")
    plan = IntentRouter().plan(query)
    risk = next(task for task in plan.tasks if task.step is RouteStep.RISK)
    assert {"data", "technical", "fundamental"}.issubset(risk.depends_on)
    assert RoutingPlanValidator.validate(plan, query) == []


def test_research_intent_gets_an_evidence_task_without_market_data() -> None:
    query = QueryParser().parse("Tìm báo cáo tài chính FPT mới nhất")
    plan = IntentRouter().plan(query)
    assert query.intent is QueryIntent.RESEARCH
    assert plan.steps == [RouteStep.RESEARCH]
    assert plan.tasks[0].required_inputs == ["research_query"]


def test_conversation_bounds_raw_history() -> None:
    state = ConversationState(max_messages=4)
    for index in range(6):
        state.add_message(ConversationMessage(role=ChatRole.USER, content=f"message {index}"))
    assert len(state.messages) == 4
    assert "message 0" in state.summary
