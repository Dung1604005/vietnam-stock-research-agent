"""Concrete task handlers for chat plans; no intent-only analysis dispatch."""
from __future__ import annotations

from typing import TYPE_CHECKING

from stock_multi_agent.chat.response import ResponseComposer
from stock_multi_agent.chat.router import IntentRouter, RouteStep, RoutingPlan
from stock_multi_agent.models.chat import AssistantResponse, ParsedQuery, QueryIntent
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.plan_executor import PlanExecutor, TaskOutputError

if TYPE_CHECKING:
    from stock_multi_agent.services.chat_service import ChatService


class ChatPlanExecution:
    """Run each approved capability, sharing data between secondary outputs.

    The supervised full-analysis graph is one explicit composite so its parallel
    branches and bounded review loops retain their own fault-handling semantics.
    Screening, recommendation and workbench services similarly own bounded batch
    or research sub-operations declared by their fixed execution contracts.
    """

    def __init__(self, service: ChatService, plan: RoutingPlan, query: ParsedQuery):
        self.service, self.plan, self.query = service, plan, query
        self.executor = PlanExecutor(plan)
        self.states: dict[str, AgentState] = {}
        self.responses: dict[RouteStep, AssistantResponse] = {}
        self.response: AssistantResponse | None = None

    def execute(self, conversation) -> AssistantResponse:
        service, query, intent = self.service, self.query, self.plan.intent
        if query.as_of and intent not in {
            QueryIntent.RESEARCH, QueryIntent.DOCUMENT_QA, QueryIntent.SOURCE_REQUEST, QueryIntent.THESIS_REVIEW,
            QueryIntent.THEORY, QueryIntent.CURRENT_RULE, QueryIntent.HELP,
        }:
            return AssistantResponse(message=(
                f"Yêu cầu phân tích tại ngày {query.as_of.isoformat()} cần dữ liệu point-in-time. "
                "Luồng chat hiện hỗ trợ lọc ngày cho tài liệu; bạn có thể hỏi báo cáo được công bố trước ngày này."
            ), metadata={"status": "UNSUPPORTED_POINT_IN_TIME", "as_of": query.as_of.isoformat()})

        if self.plan.steps != [RouteStep.CHAT]:
            self._run_primary_composite(conversation)
        for task in self.plan.tasks:
            if task.id in self.executor.records:
                continue
            self.executor.run(task, lambda step=task.step: self._run_task(step, conversation))
        response = self.response or self._compose_primary()
        self._merge_requested_secondary(response)
        return response

    def can_enrich(self) -> bool:
        from stock_multi_agent.services.execution_profiles import current_profile
        from stock_multi_agent.services.request_telemetry import remaining_seconds
        remaining = remaining_seconds()
        analytical = {RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL, RouteStep.RISK, RouteStep.MANAGER}
        return current_profile().max_llm_calls != 0 and (remaining is None or remaining > 0) and bool(analytical.intersection(self.plan.steps)) and bool(self.executor.records) and all(
            record["status"] == "COMPLETED" for record in self.executor.records.values())

    def available_response(self) -> AssistantResponse | None:
        """Retain an already validated primary result if a later task fails."""
        has_completed_work = any(record["status"] in {"COMPLETED", "PARTIAL"}
            for record in self.executor.records.values())
        if self.response is not None and self.executor.outputs and has_completed_work:
            return self.response
        if self.states and self.executor.outputs and has_completed_work:
            from stock_multi_agent.services.request_telemetry import BudgetExceeded
            try:
                return self._compose_primary()
            except (KeyError, TaskOutputError, ValueError, BudgetExceeded):
                state = next(iter(self.states.values()))
                if state.market_data is not None:
                    return ResponseComposer.partial_analysis(state)
        return None

    def _run_primary_composite(self, conversation) -> None:
        query, service, intent = self.query, self.service, self.plan.intent
        graph_steps = {RouteStep.DATA, RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL, RouteStep.RISK, RouteStep.MANAGER}
        if intent == QueryIntent.STOCK_ANALYSIS or (intent == QueryIntent.EXPLANATION and query.tickers):
            def full_analysis():
                state = service.analysis.analyze(query.tickers[0], query.investment_horizon)
                self.states[state.ticker] = state
                self.response = service._full_analysis_response(query, state, capture=intent == QueryIntent.STOCK_ANALYSIS)
                outputs = self._state_outputs(state)
                if RouteStep.CHART in self.plan.steps:
                    # DATA's composite scope includes the other explicitly
                    # requested chart symbols. CHART itself only renders.
                    from stock_multi_agent.services.financial_charts import FINANCIAL_TYPES
                    for symbol in query.tickers[:4]:
                        if symbol in self.states:
                            continue
                        financial = any(item.chart_type in FINANCIAL_TYPES
                            for item in service.charts.requests_from_query(query, symbol))
                        extra = service.analysis.create_state(symbol, query.investment_horizon,
                            required_data=["ohlcv", *(["financials"] if financial else [])])
                        try:
                            self.states[symbol] = service.analysis.run_stage(extra, "data")
                        except Exception as exc:
                            self.response.warnings.append(f"Chart data for {symbol} unavailable ({type(exc).__name__}).")
                            outputs["_task_statuses"]["DATA"] = "PARTIAL"
                    outputs["market_data"] = {symbol: value.market_data for symbol, value in self.states.items()}
                return outputs
            tasks = [task for task in self.plan.tasks if task.step in graph_steps]
            self.executor.run_composite(tasks, full_analysis, contract="supervised-full-analysis-graph-v1")
        elif intent == QueryIntent.STOCK_COMPARISON:
            def compare():
                states = []
                self.response = service._compare(query.tickers, query.investment_horizon, states_out=states)
                self.states.update({state.ticker: state for state in states})
                return {"market_data": [state.market_data for state in states],
                    "market_data_collection": [state.market_data for state in states],
                    **{key: [getattr(state, key) for state in states] for key in
                        ("technical_result", "fundamental_result", "risk_result", "final_decision")},
                    "_task_statuses": {step.value: self._collection_status(states, step) for step in graph_steps},
                    "comparison": self.response}
            tasks = [task for task in self.plan.tasks if task.step in graph_steps | {RouteStep.COMPARISON}]
            self.executor.run_composite(tasks, compare, contract="bounded-full-analysis-comparison-v1")
        elif intent in {QueryIntent.STOCK_DISCOVERY, QueryIntent.STOCK_RECOMMENDATION}:
            steps = {RouteStep.SCREENING, RouteStep.RECOMMENDATION}
            def screen_or_recommend():
                self.response = service._dispatch_composite(query, intent)
                return {"candidate_tickers": list(self.response.tickers),
                    "dual_recommendation": self.response.metadata.get("dual_recommendation")}
            self.executor.run_composite([task for task in self.plan.tasks if task.step in steps],
                screen_or_recommend, contract="bounded-screening-recommendation-v1")
        elif intent in {QueryIntent.THESIS_REVIEW, QueryIntent.PORTFOLIO_RISK, QueryIntent.EVENT_IMPACT,
            QueryIntent.CHANGE_EXPLANATION, QueryIntent.REPORT_CHANGE, QueryIntent.PEER_COMPARISON,
            QueryIntent.SCENARIO_ANALYSIS, QueryIntent.REPORT_EXPORT}:
            def workbench():
                if intent == QueryIntent.REPORT_EXPORT:
                    previous = next((item.response for item in reversed(conversation.messages) if item.response), None)
                    self.response = previous.model_copy(deep=True) if previous else AssistantResponse(
                        message="Hãy phân tích hoặc tìm tài liệu trước, sau đó chọn tải báo cáo Markdown/HTML dưới câu trả lời.")
                    self.response.metadata["export_ready"] = True
                else:
                    self.response = service._dispatch_composite(query, intent)
                return {"research_tool_result": self.response,
                    "evidence_bundle": self.response.metadata.get("evidence_bundle")}
            steps = {RouteStep.WORKBENCH}
            if intent == QueryIntent.THESIS_REVIEW:
                steps.add(RouteStep.RESEARCH)
            self.executor.run_composite([task for task in self.plan.tasks if task.step in steps],
                workbench, contract="bounded-workbench-use-case-v1")

    def _run_task(self, step: RouteStep, conversation) -> dict:
        query, service = self.query, self.service
        if step == RouteStep.CHAT:
            if self.plan.intent == QueryIntent.STOCK_COMPARISON and len(query.tickers) > 5:
                response = AssistantResponse(message="Mỗi lần so sánh tối đa 5 mã; hãy chọn 2–5 mã để giữ ngân sách truy xuất.", tickers=query.tickers)
            elif query.needs_clarification or (query.intent not in {
                QueryIntent.GENERAL_CHAT, QueryIntent.HELP, QueryIntent.UNKNOWN, QueryIntent.EXPLANATION}):
                response = AssistantResponse(message=query.clarification_question or
                    "Bạn có thể nêu rõ mã cổ phiếu và yêu cầu cần thực hiện không?", tickers=query.tickers)
            else:
                response = service._dispatch_composite(query, self.plan.intent)
            self.responses[step] = response
            return {"chat_response": response}
        if step == RouteStep.DATA:
            symbols = self._data_symbols()
            financial = bool({RouteStep.FUNDAMENTAL, RouteStep.RISK}.intersection(self.plan.steps))
            if RouteStep.CHART in self.plan.steps:
                from stock_multi_agent.services.financial_charts import FINANCIAL_TYPES
                financial = financial or any(request.chart_type in FINANCIAL_TYPES for symbol in symbols
                    for request in service.charts.requests_from_query(query, symbol))
            required = ["ohlcv", *(["financials"] if financial else []),
                *(["benchmark"] if RouteStep.RISK in self.plan.steps else [])]
            for symbol in symbols:
                state = service.analysis.create_state(symbol, query.investment_horizon, required_data=required)
                state.requested_indicators = list(query.requested_indicators)
                state = service.analysis.run_stage(state, "data")
                # DATA's fixed contract permits the existing optional forecast.
                # Explicit FORECAST requests execute it in their own next task.
                if self.plan.intent != QueryIntent.STOCK_CHART and RouteStep.FORECAST not in self.plan.steps:
                    state = service.analysis.run_stage(state, "forecast")
                self.states[symbol] = state
            return {"market_data": {symbol: state.market_data for symbol, state in self.states.items()}}
        if step in {RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL, RouteStep.RISK, RouteStep.FORECAST}:
            state = self._state()
            if step == RouteStep.FUNDAMENTAL:
                state = service.analysis.run_stage(state, "research")
            options = {}
            if step == RouteStep.TECHNICAL:
                options["allow_llm_planning"] = state.allow_llm_planning
            elif step in {RouteStep.FUNDAMENTAL, RouteStep.RISK}:
                options["allow_llm_review"] = state.allow_llm_planning
            state = service.analysis.run_stage(state, step.value.lower(), **options)
            self.states[state.ticker] = state
            output_key = "price_forecast" if step == RouteStep.FORECAST else f"{step.value.lower()}_result"
            return {output_key: getattr(state, output_key), "_task_statuses": {
                step.value: self._stage_status(state, step)}}
        if step == RouteStep.KNOWLEDGE:
            response = service.knowledge.answer(query.raw_query, current_rule=self.plan.intent == QueryIntent.CURRENT_RULE)
            self.responses[step] = response
            return {"knowledge_evidence": response}
        if step == RouteStep.RESEARCH:
            response = service._research(query)
            self.responses[step] = response
            return {"evidence_bundle": response.metadata.get("evidence_bundle")}
        if step == RouteStep.CHART:
            response = service._chart(query, states=self.states)
            self.responses[step] = response
            return {"chart_artifact": response.charts}
        if step == RouteStep.MARKET:
            response = service._market_overview(query.investment_horizon, state=self.states.get("VNINDEX"))
            self.responses[step] = response
            return {"market_overview": response}
        raise TaskOutputError(f"No executable handler for {step.value}.")

    def _data_symbols(self) -> list[str]:
        if self.plan.intent == QueryIntent.MARKET_OVERVIEW:
            return ["VNINDEX"]
        if self.plan.intent in {QueryIntent.STOCK_DISCOVERY, QueryIntent.STOCK_RECOMMENDATION}:
            return list(self.response.tickers[:4]) if self.response else []
        return self.query.tickers[:4] if RouteStep.CHART in self.plan.steps else self.query.tickers[:1]

    def _state(self) -> AgentState:
        if not self.states:
            raise TaskOutputError("The approved DATA task did not produce a usable state.")
        symbol = self.query.tickers[0] if self.query.tickers else next(iter(self.states))
        return self.states[symbol]

    @staticmethod
    def _state_outputs(state: AgentState) -> dict:
        return {**{key: getattr(state, key) for key in
            ("market_data", "technical_result", "fundamental_result", "risk_result", "final_decision")},
            "_task_statuses": {step.value: ChatPlanExecution._stage_status(state, step) for step in
                (RouteStep.DATA, RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL, RouteStep.RISK, RouteStep.MANAGER)}}

    @staticmethod
    def _stage_status(state: AgentState, step: RouteStep) -> str:
        executions = [item for item in state.task_executions if item.agent.value == step.value.lower()
            and not item.node.startswith("forecast")]
        return executions[-1].status if executions else "COMPLETED"

    @staticmethod
    def _collection_status(states: list[AgentState], step: RouteStep) -> str:
        statuses = {ChatPlanExecution._stage_status(state, step) for state in states}
        if not statuses:
            return "UNAVAILABLE"
        return next(iter(statuses)) if len(statuses) == 1 else "PARTIAL"

    def _specialist_response(self, step: RouteStep) -> AssistantResponse:
        state = self._state()
        composer = {RouteStep.TECHNICAL: ResponseComposer.technical,
            RouteStep.FUNDAMENTAL: ResponseComposer.fundamental, RouteStep.RISK: ResponseComposer.risk}[step]
        if getattr(state, f"{step.value.lower()}_result") is None:
            return ResponseComposer.partial_analysis(state)
        response = composer(state)
        self.service._attach_specialist_llm_reviews(response, state, {step.value.lower()})
        return response

    def _compose_primary(self) -> AssistantResponse:
        intent = self.plan.intent
        if self.plan.steps == [RouteStep.CHAT]:
            return self.responses[RouteStep.CHAT]
        fundamental = {"PE", "PB", "ROE", "ROA", "EPS"}
        if intent in {QueryIntent.TECHNICAL_ANALYSIS, QueryIntent.FUNDAMENTAL_ANALYSIS,
            QueryIntent.RISK_ANALYSIS, QueryIntent.INDICATOR_QUERY}:
            step = {QueryIntent.TECHNICAL_ANALYSIS: RouteStep.TECHNICAL,
                QueryIntent.FUNDAMENTAL_ANALYSIS: RouteStep.FUNDAMENTAL,
                QueryIntent.RISK_ANALYSIS: RouteStep.RISK}.get(intent)
            step = step or (RouteStep.FUNDAMENTAL if fundamental.intersection(self.query.requested_indicators) else RouteStep.TECHNICAL)
            return self._specialist_response(step)
        if intent == QueryIntent.MIXED:
            response = self.responses[RouteStep.KNOWLEDGE]
            base_query = self.query.model_copy(update={"secondary_intents": []})
            step = IntentRouter.required_steps(base_query, intent)[-1]
            applied = self._specialist_response(step)
            applied.message = f"### Áp dụng vào dữ liệu thật · {self.query.tickers[0]}\n\n{applied.message}"
            self.service._merge_secondary_response(response, applied, step.value.lower())
            response.tickers = self.query.tickers[:1]
            response.metadata.update(mixed_components=["knowledge", step.value.lower()],
                knowledge_does_not_control_recommendation=True)
            return response
        if intent == QueryIntent.FORECAST:
            state = self._state()
            return AssistantResponse(message=ResponseComposer._forecast_summary(state), tickers=self.query.tickers,
                sources=ResponseComposer._sources(state), source_details=ResponseComposer._source_details(state),
                data_timestamp=state.market_data.as_of if state.market_data else None,
                data_fetched_at=state.market_data.fetched_at if state.market_data else None,
                agent_trace=state.trace, metadata={"price_forecast": ResponseComposer._forecast_metadata(state)})
        primary = {QueryIntent.STOCK_CHART: RouteStep.CHART, QueryIntent.MARKET_OVERVIEW: RouteStep.MARKET,
            QueryIntent.THEORY: RouteStep.KNOWLEDGE, QueryIntent.CURRENT_RULE: RouteStep.KNOWLEDGE,
            QueryIntent.RESEARCH: RouteStep.RESEARCH, QueryIntent.DOCUMENT_QA: RouteStep.RESEARCH,
            QueryIntent.SOURCE_REQUEST: RouteStep.RESEARCH}.get(intent, RouteStep.CHAT)
        return self.responses[primary]

    def _merge_requested_secondary(self, response: AssistantResponse) -> None:
        if self.query.as_of or self.plan.steps == [RouteStep.CHAT]:
            return
        secondary_steps = {QueryIntent.RESEARCH: RouteStep.RESEARCH, QueryIntent.DOCUMENT_QA: RouteStep.RESEARCH,
            QueryIntent.TECHNICAL_ANALYSIS: RouteStep.TECHNICAL, QueryIntent.STOCK_CHART: RouteStep.CHART,
            QueryIntent.FUNDAMENTAL_ANALYSIS: RouteStep.FUNDAMENTAL, QueryIntent.RISK_ANALYSIS: RouteStep.RISK,
            QueryIntent.INDICATOR_QUERY: RouteStep.FUNDAMENTAL if {"PE", "PB", "ROE", "ROA", "EPS"}.intersection(
                self.query.requested_indicators) else RouteStep.TECHNICAL}
        for intent in dict.fromkeys(self.query.secondary_intents):
            step = secondary_steps.get(intent)
            if step is None:
                continue
            if step not in self.plan.steps:
                raise TaskOutputError(f"Secondary output {step.value} is absent from the approved plan.")
            if step in {RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL, RouteStep.RISK}:
                if f"{step.value.lower()}_score" in response.metadata:
                    continue
                secondary = self._specialist_response(step)
            else:
                secondary = self.responses.get(step)
                if secondary is None or secondary is response:
                    continue
            self.service._merge_secondary_response(response, secondary, step.value.lower())
