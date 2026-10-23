from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from stock_multi_agent.models.chat import ConversationState, ParsedQuery, QueryIntent


class RouteStep(str, Enum):
    DATA = "DATA"
    TECHNICAL = "TECHNICAL"
    FUNDAMENTAL = "FUNDAMENTAL"
    RISK = "RISK"
    MANAGER = "MANAGER"
    CHART = "CHART"
    MARKET = "MARKET"
    SCREENING = "SCREENING"
    RECOMMENDATION = "RECOMMENDATION"
    COMPARISON = "COMPARISON"
    RESEARCH = "RESEARCH"
    KNOWLEDGE = "KNOWLEDGE"
    CHAT = "CHAT"
    FORECAST = "FORECAST"
    WORKBENCH = "WORKBENCH"


class RoutingTask(BaseModel):
    """A bounded, inspectable unit in the execution plan.

    `RouteStep` is deliberately an enum rather than a model/agent name. This
    keeps semantic understanding from authorizing arbitrary functions or tools.
    """

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    step: RouteStep
    depends_on: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list, min_length=1)
    timeout_seconds: int = Field(ge=1, le=120)
    budget_units: int = Field(ge=0, le=10)
    execution_contract: str = ""
    nested_operations: list[str] = Field(default_factory=list)
    completion_condition: str = "All expected_outputs are present and schema-valid."


class RoutingPlan(BaseModel):
    schema_version: str = "task-plan-v1"
    intent: QueryIntent
    tasks: list[RoutingTask] = Field(min_length=1, max_length=10)
    reason: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def steps(self) -> list[RouteStep]:
        """Compatibility view used by the UI and existing callers."""
        return [task.step for task in self.tasks]

    @property
    def total_budget_units(self) -> int:
        return sum(task.budget_units for task in self.tasks)


class RoutingPlanValidationError(ValueError):
    pass


# Public contract names reuse the established models instead of duplicating schemas.
UserIntent = ParsedQuery
TaskPlan = RoutingPlan


class RoutingPlanValidator:
    """Deterministic safety boundary between semantic parsing and execution."""

    allowed_steps = frozenset(RouteStep)
    max_total_budget_units = 30

    @classmethod
    def validate(cls, plan: RoutingPlan, query: ParsedQuery, context: ConversationState | None = None) -> list[str]:
        errors: list[str] = []
        if plan.schema_version != "task-plan-v1":
            errors.append("Unsupported task plan schema version.")
        intent = query.intent
        if intent is QueryIntent.FOLLOW_UP:
            intent = context.last_intent if context and context.last_intent else QueryIntent.STOCK_ANALYSIS
        if plan.intent is not intent:
            errors.append("Task plan intent does not match the resolved user intent.")
        expected_steps = IntentRouter.required_steps(query, intent)
        if set(plan.steps) != set(expected_steps):
            errors.append("Task plan capabilities do not match the requested route: "
                + ", ".join(step.value for step in expected_steps) + ".")
        if len(set(plan.steps)) != len(plan.steps):
            errors.append("Each route capability may appear only once.")
        expected_tasks = {task.step: task for task in IntentRouter()._tasks(intent, expected_steps)}
        expected_by_id = {task.id: task for task in expected_tasks.values()}
        by_id = {task.id: task for task in plan.tasks}
        for task in plan.tasks:
            expected = expected_tasks.get(task.step)
            if expected is None:
                continue
            if task.execution_contract != expected.execution_contract:
                errors.append(f"Task {task.id} has an unsupported execution contract.")
            if set(task.nested_operations) != set(expected.nested_operations):
                errors.append(f"Task {task.id} changes its approved nested operations.")
            if set(task.required_inputs) != set(expected.required_inputs) or set(task.expected_outputs) != set(expected.expected_outputs):
                errors.append(f"Task {task.id} does not declare its capability's required input/output contract.")
            required_dependencies = {expected_by_id[dep].step for dep in expected.depends_on}
            actual_dependencies = {by_id[dep].step for dep in task.depends_on if dep in by_id}
            if required_dependencies != actual_dependencies:
                errors.append(f"Task {task.id} changes its registered capability dependencies.")
            if task.budget_units < expected.budget_units:
                errors.append(f"Task {task.id} underfunds its bounded execution contract.")
            if task.completion_condition != expected.completion_condition:
                errors.append(f"Task {task.id} changes the registered completion condition.")
        task_ids = [task.id for task in plan.tasks]
        if len(set(task_ids)) != len(task_ids):
            errors.append("Task IDs must be unique.")
        if plan.total_budget_units > cls.max_total_budget_units:
            errors.append("Task plan exceeds the bounded execution budget.")

        available_inputs = {"user_message", "research_query"}
        if query.tickers:
            available_inputs.add("ticker")
        if len(query.tickers) >= 2:
            available_inputs.add("tickers")
        if plan.intent == QueryIntent.MARKET_OVERVIEW:
            available_inputs.add("market_index")
        if plan.intent in {QueryIntent.STOCK_DISCOVERY, QueryIntent.STOCK_RECOMMENDATION}:
            available_inputs.add("universe")
        if plan.intent == QueryIntent.RESEARCH:
            available_inputs.add("research_query")

        completed: set[str] = set()
        produced = set(available_inputs)
        for task in plan.tasks:
            if task.step not in cls.allowed_steps:
                errors.append(f"Task {task.id} uses a disallowed route step.")
            if task.id in task.depends_on:
                errors.append(f"Task {task.id} cannot depend on itself.")
            missing_dependencies = set(task.depends_on).difference(completed)
            if missing_dependencies:
                errors.append(f"Task {task.id} has missing or forward dependencies: {sorted(missing_dependencies)}.")
            missing_inputs = set(task.required_inputs).difference(produced)
            if missing_inputs:
                errors.append(f"Task {task.id} is missing required inputs: {sorted(missing_inputs)}.")
            completed.add(task.id)
            produced.update(task.expected_outputs)
        return errors

    @classmethod
    def assert_valid(cls, plan: RoutingPlan, query: ParsedQuery, context: ConversationState | None = None) -> None:
        errors = cls.validate(plan, query, context)
        if errors:
            raise RoutingPlanValidationError(" ".join(errors))


class IntentRouter:
    """Converts a typed intent into the smallest safe, executable task plan."""

    _TASK_DEFAULTS: dict[RouteStep, tuple[list[str], list[str], int, int]] = {
        RouteStep.DATA: (["ticker"], ["market_data"], 30, 3),
        RouteStep.TECHNICAL: (["market_data"], ["technical_result"], 15, 2),
        RouteStep.FUNDAMENTAL: (["market_data"], ["fundamental_result"], 15, 2),
        RouteStep.RISK: (["market_data", "technical_result", "fundamental_result"], ["risk_result"], 15, 2),
        RouteStep.MANAGER: (["technical_result", "fundamental_result", "risk_result"], ["final_decision"], 10, 1),
        RouteStep.CHART: (["market_data"], ["chart_artifact"], 15, 1),
        RouteStep.MARKET: (["market_data"], ["market_overview"], 15, 1),
        RouteStep.SCREENING: (["universe"], ["candidate_tickers"], 30, 3),
        RouteStep.RECOMMENDATION: (["candidate_tickers"], ["dual_recommendation"], 120, 6),
        RouteStep.COMPARISON: (["market_data_collection", "final_decision"], ["comparison"], 30, 3),
        RouteStep.RESEARCH: (["research_query"], ["evidence_bundle"], 20, 2),
        RouteStep.KNOWLEDGE: (["user_message"], ["knowledge_evidence"], 10, 1),
        RouteStep.CHAT: (["user_message"], ["chat_response"], 5, 0),
        RouteStep.FORECAST: (["market_data"], ["price_forecast"], 30, 2),
        RouteStep.WORKBENCH: (["user_message"], ["research_tool_result"], 120, 3),
    }

    # These are executable contracts, including the bounded sub-operations of
    # a composite capability. They are not arbitrary agent/tool names supplied
    # by a semantic parser. A plan cannot invent or narrow one after approval.
    _CONTRACTS = {
        RouteStep.DATA: "market-data-and-optional-forecast-v1",
        RouteStep.TECHNICAL: "technical-with-bounded-review-v1",
        RouteStep.FUNDAMENTAL: "fundamental-with-research-context-v1",
        RouteStep.RISK: "risk-with-bounded-review-v1",
        RouteStep.MANAGER: "supervised-graph-review-reanalysis-finalize-v1",
        RouteStep.CHART: "render-approved-charts-v1",
        RouteStep.MARKET: "compose-market-overview-v1",
        RouteStep.SCREENING: "bounded-universe-data-indicator-screen-v1",
        RouteStep.RECOMMENDATION: "bounded-candidate-full-analysis-ranking-v1",
        RouteStep.COMPARISON: "compare-full-analysis-collection-v1",
        RouteStep.RESEARCH: "retrieve-evidence-bundle-v1",
        RouteStep.KNOWLEDGE: "local-knowledge-answer-v1",
        RouteStep.CHAT: "local-chat-or-clarification-v1",
        RouteStep.FORECAST: "next-session-price-forecast-v1",
        RouteStep.WORKBENCH: "bounded-research-workbench-v1",
    }

    _OPERATIONS = {
        RouteStep.DATA: ["fetch_market_data", "optional_price_forecast", "discover_source_citation_links"],
        RouteStep.TECHNICAL: ["technical_analysis", "bounded_indicator_planning", "optional_response_narrative"],
        RouteStep.FUNDAMENTAL: ["retrieve_research_context", "fundamental_analysis", "bounded_specialist_review", "optional_response_narrative"],
        RouteStep.RISK: ["risk_analysis", "bounded_specialist_review", "optional_response_narrative"],
        RouteStep.MANAGER: ["manager_plan", "dispatch_specialists", "bounded_reanalysis", "bounded_data_refresh",
            "bounded_research_follow_up", "bounded_forecast_refresh", "decision_finalize", "response_synthesis",
            "persist_research_snapshot", "event_impact"],
        RouteStep.CHART: ["render_charts_from_approved_data"],
        RouteStep.MARKET: ["compose_market_overview"],
        RouteStep.SCREENING: ["select_bounded_universe", "fetch_candidate_data", "optional_price_forecast", "evaluate_candidate_indicators"],
        RouteStep.RECOMMENDATION: ["bounded_candidate_full_analysis", "rank_dual_recommendation"],
        RouteStep.COMPARISON: ["compare_full_analysis_collection", "peer_percentiles"],
        RouteStep.RESEARCH: ["retrieve_evidence_bundle", "bounded_research_synthesis"],
        RouteStep.KNOWLEDGE: ["local_knowledge_answer"], RouteStep.CHAT: ["local_chat_or_clarification"],
        RouteStep.FORECAST: ["next_session_price_forecast"],
        RouteStep.WORKBENCH: ["fetch_research_data", "bounded_research_fallback", "bounded_peer_full_analysis",
            "read_persisted_snapshots", "track_thesis", "compute_research_tool", "prepare_report_export"],
    }

    def plan(self, query: ParsedQuery, context: ConversationState | None = None) -> RoutingPlan:
        intent = query.intent
        if intent == QueryIntent.FOLLOW_UP:
            intent = context.last_intent if context and context.last_intent else QueryIntent.STOCK_ANALYSIS
        steps = self.required_steps(query, intent)
        plan = RoutingPlan(intent=intent, tasks=self._tasks(intent, steps),
            reason=f"{intent.value} needs {' → '.join(step.value for step in steps)}", confidence=query.confidence)
        RoutingPlanValidator.assert_valid(plan, query, context)
        return plan

    @classmethod
    def required_steps(cls, query: ParsedQuery, intent: QueryIntent) -> list[RouteStep]:
        if query.needs_clarification:
            return [RouteStep.CHAT]
        single_ticker_intents = {
            QueryIntent.STOCK_ANALYSIS, QueryIntent.TECHNICAL_ANALYSIS,
            QueryIntent.FUNDAMENTAL_ANALYSIS, QueryIntent.RISK_ANALYSIS,
            QueryIntent.STOCK_CHART, QueryIntent.INDICATOR_QUERY, QueryIntent.MIXED, QueryIntent.FORECAST,
        }
        if (intent in single_ticker_intents and not query.tickers) or (
            intent is QueryIntent.STOCK_COMPARISON and not 2 <= len(query.tickers) <= 5
        ):
            return [RouteStep.CHAT]
        routes = {
            QueryIntent.STOCK_ANALYSIS: [RouteStep.DATA, RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL, RouteStep.RISK, RouteStep.MANAGER],
            QueryIntent.TECHNICAL_ANALYSIS: [RouteStep.DATA, RouteStep.TECHNICAL],
            QueryIntent.FUNDAMENTAL_ANALYSIS: [RouteStep.DATA, RouteStep.FUNDAMENTAL],
            QueryIntent.RISK_ANALYSIS: [RouteStep.DATA, RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL, RouteStep.RISK],
            QueryIntent.MARKET_OVERVIEW: [RouteStep.DATA, RouteStep.MARKET],
            QueryIntent.STOCK_DISCOVERY: [RouteStep.SCREENING],
            QueryIntent.STOCK_RECOMMENDATION: [RouteStep.SCREENING, RouteStep.RECOMMENDATION],
            QueryIntent.STOCK_COMPARISON: [RouteStep.DATA, RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL, RouteStep.RISK, RouteStep.MANAGER, RouteStep.COMPARISON],
            QueryIntent.STOCK_CHART: [RouteStep.DATA, RouteStep.CHART],
            QueryIntent.INDICATOR_QUERY: [RouteStep.DATA, RouteStep.TECHNICAL],
            QueryIntent.RESEARCH: [RouteStep.RESEARCH],
            QueryIntent.DOCUMENT_QA: [RouteStep.RESEARCH],
            QueryIntent.THESIS_REVIEW: [RouteStep.RESEARCH, RouteStep.WORKBENCH],
            QueryIntent.SOURCE_REQUEST: [RouteStep.RESEARCH],
            QueryIntent.FORECAST: [RouteStep.DATA, RouteStep.FORECAST],
            QueryIntent.PORTFOLIO_RISK: [RouteStep.WORKBENCH],
            QueryIntent.EVENT_IMPACT: [RouteStep.WORKBENCH],
            QueryIntent.CHANGE_EXPLANATION: [RouteStep.WORKBENCH],
            QueryIntent.REPORT_CHANGE: [RouteStep.WORKBENCH],
            QueryIntent.PEER_COMPARISON: [RouteStep.WORKBENCH],
            QueryIntent.SCENARIO_ANALYSIS: [RouteStep.WORKBENCH],
            QueryIntent.REPORT_EXPORT: [RouteStep.WORKBENCH],
            QueryIntent.THEORY: [RouteStep.KNOWLEDGE],
            QueryIntent.CURRENT_RULE: [RouteStep.KNOWLEDGE],
            QueryIntent.MIXED: [RouteStep.KNOWLEDGE, RouteStep.DATA, RouteStep.FUNDAMENTAL],
            QueryIntent.EXPLANATION: [RouteStep.CHAT],
            QueryIntent.GENERAL_CHAT: [RouteStep.CHAT],
            QueryIntent.HELP: [RouteStep.CHAT],
            QueryIntent.UNKNOWN: [RouteStep.CHAT],
        }
        steps = list(routes.get(intent, [RouteStep.CHAT]))
        if intent == QueryIntent.STOCK_RECOMMENDATION and (query.filters or query.exchange):
            # A predicate screen never executes the deep recommendation service.
            steps = [RouteStep.SCREENING]
        if intent == QueryIntent.EXPLANATION and query.tickers:
            steps = [RouteStep.DATA, RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL, RouteStep.RISK, RouteStep.MANAGER]
        fundamental_indicators = {"PE", "PB", "ROE", "ROA", "EPS"}
        if intent == QueryIntent.INDICATOR_QUERY and fundamental_indicators.intersection(query.requested_indicators):
            steps = [RouteStep.DATA, RouteStep.FUNDAMENTAL]
        if intent == QueryIntent.MIXED:
            fundamental = bool(fundamental_indicators.intersection(query.requested_indicators)) or any(
                item in query.normalized_query for item in ("dinh gia", "dcf", "ev ebitda", "bien loi nhuan"))
            steps = [RouteStep.KNOWLEDGE, RouteStep.DATA,
                RouteStep.FUNDAMENTAL if fundamental else RouteStep.TECHNICAL]
        secondary_routes = {
            QueryIntent.TECHNICAL_ANALYSIS: [RouteStep.TECHNICAL],
            QueryIntent.STOCK_CHART: [RouteStep.CHART],
            QueryIntent.INDICATOR_QUERY: [RouteStep.TECHNICAL],
            QueryIntent.RISK_ANALYSIS: [RouteStep.RISK],
            QueryIntent.FUNDAMENTAL_ANALYSIS: [RouteStep.FUNDAMENTAL],
            QueryIntent.RESEARCH: [RouteStep.RESEARCH],
            QueryIntent.DOCUMENT_QA: [RouteStep.RESEARCH],
        }
        for secondary in query.secondary_intents:
            selected = secondary_routes.get(secondary, [])
            if secondary == QueryIntent.INDICATOR_QUERY and fundamental_indicators.intersection(query.requested_indicators):
                selected = [RouteStep.FUNDAMENTAL]
            for step in selected:
                if step not in steps:
                    steps.append(step)
        expanded = cls._with_required_steps(steps)
        if RouteStep.DATA in expanded and not query.tickers and intent not in {
            QueryIntent.STOCK_DISCOVERY, QueryIntent.STOCK_RECOMMENDATION, QueryIntent.MARKET_OVERVIEW}:
            return [RouteStep.CHAT]
        return expanded

    @staticmethod
    def _with_required_steps(steps: list[RouteStep]) -> list[RouteStep]:
        """Expand a selected specialist only when its deterministic inputs need it."""
        result: list[RouteStep] = []
        prerequisites = {RouteStep.TECHNICAL: [RouteStep.DATA], RouteStep.FUNDAMENTAL: [RouteStep.DATA],
            RouteStep.FORECAST: [RouteStep.DATA], RouteStep.CHART: [RouteStep.DATA],
            RouteStep.RISK: [RouteStep.DATA, RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL]}
        def add(step):
            for requirement in prerequisites.get(step, []):
                add(requirement)
            if step not in result:
                result.append(step)
        for step in steps:
            add(step)
        return result

    def _tasks(self, intent: QueryIntent, steps: list[RouteStep]) -> list[RoutingTask]:
        tasks: list[RoutingTask] = []
        task_by_step: dict[RouteStep, str] = {}
        for step in steps:
            required, expected, timeout, budget = self._TASK_DEFAULTS[step]
            task_id = step.value.lower()
            dependencies: list[str] = []
            task_inputs = list(required)
            if step == RouteStep.DATA and intent == QueryIntent.MARKET_OVERVIEW:
                task_inputs = ["market_index"]
            elif step == RouteStep.DATA and intent in {QueryIntent.STOCK_DISCOVERY, QueryIntent.STOCK_RECOMMENDATION}:
                task_inputs = ["candidate_tickers"]
                dependencies = [task_by_step[RouteStep.SCREENING]]
            elif step == RouteStep.DATA and intent == QueryIntent.STOCK_COMPARISON:
                task_inputs = ["tickers"]
                expected = ["market_data", "market_data_collection"]
            elif step == RouteStep.TECHNICAL:
                dependencies = [task_by_step[RouteStep.DATA]]
            elif step == RouteStep.FORECAST:
                dependencies = [task_by_step[RouteStep.DATA]]
            elif step == RouteStep.FUNDAMENTAL:
                dependencies = [task_by_step[RouteStep.DATA]]
            elif step == RouteStep.RISK:
                dependencies = [task_by_step[item] for item in (RouteStep.DATA, RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL)]
            elif step == RouteStep.MANAGER:
                dependencies = [task_by_step[item] for item in (RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL, RouteStep.RISK)]
            elif step == RouteStep.CHART:
                dependencies = [task_by_step[RouteStep.DATA]]
            elif step == RouteStep.MARKET:
                dependencies = [task_by_step[RouteStep.DATA]]
            elif step == RouteStep.COMPARISON:
                dependencies = [task_by_step[RouteStep.DATA], task_by_step[RouteStep.MANAGER]]
            elif step == RouteStep.SCREENING:
                task_inputs = ["universe"]
            elif step == RouteStep.RECOMMENDATION:
                dependencies = [task_by_step[RouteStep.SCREENING]]
            elif step == RouteStep.WORKBENCH and intent == QueryIntent.THESIS_REVIEW:
                task_inputs = ["user_message", "evidence_bundle"]
                dependencies = [task_by_step[RouteStep.RESEARCH]]
            task = RoutingTask(
                id=task_id,
                step=step,
                depends_on=dependencies,
                required_inputs=task_inputs,
                expected_outputs=list(expected),
                timeout_seconds=timeout,
                budget_units=budget,
                execution_contract=self._CONTRACTS[step],
                nested_operations=list(self._OPERATIONS[step]),
            )
            tasks.append(task)
            task_by_step[step] = task_id
        return tasks
