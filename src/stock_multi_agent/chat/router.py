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
    COMPARISON = "COMPARISON"
    RESEARCH = "RESEARCH"
    KNOWLEDGE = "KNOWLEDGE"
    CHAT = "CHAT"


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


class RoutingPlan(BaseModel):
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


class RoutingPlanValidator:
    """Deterministic safety boundary between semantic parsing and execution."""

    allowed_steps = frozenset(RouteStep)
    max_total_budget_units = 30

    @classmethod
    def validate(cls, plan: RoutingPlan, query: ParsedQuery) -> list[str]:
        errors: list[str] = []
        task_ids = [task.id for task in plan.tasks]
        if len(set(task_ids)) != len(task_ids):
            errors.append("Task IDs must be unique.")
        if plan.total_budget_units > cls.max_total_budget_units:
            errors.append("Task plan exceeds the bounded execution budget.")

        available_inputs = {"user_message"}
        if query.tickers:
            available_inputs.add("ticker")
        if len(query.tickers) >= 2:
            available_inputs.add("tickers")
        if plan.intent == QueryIntent.MARKET_OVERVIEW:
            available_inputs.add("market_index")
        if plan.intent == QueryIntent.STOCK_DISCOVERY:
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
    def assert_valid(cls, plan: RoutingPlan, query: ParsedQuery) -> None:
        errors = cls.validate(plan, query)
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
        RouteStep.COMPARISON: (["market_data_collection"], ["comparison"], 30, 3),
        RouteStep.RESEARCH: (["research_query"], ["evidence_bundle"], 20, 2),
        RouteStep.KNOWLEDGE: (["user_message"], ["knowledge_evidence"], 10, 1),
        RouteStep.CHAT: (["user_message"], ["chat_response"], 5, 0),
    }

    def plan(self, query: ParsedQuery, context: ConversationState | None = None) -> RoutingPlan:
        intent = query.intent
        if query.needs_clarification:
            return self._validated_plan(intent, [RouteStep.CHAT], query, "The semantic parser requires clarification before running agents.")
        single_ticker_intents = {
            QueryIntent.STOCK_ANALYSIS, QueryIntent.TECHNICAL_ANALYSIS,
            QueryIntent.FUNDAMENTAL_ANALYSIS, QueryIntent.RISK_ANALYSIS,
            QueryIntent.STOCK_CHART, QueryIntent.INDICATOR_QUERY, QueryIntent.MIXED,
        }
        if (intent in single_ticker_intents and not query.tickers) or (
            intent is QueryIntent.STOCK_COMPARISON and len(query.tickers) < 2
        ):
            return self._validated_plan(
                intent,
                [RouteStep.CHAT],
                query,
                "Required ticker entities are missing; ask for clarification before running agents.",
            )
        if intent == QueryIntent.FOLLOW_UP:
            intent = context.last_intent if context and context.last_intent else QueryIntent.STOCK_ANALYSIS
        routes = {
            QueryIntent.STOCK_ANALYSIS: [RouteStep.DATA, RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL, RouteStep.RISK, RouteStep.MANAGER],
            QueryIntent.TECHNICAL_ANALYSIS: [RouteStep.DATA, RouteStep.TECHNICAL],
            QueryIntent.FUNDAMENTAL_ANALYSIS: [RouteStep.DATA, RouteStep.FUNDAMENTAL],
            QueryIntent.RISK_ANALYSIS: [RouteStep.DATA, RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL, RouteStep.RISK],
            QueryIntent.MARKET_OVERVIEW: [RouteStep.DATA, RouteStep.MARKET],
            QueryIntent.STOCK_DISCOVERY: [RouteStep.SCREENING, RouteStep.DATA, RouteStep.TECHNICAL],
            QueryIntent.STOCK_COMPARISON: [RouteStep.DATA, RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL, RouteStep.RISK, RouteStep.MANAGER, RouteStep.COMPARISON],
            QueryIntent.STOCK_CHART: [RouteStep.DATA, RouteStep.CHART],
            QueryIntent.INDICATOR_QUERY: [RouteStep.DATA, RouteStep.TECHNICAL],
            QueryIntent.RESEARCH: [RouteStep.RESEARCH],
            QueryIntent.THEORY: [RouteStep.KNOWLEDGE],
            QueryIntent.CURRENT_RULE: [RouteStep.KNOWLEDGE],
            QueryIntent.MIXED: [RouteStep.KNOWLEDGE, RouteStep.DATA, RouteStep.FUNDAMENTAL],
            QueryIntent.EXPLANATION: [RouteStep.CHAT],
            QueryIntent.GENERAL_CHAT: [RouteStep.CHAT],
            QueryIntent.HELP: [RouteStep.CHAT],
            QueryIntent.UNKNOWN: [RouteStep.CHAT],
        }
        steps = list(routes.get(intent, [RouteStep.CHAT]))
        fundamental_indicators = {"PE", "PB", "ROE", "ROA", "EPS"}
        if intent == QueryIntent.INDICATOR_QUERY and fundamental_indicators.intersection(query.requested_indicators):
            steps = [RouteStep.DATA, RouteStep.FUNDAMENTAL]
        if intent == QueryIntent.MIXED and query.requested_indicators and not fundamental_indicators.intersection(query.requested_indicators):
            steps = [RouteStep.KNOWLEDGE, RouteStep.DATA, RouteStep.TECHNICAL]
        secondary_routes = {
            QueryIntent.STOCK_CHART: [RouteStep.CHART],
            QueryIntent.INDICATOR_QUERY: [RouteStep.TECHNICAL],
            QueryIntent.RISK_ANALYSIS: [RouteStep.RISK],
            QueryIntent.FUNDAMENTAL_ANALYSIS: [RouteStep.FUNDAMENTAL],
        }
        for secondary in query.secondary_intents:
            for step in secondary_routes.get(secondary, []):
                if step not in steps:
                    steps.append(step)
        steps = self._with_required_steps(steps)
        return self._validated_plan(intent, steps, query, f"{intent.value} needs {' → '.join(step.value for step in steps)}")

    @staticmethod
    def _with_required_steps(steps: list[RouteStep]) -> list[RouteStep]:
        """Expand a selected specialist only when its deterministic inputs need it."""
        result = list(steps)
        if RouteStep.RISK not in result:
            return result
        risk_index = result.index(RouteStep.RISK)
        for requirement in (RouteStep.DATA, RouteStep.TECHNICAL, RouteStep.FUNDAMENTAL):
            if requirement not in result:
                result.insert(risk_index, requirement)
                risk_index += 1
        return result

    def _validated_plan(self, intent: QueryIntent, steps: list[RouteStep], query: ParsedQuery, reason: str) -> RoutingPlan:
        plan = RoutingPlan(intent=intent, tasks=self._tasks(intent, steps), reason=reason, confidence=query.confidence)
        RoutingPlanValidator.assert_valid(plan, query)
        return plan

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
            elif step == RouteStep.DATA and intent == QueryIntent.STOCK_DISCOVERY:
                task_inputs = ["candidate_tickers"]
                dependencies = [task_by_step[RouteStep.SCREENING]]
            elif step == RouteStep.DATA and intent == QueryIntent.STOCK_COMPARISON:
                task_inputs = ["tickers"]
                expected = ["market_data", "market_data_collection"]
            elif step == RouteStep.TECHNICAL:
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
                dependencies = [task_by_step[RouteStep.DATA]]
            elif step == RouteStep.SCREENING:
                task_inputs = ["universe"]
            task = RoutingTask(
                id=task_id,
                step=step,
                depends_on=dependencies,
                required_inputs=task_inputs,
                expected_outputs=list(expected),
                timeout_seconds=timeout,
                budget_units=budget,
            )
            tasks.append(task)
            task_by_step[step] = task_id
        return tasks
