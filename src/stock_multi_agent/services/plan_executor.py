"""Execute approved route capabilities and retain evidence of actual completion.

Full LangGraph analyses and bounded batch services are atomic composites: their
internal retry/fan-out graph remains owned by that service. A composite must
claim all its approved tasks up front and return their real typed artifacts.
Selective analyses execute one task at a time and reuse the preceding state.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

from pydantic import BaseModel, TypeAdapter

from stock_multi_agent.chat.router import RoutingPlan, RoutingTask
from stock_multi_agent.models.analysis import MarketData, TechnicalAnalysis, FundamentalAnalysis, RiskAnalysis, FinalDecision
from stock_multi_agent.models.chat import AssistantResponse
from stock_multi_agent.models.chart import ChartArtifact
from stock_multi_agent.models.forecast import PriceForecast
from stock_multi_agent.models.research import EvidenceBundle
from stock_multi_agent.models.screening import DualRecommendation
from stock_multi_agent.services.request_telemetry import BudgetExceeded, check_deadline, task_deadline_scope


class TaskOutputError(ValueError):
    pass


class PlanExecutor:
    output_schemas = {
        "market_data": MarketData | dict[str, MarketData] | list[MarketData],
        "market_data_collection": list[MarketData],
        "technical_result": TechnicalAnalysis | list[TechnicalAnalysis],
        "fundamental_result": FundamentalAnalysis | list[FundamentalAnalysis],
        "risk_result": RiskAnalysis | list[RiskAnalysis],
        "final_decision": FinalDecision | list[FinalDecision],
        "candidate_tickers": list[str], "dual_recommendation": DualRecommendation,
        "chart_artifact": list[ChartArtifact], "price_forecast": PriceForecast,
        "evidence_bundle": EvidenceBundle,
        **dict.fromkeys(("comparison", "knowledge_evidence", "chat_response", "market_overview",
            "research_tool_result"), AssistantResponse),
    }

    def __init__(self, plan: RoutingPlan):
        self.plan = plan
        self.outputs: dict[str, Any] = {}
        self.records: dict[str, dict[str, Any]] = {}

    def run(self, task: RoutingTask, action: Callable[[], Mapping[str, Any]]) -> Mapping[str, Any]:
        return self.run_composite([task], action, contract=task.execution_contract)

    def run_composite(
        self, tasks: list[RoutingTask], action: Callable[[], Mapping[str, Any]], *, contract: str,
    ) -> Mapping[str, Any]:
        """Call only after the supplied capabilities have passed plan validation.

        No expected output is assumed from a route name or an invocation. The
        action returns actual artifacts, and absent artifacts remain unavailable.
        """
        approved = {item.id: item for item in self.plan.tasks}
        claimed = {item.id for item in tasks}
        for task in tasks:
            if approved.get(task.id) != task or task.id in self.records:
                raise TaskOutputError(f"Task {task.id} is unapproved or was already executed.")
            for dependency in set(task.depends_on).difference(claimed):
                if self.records.get(dependency, {}).get("status") not in {"COMPLETED", "PARTIAL"}:
                    raise TaskOutputError(f"Task {task.id} has an incomplete dependency: {dependency}.")
        started = perf_counter()
        entered_action = False
        deadline_exceeded = False
        for task in tasks:
            self.records[task.id] = self._record(task, "RUNNING", contract)
        try:
            with task_deadline_scope(sum(task.timeout_seconds for task in tasks)):
                check_deadline()
                entered_action = True
                result = dict(action())
                try:
                    check_deadline()
                except BudgetExceeded:
                    # Graph finalization deliberately retains supported partial
                    # work after a cooperative deadline. Do not discard it here.
                    deadline_exceeded = True
            for task in tasks:
                missing = [key for key in task.expected_outputs if key not in result or result[key] is None
                    or (key in {"market_data", "market_data_collection", "technical_result", "fundamental_result",
                        "risk_result", "final_decision"} and isinstance(result[key], (list, dict)) and not result[key])]
                produced = [key for key in task.expected_outputs if key not in missing]
                record = self.records[task.id]
                reported = result.get("_task_statuses", {}).get(task.step.value, "COMPLETED")
                status = "UNAVAILABLE" if missing else reported
                if deadline_exceeded and status == "COMPLETED":
                    status = "PARTIAL"
                if "chart_artifact" in produced and not result["chart_artifact"] and status == "COMPLETED":
                    status = "PARTIAL"
                record.update(status=status,
                    produced_outputs=produced, missing_outputs=missing,
                    latency_ms=round((perf_counter() - started) * 1000))
                if deadline_exceeded:
                    record["deadline_exceeded"] = True
                for key in produced:
                    schema = self.output_schemas.get(key)
                    if schema is None:
                        raise TaskOutputError(f"No registered output schema for {key}.")
                    TypeAdapter(schema).validate_python(self._validation_value(result[key]))
                    self.outputs[key] = result[key]
                    record.setdefault("output_types", {})[key] = self._type_name(result[key])
            return result
        except Exception as exc:
            for task in tasks:
                self.records[task.id].update(status="FAILED" if entered_action else "SKIPPED", error_type=type(exc).__name__,
                    latency_ms=round((perf_counter() - started) * 1000))
            raise

    def snapshot(self) -> list[dict[str, Any]]:
        return [self.records.get(task.id, self._record(task, "SKIPPED", task.execution_contract))
            for task in self.plan.tasks]

    @staticmethod
    def _validation_value(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, list):
            return [PlanExecutor._validation_value(item) for item in value]
        if isinstance(value, dict):
            return {key: PlanExecutor._validation_value(item) for key, item in value.items()}
        return value

    @staticmethod
    def _record(task: RoutingTask, status: str, contract: str) -> dict[str, Any]:
        return {"task_id": task.id, "step": task.step.value, "status": status,
            "execution_contract": task.execution_contract, "composite_contract": contract,
            "nested_operations": task.nested_operations, "depends_on": task.depends_on,
            "expected_outputs": task.expected_outputs, "produced_outputs": [],
            "missing_outputs": list(task.expected_outputs), "timeout_seconds": task.timeout_seconds,
            "budget_units": task.budget_units}

    @staticmethod
    def _type_name(value: Any) -> str:
        if isinstance(value, BaseModel):
            # Construction already validated the schema; validation here also
            # catches invalid values injected through model_copy/update.
            type(value).model_validate(value.model_dump())
        return type(value).__name__
