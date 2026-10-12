from __future__ import annotations

import os

from stock_multi_agent.agents.base_agent import BaseAgent
from stock_multi_agent.models.agent_contracts import AgentSpec, DataAgentContext
from stock_multi_agent.models.messages import AgentName, MessageType
from stock_multi_agent.models.state import AgentState, TraceEvent
from stock_multi_agent.tools.market_data import MarketDataTool


class DataAgent(BaseAgent):
    name = AgentName.DATA
    goal = "Acquire, validate, and normalize real stock data without making a recommendation."
    spec = AgentSpec(
        role=name,
        goal=goal,
        required_context=["ticker", "investment_horizon", "required_data"],
        allowed_tools=["get_stock_data", "fetch_benchmark"],
        output_type="MarketData",
        can_use_llm=False,
        decision_constraints=[
            "Never fabricate market or financial data",
            "Every accepted dataset must preserve source and timestamp metadata",
            "Never issue an investment recommendation",
        ],
    )

    def __init__(self, market_tool: MarketDataTool, llm_advisor=None) -> None:
        self.market_tool = market_tool
        self.llm_advisor = llm_advisor

    def run(self, state: AgentState) -> dict[str, object]:
        context = DataAgentContext.from_state(state)
        observation = self.event("OBSERVE", "READ_REQUEST", f"Ticker={context.ticker}; horizon={context.investment_horizon}")
        include_financials = "financials" in context.required_data
        requested = "OHLCV and financial metrics" if include_financials else "OHLCV only"
        call = self.event("ACT", "CALL_TOOL", f"Request real {requested}", "get_stock_data")
        data = self.market_tool.get_stock_data(context.ticker, include_financials=include_financials)
        benchmark_data = None
        benchmark_trace: list[TraceEvent] = []
        if (
            "benchmark" in context.required_data
            and context.ticker != "VNINDEX"
            and os.getenv("STOCK_AGENT_BENCHMARK_ENABLED", "true").lower() not in {"0", "false", "off"}
        ):
            try:
                benchmark_data = self.market_tool.get_stock_data("VNINDEX", include_financials=False)
                benchmark_trace.append(self.event(
                    "ACT", "FETCH_BENCHMARK", "Request VNINDEX benchmark series",
                ))
            except Exception as exc:
                # A benchmark is complementary analysis context. Its failure
                # must not make a verified issuer-data request fail.
                benchmark_trace.append(self.event(
                    "OBSERVE", "BENCHMARK_UNAVAILABLE", f"VNINDEX benchmark unavailable: {type(exc).__name__}",
                ))
        result = self.message(
            AgentName.MANAGER,
            MessageType.RESULT,
            "Real market data normalized",
            {
                "ticker": data.ticker,
                "source": data.source,
                "bars": len(data.bars),
                "financial_metrics": data.financials.available_count(),
                "as_of": data.as_of.isoformat() if data.as_of else None,
                "fetched_at": data.fetched_at.isoformat(),
                "source_ids": [source.source_id for source in data.source_refs],
                "quality_status": data.quality_report.status.value,
                "warnings": data.warnings,
            },
            confidence=(
                min(1.0, len(data.bars) / 200)
                * (0.7 + 0.3 * data.financials.available_count() / 10)
                * (0.8 if data.quality_report.status.value == "WARN" else 1.0)
            ),
        )
        source_ids = [source.source_id for source in data.source_refs]
        llm_review = self.llm_review(
            "Audit data sufficiency, provenance, and visible limitations",
            {
                "ticker": data.ticker,
                "source": data.source,
                "sessions": len(data.bars),
                "financial_metric_count": data.financials.available_count(),
                "as_of": data.as_of.isoformat() if data.as_of else None,
                "quality_status": data.quality_report.status.value,
                "quality_issues": [issue.message for issue in data.quality_report.issues],
                "warnings": data.warnings,
            },
            source_ids,
        )
        if llm_review is not None:
            benchmark_trace.append(self.event(
                "REASON", "LLM_DATA_AUDIT", f"Structured LLM review={llm_review.status}", "local_llm",
            ))
        return {
            "market_data": data,
            "benchmark_data": benchmark_data,
            "messages": [result],
            "llm_reviews": [llm_review] if llm_review is not None else [],
            "trace": [
                observation,
                call,
                self.event(
                    "OBSERVE",
                    "VALIDATE_RESULT",
                    f"Received {len(data.bars)} valid sessions from {data.source}; quality={data.quality_report.status.value}",
                ),
                *benchmark_trace,
            ],
            "completed_tasks": ["data_acquisition"],
            "agent_reports": [self.report(
                stage="data_acquisition",
                status="PARTIAL" if data.quality_report.issues else "COMPLETED",
                summary=f"Loaded {len(data.bars)} sessions from {data.source}",
                confidence=result.confidence or 0.0,
                output_key="market_data",
                tools_used=["get_stock_data", *(["fetch_benchmark"] if benchmark_data is not None else [])],
                assumptions=[f"Provider price adjustment={data.price_adjustment.value}"],
                missing_information=[issue.message for issue in data.quality_report.issues],
                evidence_count=len(data.source_refs),
            )],
        }
