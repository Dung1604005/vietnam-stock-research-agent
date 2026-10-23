"""Offline synthetic contract ablations, isolated from product decision policy.

These checks compare bounded orchestration cost and constraint preservation on
frozen inputs. They are not a return backtest, confidence calibration, or an
independently labelled test of investment quality.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from langgraph.checkpoint.memory import InMemorySaver

from stock_multi_agent.agents import DataAgent, FundamentalAgent, ManagerAgent, ResearchAgent, RiskAgent, TechnicalAgent
from stock_multi_agent.graph.stock_agent_graph import StockAgentGraph
from stock_multi_agent.models.analysis import FinancialMetrics, MarketData, PriceBar, Recommendation, Signal
from stock_multi_agent.models.data import (
    DataKind, DataQualityReport, FinancialMetricObservation, Instrument, IssuerEvent, PriceAdjustment, SourceReference,
)
from stock_multi_agent.models.research import EvidenceBundle
from stock_multi_agent.models.research_contracts import RequestBudget
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.providers.llm_provider import MockLLMProvider
from stock_multi_agent.services.execution_profiles import execution_scope
from stock_multi_agent.services.price_forecast import PriceForecastService
from stock_multi_agent.services.request_telemetry import check_deadline, observed, request_scope
from stock_multi_agent.tools import FinancialToolkit, MarketDataTool, TechnicalIndicatorToolkit


class CoordinationCase(BaseModel):
    model_config = ConfigDict(extra='forbid')
    case_id: str = Field(min_length=1)
    market_data: MarketData
    investment_horizon: str = '3 months'
    forbidden_recommendations: list[Recommendation] = Field(default_factory=list)
    constraint_reason: str = ''
    label_provenance: Literal['SYNTHETIC_CONTRACT', 'HUMAN_REVIEWED_SNAPSHOT_CONSTRAINT'] = 'SYNTHETIC_CONTRACT'


def snapshot_sha256(data: MarketData) -> str:
    packed = json.dumps(data.model_dump(mode='json'), sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(packed.encode('utf-8')).hexdigest()


class _FrozenProvider:
    def __init__(self, data):
        self.data = data.model_copy(deep=True)

    def get_market_data(self, ticker, include_financials=True):
        check_deadline()
        if ticker != self.data.ticker:
            raise ValueError('Benchmark never acquires additional instruments')
        return self.data.model_copy(deep=True)

    refresh_market_data = get_market_data


class _EmptyResearchStore:
    @observed('benchmark.research_retrieval', 'tool')
    def search(self, query, **kwargs):
        return EvidenceBundle(query=query, retrieval_note='Frozen empty synthetic research corpus')


class CoordinationBenchmarkService:
    VARIANTS = ('single_pass', 'bounded_coordination', 'no_risk_baseline')

    def _graph(self, data: MarketData) -> StockAgentGraph:
        tools = TechnicalIndicatorToolkit()
        return StockAgentGraph(
            data_agent=DataAgent(MarketDataTool(_FrozenProvider(data))),
            research_agent=ResearchAgent(store=_EmptyResearchStore()),
            technical_agent=TechnicalAgent(tools),
            fundamental_agent=FundamentalAgent(FinancialToolkit()),
            risk_agent=RiskAgent(tools),
            manager_agent=ManagerAgent(llm_provider=MockLLMProvider(), routing_enabled=False, synthesis_enabled=False),
            price_forecast_service=PriceForecastService(),
            checkpointer=InMemorySaver(),
        )

    @staticmethod
    def _merge(state, update):
        from stock_multi_agent.services.analysis_service import AnalysisService
        return AnalysisService._merge(state, update)

    def _no_risk_baseline(self, graph, state):
        """Diagnostic ablation only. Never passed to a production Manager.

        Combine the two specialist signals without a critic, conflict gate or
        confidence policy. Missing Risk coverage is reported explicitly.
        """
        state = self._merge(state, graph.data_agent.run(state))
        state = self._merge(state, graph._forecast(state))
        state = self._merge(state, graph.research_agent.run(state))
        state = self._merge(state, graph.technical_agent.run(state, allow_llm_planning=False))
        state = self._merge(state, graph.fundamental_agent.run(state, allow_llm_review=False))
        check_deadline()
        signals = (state.technical_result.signal, state.fundamental_result.signal)
        recommendation = (
            Recommendation.BUY if signals == (Signal.BUY, Signal.BUY) else
            Recommendation.HOLD if signals == (Signal.HOLD, Signal.HOLD) else
            Recommendation.AVOID if Signal.SELL in signals else Recommendation.WATCH
        )
        return state, recommendation

    @staticmethod
    def _measure(case, variant, state, recommendation, metrics, latency_ms, error_type, allowed_iterations):
        signals = [getattr(state, name).signal for name in ('technical_result', 'fundamental_result')
                   if getattr(state, name) is not None]
        conflicts = ManagerAgent.assess_conflicts(state) if state.risk_result else None
        unresolved = [item for item in conflicts if not item.resolved] if conflicts is not None else None
        violations = []
        if recommendation in case.forbidden_recommendations:
            violations.append({'code': 'CASE_RECOMMENDATION_CONSTRAINT', 'detail': case.constraint_reason})
        if recommendation is Recommendation.BUY and Signal.UNKNOWN in signals:
            violations.append({'code': 'BUY_WITH_UNKNOWN_SPECIALIST', 'detail': 'A specialist abstained'})
        if recommendation is Recommendation.BUY and state.risk_result and state.risk_result.veto:
            violations.append({'code': 'BUY_WITH_RISK_VETO', 'detail': 'Critic veto was not preserved'})
        if recommendation is Recommendation.BUY and unresolved and any(item.blocks_buy for item in unresolved):
            violations.append({'code': 'BUY_WITH_BLOCKING_CONFLICT', 'detail': 'Material conflict remains unresolved'})
        if state.iteration > allowed_iterations:
            violations.append({'code': 'ITERATION_BOUND_EXCEEDED', 'detail': str(allowed_iterations)})
        if error_type or recommendation is None:
            violations.append({'code': 'RUN_INCOMPLETE', 'detail': error_type or 'No recommendation'})
        return {
            'case_id': case.case_id, 'variant': variant,
            'snapshot_sha256': snapshot_sha256(case.market_data),
            'label_provenance': case.label_provenance,
            'recommendation': recommendation.value if recommendation is not None else None,
            'technical_signal': state.technical_result.signal.value if state.technical_result else None,
            'fundamental_signal': state.fundamental_result.signal.value if state.fundamental_result else None,
            'unknown_specialist_signals': sum(signal is Signal.UNKNOWN for signal in signals),
            'missing_specialist_results': [name for name in ('technical_result', 'fundamental_result', 'risk_result')
                                           if getattr(state, name) is None],
            'risk_assessed': state.risk_result is not None,
            'risk_veto': state.risk_result.veto if state.risk_result is not None else None,
            'unresolved_conflicts': len(unresolved) if unresolved is not None else None,
            'unresolved_blocking_conflicts': sum(item.blocks_buy for item in unresolved) if unresolved is not None else None,
            'conflict_measurement': 'production typed conflict detector' if unresolved is not None else 'not assessed: critic omitted',
            'iterations': state.iteration, 'allowed_iterations': allowed_iterations,
            'latency_ms': round(latency_ms, 3),
            'tool_calls': metrics.tool_calls, 'llm_calls': metrics.llm_calls,
            'api_calls': metrics.api_calls, 'retries': metrics.retries,
            'budget': metrics.budget.model_dump(mode='json'),
            'constraint_violations': violations,
            'error_type': error_type,
        }

    def run(self, cases: list[CoordinationCase], *, budget: RequestBudget | None = None, max_iterations: int = 2) -> dict:
        if not cases or len({case.case_id for case in cases}) != len(cases):
            raise ValueError('Benchmark requires at least one case and unique case IDs')
        if not 0 <= max_iterations <= 10:
            raise ValueError('max_iterations must be between 0 and 10')
        shared_budget = (budget or RequestBudget(max_llm_calls=0, max_tool_calls=250)).model_copy(deep=True)
        if shared_budget.max_llm_calls != 0:
            raise ValueError('Offline coordination benchmark requires max_llm_calls=0')
        rows = []
        # Direct graph invocation preserves its declared iteration limit while
        # OFFLINE independently guards all registered outbound providers.
        with execution_scope('OFFLINE'):
            for source_case in cases:
                case = source_case.model_copy(deep=True)
                original_hash = snapshot_sha256(case.market_data)
                for variant in self.VARIANTS:
                    graph = self._graph(case.market_data)
                    allowed = max_iterations if variant == 'bounded_coordination' else 0
                    state = AgentState(
                        ticker=case.market_data.ticker, investment_horizon=case.investment_horizon,
                        user_goal='Synthetic frozen snapshot coordination contract benchmark',
                        required_data=['ohlcv', 'financials'], max_iterations=allowed,
                        allow_llm_planning=False, request_budget=shared_budget.model_copy(deep=True),
                        execution_mode='OFFLINE',
                    )
                    recommendation, error_type = None, None
                    with request_scope(shared_budget.model_copy(deep=True)) as metrics:
                        started = time.perf_counter()
                        try:
                            if variant == 'no_risk_baseline':
                                state, recommendation = self._no_risk_baseline(graph, state)
                            else:
                                state = graph.invoke(state)
                                recommendation = state.final_decision.recommendation if state.final_decision else None
                        except Exception as exc:
                            error_type = type(exc).__name__
                        latency = (time.perf_counter() - started) * 1000
                        rows.append(self._measure(case, variant, state, recommendation, metrics, latency, error_type, allowed))
                    if snapshot_sha256(case.market_data) != original_hash:
                        raise RuntimeError('Frozen benchmark snapshot was mutated')
        summary = {}
        for variant in self.VARIANTS:
            runs = [row for row in rows if row['variant'] == variant]
            summary[variant] = {
                'runs': len(runs),
                'constraint_violation_count': sum(len(row['constraint_violations']) for row in runs),
                'unknown_specialist_signals': sum(row['unknown_specialist_signals'] for row in runs),
                'unresolved_conflicts': sum(row['unresolved_conflicts'] for row in runs) if all(row['unresolved_conflicts'] is not None for row in runs) else None,
                'tool_calls': sum(row['tool_calls'] for row in runs),
                'llm_calls': sum(row['llm_calls'] for row in runs),
                'latency_ms': round(sum(row['latency_ms'] for row in runs), 3),
                'failed_runs': sum(row['error_type'] is not None for row in runs),
            }
        return {
            'schema_version': 'coordination-contract-benchmark-v1',
            'dataset_version': 'synthetic-frozen-coordination-v1',
            'scope': 'Synthetic contract benchmark; same frozen market input, empty research corpus, deterministic forecast baseline and shared budget for each variant. Single pass remains a multi-role graph with max_iterations=0; it is not a single LLM agent.',
            'limitations': [
                'Contract labels are author-created unless a case explicitly declares human review; no independent investment outcome labels.',
                'No return, profitability, calibration or general superiority claim is supported.',
                'No-risk aggregation is evaluator-only and has no production decision policy; its unresolved conflict count is unassessed.',
                'Latency is a local observation sensitive to machine load and graph scheduling; tool calls count instrumented boundaries, not every Python calculation.',
            ],
            'budget': shared_budget.model_dump(mode='json'), 'rows': rows, 'summary': summary,
        }


def synthetic_coordination_cases() -> list[CoordinationCase]:
    """Versioned, fixed-date author-created fixtures; never a product provider."""
    as_of = date(2026, 10, 1)
    dates, day = [], as_of
    while len(dates) < 260:
        if day.weekday() < 5:
            dates.append(day)
        day -= timedelta(days=1)
    bars = []
    for index, day in enumerate(reversed(dates)):
        close = 80_000 + 180 * index + 1300 * math.sin(index * .6) + max(0, index - 230) ** 2 * 20
        bars.append(PriceBar(date=day, open=close - 200, high=close + 800, low=close - 800, close=close,
                             volume=2_000_000 if index >= 255 else 1_000_000))
    fetched = datetime(2026, 10, 1, 18, tzinfo=timezone.utc)
    market = MarketData(
        ticker='SYN', source='synthetic-contract-fixture-v1', bars=bars, fetched_at=fetched,
        access_mode='FROZEN_SYNTHETIC', price_unit='VND', price_adjustment=PriceAdjustment.ADJUSTED,
        quality_report=DataQualityReport(checked_at=fetched),
        instrument=Instrument(ticker='SYN', sector='Technology'),
        financials=FinancialMetrics(revenue_growth=.2, profit_growth=.25, eps=5, pe=12, pb=2, roe=.22, roa=.12,
                                    debt_to_equity=.3, operating_cash_flow=1_000_000, profit_margin=.18),
        source_refs=[SourceReference(provider='synthetic', source_id='synthetic:ohlcv', data_kind=DataKind.OHLCV, fetched_at=fetched, as_of=as_of),
                     SourceReference(provider='synthetic', source_id='synthetic:ratios', data_kind=DataKind.FINANCIAL_RATIOS, fetched_at=fetched, as_of=as_of)],
    )
    missing = market.model_copy(deep=True)
    missing.financials = FinancialMetrics()
    conflicting = market.model_copy(deep=True)
    conflicting.financial_observations = [FinancialMetricObservation(
        metric='pe', value=value, unit='multiple', period_label='2026-Q3', period_end=date(2026, 9, 30), source_id=f'synthetic:ratios:{index}',
    ) for index, value in enumerate((12, 22))]
    suspended = market.model_copy(deep=True)
    suspended.source_refs.append(SourceReference(provider='synthetic', source_id='synthetic:events', data_kind=DataKind.ISSUER_EVENTS, fetched_at=fetched, as_of=as_of))
    suspended.issuer_events = [IssuerEvent(event_code='SUSP', title='Trading suspended by exchange', event_date=as_of, source_id='synthetic:events')]
    return [
        CoordinationCase(case_id='complete_inputs', market_data=market),
        CoordinationCase(case_id='missing_fundamentals', market_data=missing, forbidden_recommendations=[Recommendation.BUY], constraint_reason='Missing fundamentals cannot support BUY'),
        CoordinationCase(case_id='metric_conflict', market_data=conflicting, forbidden_recommendations=[Recommendation.BUY], constraint_reason='Two comparable observations report different P/E values'),
        CoordinationCase(case_id='verified_suspension', market_data=suspended, forbidden_recommendations=[Recommendation.BUY], constraint_reason='Synthetic verified current trading suspension forbids BUY'),
    ]
