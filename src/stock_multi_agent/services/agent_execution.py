"""Stage authority, input lineage and conservative specialist failure results."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import uuid4

from stock_multi_agent.models.analysis import FundamentalAnalysis, RiskAnalysis, RiskLevel, RiskSection, Signal, TechnicalAnalysis
from stock_multi_agent.models.messages import AgentName, MessageType
from stock_multi_agent.models.state import AgentState, TaskExecution


@dataclass(frozen=True)
class ExecutionContext:
    run_id: str
    task_id: str
    input_fingerprint: str
    reply_to: object = None


_context = ContextVar('agent_execution_context', default=None)


def current_execution():
    return _context.get()


@contextmanager
def execution_for_state(state):
    """Restore a saved policy without widening a caller's stricter limits."""
    from stock_multi_agent.services.execution_profiles import (
        constrain_budget, current_profile, execution_scope, resolve_profile,
    )
    from stock_multi_agent.services.request_telemetry import current_metrics, request_scope
    saved = state.execution_profile or resolve_profile(state.execution_mode)
    caller = current_profile()
    changes = {'external_allowed': saved.external_allowed and caller.external_allowed}
    for key in ('max_llm_calls', 'max_tool_calls', 'deadline_seconds'):
        limits = [value for value in (getattr(saved, key), getattr(caller, key)) if value is not None]
        changes[key] = min(limits) if limits else None
    profile = saved.model_copy(update=changes)
    budget = constrain_budget(state.request_budget, profile)
    with execution_scope(profile):
        metrics = current_metrics()
        if metrics is None:
            with request_scope(budget):
                yield
        else:
            # Budget narrowing is monotonic for a shared request. Restoring a
            # wider budget while a sibling is still running would be unsafe.
            with metrics._lock:
                metrics.budget = metrics.budget.model_copy(update={
                    key: min(getattr(metrics.budget, key), getattr(budget, key))
                    for key in ('max_llm_calls', 'max_tool_calls', 'max_output_tokens', 'deadline_seconds')
                })
            yield


def input_fingerprint(state):
    def stable(value):
        if isinstance(value, dict):
            return {key: stable(item) for key, item in value.items()
                    if key not in {'fetched_at', 'checked_at', 'retrieved_at', 'timestamp'}}
        return [stable(item) for item in value] if isinstance(value, list) else value
    payload = {key: getattr(state, key) for key in ('ticker', 'investment_horizon', 'user_goal', 'required_data',
        'requested_indicators', 'market_data', 'benchmark_data', 'price_forecast',
        'research_evidence', 'technical_result', 'fundamental_result', 'risk_result', 'portfolio_positions',
        'execution_profile', 'request_budget', 'allow_llm_planning')}
    encoded = json.dumps(stable(json.loads(json.dumps(payload, default=lambda item: item.model_dump(mode='json')))), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


@contextmanager
def execution_context(state, role, node):
    completed = {item.task_id for item in state.task_executions}
    task = next((item for item in reversed(state.messages)
                 if item.receiver == role and item.message_type in {MessageType.TASK, MessageType.REQUEST_REANALYSIS}
                 and item.run_id in {None, state.run_id} and item.task_id not in completed), None)
    context = ExecutionContext(state.run_id, task.task_id if task and task.task_id else str(uuid4()),
                               input_fingerprint(state), task.id if task else None)
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


COMMON = {'messages', 'trace', 'completed_tasks', 'agent_reports', 'llm_reviews'}
OUTPUTS = {
    AgentName.DATA: {'market_data', 'benchmark_data'},
    AgentName.RESEARCH: {'research_evidence', 'research_freshness_note'},
    AgentName.TECHNICAL: {'technical_result'},
    AgentName.FUNDAMENTAL: {'fundamental_result', 'financial_fact_proposals'},
    AgentName.RISK: {'risk_result'},
    AgentName.MANAGER: {'current_plan', 'conflict_detected', 'reanalysis_requested', 'active_conflicts',
        'reanalysis_action', 'reanalysis_history', 'reanalysis_evidence_fingerprint', 'iteration', 'final_decision'},
}
PRIMARY_OUTPUT = {AgentName.DATA: 'market_data', AgentName.RESEARCH: 'research_evidence',
    AgentName.TECHNICAL: 'technical_result', AgentName.FUNDAMENTAL: 'fundamental_result',
    AgentName.RISK: 'risk_result', AgentName.MANAGER: 'final_decision'}


class AgentContractError(ValueError):
    pass


def validate_update(state, role, update, allowed_outputs=None):
    if not isinstance(update, dict):
        raise AgentContractError('Agent update must be a mapping')
    allowed = OUTPUTS.get(role) if allowed_outputs is None else set(allowed_outputs)
    if allowed is not None and set(update) - (COMMON | allowed):
        raise AgentContractError('Agent attempted to update state outside its role')
    for message in update.get('messages', []):
        if getattr(message, 'sender', None) != role:
            raise AgentContractError('Agent cannot impersonate another sender')
        if message.message_type is MessageType.FINAL_DECISION and role is not AgentName.MANAGER:
            raise AgentContractError('Only Manager can issue a final decision')
    for report in update.get('agent_reports', []):
        if getattr(report, 'agent', None) != role:
            raise AgentContractError('Agent cannot report on behalf of another role')
    for field in ('trace', 'llm_reviews'):
        if any(getattr(item, 'agent', None) != role for item in update.get(field, [])):
            raise AgentContractError('Agent cannot impersonate another role in its audit envelope')
    # Validate a candidate before exposing any of its values to sibling nodes.
    AgentState.model_validate({**state.model_dump(), **update})


def unavailable_result(state, agent, node, error):
    """Missing observations are uncertainty, not an invented issuer-risk event."""
    reason = f'{agent.name.value} unavailable ({type(error).__name__}); no unsupported conclusion was produced.'
    update = {'messages': [agent.message(AgentName.MANAGER, MessageType.WARNING, reason)],
        'trace': [agent.event('OBSERVE', 'STAGE_UNAVAILABLE', reason)],
        'agent_reports': [agent.report(stage=node, status='UNAVAILABLE', summary=reason,
            confidence=0, output_key='price_forecast' if node.startswith('forecast') else PRIMARY_OUTPUT[agent.name],
            missing_information=[reason])],
        'completed_tasks': []}
    if agent.name == AgentName.TECHNICAL:
        update['technical_result'] = TechnicalAnalysis(signal=Signal.UNKNOWN, score=5, confidence=0,
            evidence=[], risks=[reason], tools_used=[], reasoning_summary=reason, timeframe=state.investment_horizon)
    elif agent.name == AgentName.FUNDAMENTAL:
        update['fundamental_result'] = FundamentalAnalysis(signal=Signal.UNKNOWN, score=5, confidence=0,
            strengths=[], weaknesses=[reason], evidence=[], reasoning_summary=reason)
    elif agent.name == AgentName.RISK:
        update['risk_result'] = RiskAnalysis(assessment_status='UNAVAILABLE', risk_level=RiskLevel.LOW, risk_score=0,
            confidence=0, criticisms=[], warnings=[reason], veto=False,
            data_uncertainty_level=RiskLevel.HIGH, data_uncertainty_score=10,
            reasoning_summary=reason, policy_version='risk-unavailable-v1',
            sections=[RiskSection(name='assessment', status='UNKNOWN', coverage=0,
                                  missing_information=[reason])])
    elif agent.name == AgentName.RESEARCH:
        # A failed optional refresh must not erase previously verified citations.
        update.update(research_evidence=state.research_evidence,
                      research_freshness_note=reason)
    return update


def execute_agent(state, agent, node, action, *, essential=False, check_budget=True, allowed_outputs=None):
    from stock_multi_agent.services.request_telemetry import check_deadline
    with execution_context(state, agent.name, node) as context:
        error = None
        try:
            if check_budget:
                check_deadline()
            # Only validated output keys may escape a stage. A mutable nested
            # input must not let one specialist alter a parallel sibling's data.
            update = action(state.model_copy(deep=True))
            if check_budget:
                check_deadline()
            validate_update(state, agent.name, update, allowed_outputs)
        except Exception as exc:
            # Initial issuer data is essential. Preserve the checkpoint for a
            # later resume instead of inventing bars or a final recommendation.
            if essential or agent.name is AgentName.MANAGER:
                raise
            error = type(exc).__name__
            update = unavailable_result(state, agent, node, exc)
            if allowed_outputs is not None and 'price_forecast' in allowed_outputs:
                update['price_forecast'] = None
        reports = update.get('agent_reports', [])
        status = 'UNAVAILABLE' if error else ('PARTIAL' if any(r.status != 'COMPLETED' for r in reports) else 'COMPLETED')
        update['task_executions'] = [TaskExecution(run_id=state.run_id, task_id=context.task_id,
            node=node, agent=agent.name, input_fingerprint=context.input_fingerprint,
            attempt=1 + sum(item.node == node for item in state.task_executions),
            status=status, output_keys=sorted(set(update) - COMMON), error_type=error)]
        return update
