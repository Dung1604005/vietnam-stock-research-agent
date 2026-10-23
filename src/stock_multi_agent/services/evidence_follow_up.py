"""Manager-approved, one-shot evidence actions. LLM text never executes code/tools."""
from __future__ import annotations

import hashlib
import json
import time

from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.services.request_telemetry import current_metrics


ALLOWED_TASKS = frozenset({'FETCH_FINANCIAL_HISTORY', 'VERIFY_FINANCIAL_FACT',
                           'FIND_RECENT_REPORT', 'CHECK_RISK_EVENT'})
ADDITIVE_FIELDS = ('messages', 'llm_reviews', 'agent_reports', 'trace', 'completed_tasks',
                   'reanalysis_history', 'evidence_follow_up_outcomes', 'task_executions')


def evidence_fingerprint(state):
    def stable(value):
        if isinstance(value, dict):
            return {key: stable(item) for key, item in value.items() if key not in {'fetched_at', 'checked_at', 'retrieved_at'}}
        return [stable(item) for item in value] if isinstance(value, list) else value
    payload = stable({'ticker': state.ticker,
        'market': state.market_data.model_dump(mode='json') if state.market_data else None,
        'benchmark': state.benchmark_data.model_dump(mode='json') if state.benchmark_data else None,
        'research': [item.model_dump(mode='json') for item in state.research_evidence]})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def approve_proposal(state, review, proposal):
    """Return a policy reason, not model-supplied authority or arbitrary actions."""
    if not state.allow_llm_planning or state.max_iterations <= state.iteration:
        return False, 'iteration_or_batch_budget'
    if state.evidence_follow_up_outcomes:
        return False, 'one_evidence_follow_up_per_request'
    if review.status != 'GROUNDED' or review.agent not in {AgentName.FUNDAMENTAL, AgentName.RISK}:
        return False, 'untrusted_review'
    if proposal.task_kind not in ALLOWED_TASKS:
        return False, 'task_not_allowed'
    if proposal.metric:
        from stock_multi_agent.services.claim_grounding import canonical_metric
        if canonical_metric(proposal.metric) not in {'revenue', 'net_profit', 'operating_cash_flow',
            'total_assets', 'total_equity', 'total_liabilities', 'roe', 'roa', 'eps', 'pe', 'pb',
            'debt_to_equity', 'profit_margin', 'revenue_growth', 'profit_growth'}:
            return False, 'unsupported_metric'
    allowed = {item.source_id for item in state.market_data.source_refs} if state.market_data else set()
    allowed.update(item.chunk_id for item in state.research_evidence)
    if not set(proposal.source_ids).issubset(allowed):
        return False, 'unknown_source'
    if not proposal.source_ids and proposal.task_kind not in {'FIND_RECENT_REPORT', 'FETCH_FINANCIAL_HISTORY'}:
        return False, 'missing_source_anchor'
    if proposal.task_kind == 'VERIFY_FINANCIAL_FACT' and review.grounding_status != 'CLAIM_ANCHORS_VERIFIED':
        return False, 'fact_verification_needs_claim_anchor'
    if proposal.period_end and state.market_data and proposal.period_end > min(state.market_data.as_of,
            state.market_data.fetched_at.date()).isoformat():
        return False, 'future_reporting_period'
    metrics = current_metrics()
    if metrics and (metrics.budget.deadline_seconds - (time.perf_counter() - metrics.started) < 10
                    or metrics.budget.max_tool_calls - metrics.tool_calls < 4):
        return False, 'remaining_request_budget'
    return True, 'approved_allowlisted_evidence_action'


def _apply(state, delta):
    for field, value in delta.items():
        setattr(state, field, [*getattr(state, field), *value] if field in ADDITIVE_FIELDS else value)


def run_evidence_follow_up(state, graph):
    """At most one approved action, no graph recursion; retain original on failure."""
    if not state.allow_llm_planning or state.evidence_follow_up_outcomes:
        return {}
    attempted = []
    selected = None
    for review in reversed(state.llm_reviews):
        for proposal in review.follow_up_proposals:
            approved, reason = graph.manager_agent.approve_evidence_follow_up(state, review, proposal)
            record = {'agent': review.agent.value, **proposal.model_dump(mode='json'),
                      'approval': reason, 'status': 'APPROVED' if approved else 'REJECTED'}
            if approved:
                selected = record
                break
            attempted.append(record)
        if selected:
            break
    if selected is None:
        return {'evidence_follow_up_outcomes': attempted} if attempted else {}
    work = state.model_copy(deep=True)
    before = evidence_fingerprint(state)
    lengths = {key: len(getattr(state, key)) for key in ADDITIVE_FIELDS}
    from stock_multi_agent.services.agent_execution import execute_agent
    def run(agent, node, action, **kwargs):
        _apply(work, execute_agent(work, agent, node, action, **kwargs))
    try:
        metrics = current_metrics()
        if metrics:
            metrics.consume('tool')  # covers even fixture/custom tools without instrumentation
        if selected['task_kind'] == 'FETCH_FINANCIAL_HISTORY':
            work.required_data = list(dict.fromkeys([*work.required_data, 'financials']))
            run(graph.data_agent, 'data_follow_up', lambda value: graph.data_agent.run(value, refresh=True), essential=True)
        else:
            # Exact approved objective is context, never a URL/function name to execute.
            work.user_goal = f'{state.user_goal}\nEvidence follow-up: {selected["task_kind"]}; metric={selected.get("metric") or "unspecified"}; period={selected.get("period_end") or "latest"}'
            run(graph.research_agent, 'research_follow_up', lambda value: graph.research_agent.run(value,
                expanded=True, follow_up_task=selected['task_kind'], follow_up_metric=selected.get('metric'),
                follow_up_period=selected.get('period_end')))
            work.user_goal = state.user_goal
            advisor = getattr(graph.fundamental_agent, 'llm_advisor', None)
            if selected['task_kind'] in {'VERIFY_FINANCIAL_FACT', 'FIND_RECENT_REPORT'} and advisor is not None and hasattr(advisor, 'map_statement_facts'):
                from stock_multi_agent.services.research_financial_facts import apply_research_financial_proposals
                proposals = advisor.map_statement_facts(work.research_evidence, work.ticker)
                work.market_data, audit = apply_research_financial_proposals(work.market_data,
                    work.research_evidence, proposals, work.ticker)
                work.financial_fact_proposals = audit
                selected['financial_fact_audit'] = audit
        changed = evidence_fingerprint(work) != before
        selected['status'] = 'NEW_EVIDENCE' if changed else 'NO_NEW_EVIDENCE'
        if changed:
            # Recompute from data, never copy an LLM-proposed numerical score.
            if selected['task_kind'] == 'FETCH_FINANCIAL_HISTORY':
                run(graph.data_agent, 'forecast_follow_up', graph._forecast, allowed_outputs={'price_forecast'})
                run(graph.technical_agent, 'technical_follow_up', lambda value: graph.technical_agent.run(value,
                    extended=True, allow_llm_planning=False))
                from stock_multi_agent.models.llm import AgentLLMReview
                if any(item.agent == AgentName.TECHNICAL for item in work.llm_reviews):
                    work.llm_reviews.append(AgentLLMReview(agent=AgentName.TECHNICAL,
                        task='Invalidate pre-refresh technical review', status='UNAVAILABLE',
                        assessment='Dữ liệu đã thay đổi; giữ chỉ báo Python tính lại, không tái sử dụng review LLM cũ.',
                        grounding_audit={'invalidated_after_data_refresh': True}))
            run(graph.fundamental_agent, 'fundamental_follow_up', lambda value: graph.fundamental_agent.run(value, allow_llm_review=False))
            run(graph.risk_agent, 'risk_follow_up', lambda value: graph.risk_agent.run(value, allow_llm_review=False))
            _apply(work, graph._specialist_reviews(work))
            work.active_conflicts = graph.manager_agent.assess_conflicts(work)
            work.conflict_detected = bool(work.active_conflicts)
        else:
            # No changed evidence means no refreshed timestamps mixed with old analyses.
            audit = {key: getattr(work, key)[lengths[key]:]
                     for key in ('messages', 'trace', 'agent_reports', 'llm_reviews', 'task_executions')}
            work = state.model_copy(deep=True)
            _apply(work, audit)
        work.iteration = state.iteration + 1
        selected.update(before_fingerprint=before, after_fingerprint=evidence_fingerprint(work))
        work.evidence_follow_up_outcomes.extend([*attempted, selected])
        work.trace.append(graph.manager_agent.event('ACT', 'BOUNDED_EVIDENCE_FOLLOW_UP',
            f'{selected["task_kind"]}: {selected["status"]}; no recursive retry'))
        delta = {key: getattr(work, key)[lengths[key]:] for key in ADDITIVE_FIELDS}
        for key in ('market_data', 'benchmark_data', 'price_forecast', 'research_evidence',
                    'research_freshness_note', 'technical_result', 'fundamental_result', 'risk_result',
                    'financial_fact_proposals', 'iteration'):
            delta[key] = getattr(work, key)
        delta['active_conflicts'] = work.active_conflicts
        delta['conflict_detected'] = work.conflict_detected
        return delta
    except Exception as exc:
        # Atomic state hand-off: a partial refresh must not mix old results with new data.
        selected.update(status='FAILED', error_type=type(exc).__name__)
        return {'evidence_follow_up_outcomes': [*attempted, selected], 'iteration': state.iteration + 1,
                'trace': [graph.manager_agent.event('OBSERVE', 'EVIDENCE_FOLLOW_UP_FAILED',
                    f'Kept previous evidence/results; {type(exc).__name__}')]}
