"""Public-safe versioned export. Never dump response metadata or request parameters."""
from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit


_PRIVATE_KEY = re.compile(r'(?:secret|password|api.?key|token|authorization|cookie|header|credential|request_parameters|prompt|reasoning_chain|traceback|exception|local_path|portfolio_weights)', re.I)
_SAFE_VERSION_KEYS = {'prompt_version', 'prompt_sha256', 'prompt_hash', 'schema_version',
    'schema_hash', 'schema_sha256', 'tokenizer_revision', 'tokenizer_hash', 'tokenizer_sha256',
    'max_output_tokens', 'output_token_upper_bound', 'token_usage', 'max_new_tokens'}
_URL = re.compile(r'https?://[^\s<>\]\)"\']+', re.I)
_ALLOWED_METADATA = {
    'execution_profile', 'data_origins', 'request_metrics', 'progress_events', 'execution_events',
    'research_citations', 'knowledge_citations', 'llm_reviews', 'llm_specialist_reviews', 'claim_audit',
    'analysis_explanation', 'financial_evidence', 'canonical_financial_evidence',
    'evidence_follow_up', 'evidence_follow_up_outcomes', 'model_identity',
    'technical_score', 'fundamental_score', 'risk_score', 'data_uncertainty_score',
    'recommendation', 'decision', 'policy_version', 'forecast', 'price_forecast',
    'financial_fact_proposals', 'evidence_bundle', 'risk_coverage', 'chart_limitations',
    'decision_policy_version', 'decision_calibration_status', 'technical_signal',
    'technical_policy_version', 'technical_feature_contributions', 'technical_diagnostics',
    'fundamental_signal', 'fundamental_policy_version', 'fundamental_feature_contributions',
    'risk_level', 'risk_policy_version', 'risk_feature_contributions', 'issuer_risk_score',
    'issuer_risk_level', 'data_uncertainty_level', 'evidence_completeness', 'conflicts',
    'agent_reports', 'agent_evaluation', 'financial_comparability_limitations',
    'analysis_run_id', 'workflow_version', 'analysis_status', 'risk_assessment_status',
    'task_execution', 'task_plan_status', 'task_executions',
    'prompt_version', 'prompt_sha256', 'schema_version', 'schema_sha256',
}
_SOURCE_KEYS = {'provider', 'source_id', 'data_kind', 'fetched_at', 'as_of', 'source_url',
    'schema_version', 'content_sha256', 'publisher', 'published_at', 'retrieved_at',
    'chunk_id', 'document_id', 'quote', 'quality_tier', 'reporting_period', 'period_end',
    'checksum', 'title', 'tickers', 'period_start', 'document_type'}
_SENSITIVE_NESTED = {'messages', 'raw_query', 'user_query', 'response_text', 'raw_response',
    'raw_output', 'raw_error', 'endpoint', 'base_url', 'url', 'snapshot_id', 'snapshot_path',
    'db_path', 'database_path', 'weights', 'positions', 'portfolio', 'portfolio_risk',
    'query', 'question', 'search_query', 'input', 'input_text', 'user_goal', 'conversation',
    'conversation_id', 'conversation_history', 'chain_of_thought', 'reasoning', 'raw'}


def sanitize_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        if parts.scheme not in {'http', 'https'} or not parts.hostname:
            return '[URL omitted]'
        host = parts.hostname
        if ':' in host:
            host = f'[{host}]'
        if parts.port:
            host += f':{parts.port}'
        return urlunsplit((parts.scheme, host, parts.path, '', ''))
    except ValueError:
        return '[URL omitted]'


def sanitize_text(value: str) -> str:
    value = _URL.sub(lambda match: sanitize_url(match.group()), str(value))
    value = re.sub(r'(?i)\b(?:Bearer\s+)[A-Za-z0-9._~+/=-]+', '[REDACTED]', value)
    value = re.sub(r'(?i)\b(?:api[_ -]?key|password|secret|access[_ -]?token|authorization)\s*[:=]\s*[^\s,;]+', '[REDACTED]', value)
    value = re.sub(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', '[EMAIL REDACTED]', value, flags=re.I)
    value = re.sub(r'(?i)\b(?:phone|tel|sđt|điện thoại|account|số tài khoản)\s*[:=]\s*[+\d][\d .-]{7,}', '[PII REDACTED]', value)
    value = re.sub(r'(?<!\w)\+84[ .-]?\d(?:[ .-]?\d){8,9}\b', '[PHONE REDACTED]', value)
    value = re.sub(r'(?i)\b[A-Z]:[\\/][^\r\n<>"\']+', '[LOCAL PATH REDACTED]', value)
    value = re.sub(r'/(?:home|Users|mount/src|kaggle/input)/[^\s<>"\']+', '[LOCAL PATH REDACTED]', value)
    return value


def sanitize_payload(value, *, include_private_portfolio=False, depth=0):
    """Sanitize already allowlisted export sections; never use as a raw state dumper."""
    if depth > 14:
        return '[TRUNCATED]'
    if hasattr(value, 'model_dump'):
        value = value.model_dump(mode='json')
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key = str(key)
            if (_PRIVATE_KEY.search(key) and key not in _SAFE_VERSION_KEYS) or (key in _SENSITIVE_NESTED and not (include_private_portfolio and key in {'weights', 'positions', 'portfolio', 'portfolio_risk'})):
                continue
            if key.endswith('_url') and isinstance(item, str):
                result[key] = sanitize_url(item)
                continue
            result[sanitize_text(key)] = sanitize_payload(item, include_private_portfolio=include_private_portfolio, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize_payload(item, include_private_portfolio=include_private_portfolio, depth=depth + 1) for item in value[:10000]]
    if isinstance(value, str):
        return sanitize_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_text(str(value))


def build_audit(response, *, include_private_portfolio=False) -> dict:
    raw = response.model_dump(mode='json') if hasattr(response, 'model_dump') else dict(response)
    metadata = raw.get('metadata', {})
    selected = {key: metadata[key] for key in _ALLOWED_METADATA if key in metadata}
    if include_private_portfolio and 'portfolio_risk' in metadata:
        selected['portfolio_risk'] = metadata['portfolio_risk']
    sources = [{key: source[key] for key in _SOURCE_KEYS if key in source}
               for source in raw.get('source_details', [])]
    charts = []
    for chart in raw.get('charts', []):
        charts.append({key: chart[key] for key in ('request', 'title', 'records', 'data_timestamp',
            'data_fetched_at', 'source', 'axis_unit', 'explanation', 'warnings') if key in chart})
    audit = {'schema_version': 'research-audit-v1', 'generated_at': datetime.now(timezone.utc).isoformat(),
        'tickers': raw.get('tickers', []), 'message': raw.get('message', ''),
        'data_timestamp': raw.get('data_timestamp'), 'data_fetched_at': raw.get('data_fetched_at'),
        'source_details': sources, 'tables': raw.get('tables', []), 'charts': charts,
        'warnings': raw.get('warnings', []), 'audit': selected,
        'limitations': ['Research/classroom use only; not a trading instruction.',
            'Export omits credentials, raw prompts, request parameters and private portfolio by default.',
            'Chart records/specifications are exported, not interactive chart HTML.']}
    if 'portfolio_risk' in metadata and not include_private_portfolio:
        # Even message/source IDs/tables can reveal private holdings, not just weights metadata.
        audit.update(message='Private portfolio details omitted. Explicit opt-in is required to export this analysis.',
            tickers=[], source_details=[], tables=[], charts=[], warnings=[])
        audit['audit'] = {key: selected[key] for key in ('execution_profile', 'request_metrics') if key in selected}
    return sanitize_payload(audit, include_private_portfolio=include_private_portfolio)


def export_audit_json(response, **kwargs) -> str:
    return json.dumps(build_audit(response, **kwargs), ensure_ascii=False, indent=2, allow_nan=False)


def export_markdown(response, **kwargs) -> str:
    audit = build_audit(response, **kwargs)
    # Escape HTML from source/model text even when a Markdown viewer allows raw HTML.
    text = audit.pop('message')
    rendered = html.escape(text, quote=False)
    for key in ('tables', 'source_details', 'charts', 'warnings', 'audit', 'limitations'):
        payload = json.dumps(audit.pop(key), ensure_ascii=False, indent=2, allow_nan=False)
        rendered += '\n\n## ' + key + '\n\n```json\n' + html.escape(payload, quote=False).replace('```', '\\`\\`\\`') + '\n```'
    return rendered


def export_html(response, **kwargs) -> str:
    audit = build_audit(response, **kwargs)
    return '<!doctype html><html lang="vi"><meta charset="utf-8"><title>Research report</title><body><pre>' + html.escape(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)) + '</pre></body></html>'
