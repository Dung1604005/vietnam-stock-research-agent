"""Explicit local-user navigation, not a broker integration or agent tool.

Only the UI completion hook invokes the OS browser. No launch on import/history
render, no arbitrary LLM URLs, no login/order submission, no network fetching.
"""
from __future__ import annotations

from collections.abc import MutableMapping
import re
import unicodedata
from urllib.parse import parse_qsl, unquote, urlsplit
from uuid import uuid4

from stock_multi_agent.models.chat import AssistantResponse
from stock_multi_agent.services.interaction_actions import build_interaction_actions, validate_external_url, validate_broker_template
from stock_multi_agent.services.source_registry import SourceRegistry

_PORTALS = {'dstock.vndirect.com.vn', 'web.stockbiz.vn', 'vn.tradingview.com', 'www.tradingview.com'}


def is_local_browser_host(host: str | None) -> bool:
    """Do not open the server machine's browser for a remote Streamlit visitor."""
    if not isinstance(host, str) or not host or any(c.isspace() for c in host):
        return False
    try:
        parsed = urlsplit('//' + host)
        return (parsed.hostname in {'localhost', '127.0.0.1', '::1'}
                and parsed.username is None and parsed.password is None
                and not parsed.path and not parsed.query and not parsed.fragment)
    except ValueError:
        return False


def _plain(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', value.lower().replace('đ', 'd')) if not unicodedata.combining(c))


def _requested_kinds(response, request_text):
    requested = response.metadata.get('navigation_request', {})
    kind = requested.get('kind') if isinstance(requested, dict) else None
    if kind in {'STOCK', 'QUOTE', 'CHART', 'REPORT'}:
        return [kind]
    if kind == 'BROKER':
        return ['BROKER', 'STOCK']
    text = _plain(request_text)
    if re.search(r'\bmo\b.*\b(?:cong giao dich|dat lenh|vndirect|dboard)\b', text) and 'dstock' not in text:
        return ['BROKER', 'STOCK']
    if re.search(r'\b(?:bao cao|cong bo|filing|report)\b', text):
        return ['REPORT']
    if re.search(r'\b(?:bieu do|chart|do thi)\b', text):
        return ['CHART']
    if re.search(r'\b(?:xem gia|bang gia|gia hien tai|gia hom nay)\b', text):
        return ['QUOTE']
    return ['STOCK', 'QUOTE', 'REPORT', 'CHART']


def _safe_for_automatic_open(action, registry, broker_template):
    try:
        validate_external_url(action.url)
        parsed = urlsplit(action.url)
        # Even an official domain is not permission to invoke an account action.
        if re.search(r'(?:^|[/_.-])(?:submit|execute|placeorder|place-order|cancelorder|buy|sell)(?:$|[/_.-])', unquote(parsed.path + '/' + parsed.fragment.split('?', 1)[0]).lower()):
            return False
        query_parts = [parsed.query]
        if '?' in parsed.fragment:
            query_parts.append(parsed.fragment.split('?', 1)[1])
        if any(key.lower() in {'action', 'side', 'quantity', 'qty', 'price', 'order', 'command', 'redirect', 'redirect_url', 'returnurl', 'url'}
               for part in query_parts for key, _ in parse_qsl(part, keep_blank_values=True)):
            return False
        if action.kind in {'STOCK', 'QUOTE', 'CHART'}:
            return parsed.hostname in _PORTALS
        if action.kind == 'BROKER' and broker_template:
            validate_broker_template(broker_template)
            return action.url == broker_template.format(ticker=action.ticker or '')
        if action.kind == 'REPORT':
            source = registry.resolve(action.url)
            return bool(source and source.quality_tier in {'T1', 'T2'} and source.accepts(action.url))
    except (ValueError, TypeError, KeyError, OSError):
        pass
    return False


def open_response_actions(response: AssistantResponse, request_text: str, *, state: MutableMapping,
                          host: str | None, execution_mode='STANDARD', enabled=True, max_tabs=3,
                          broker_url_template='https://trade.vndirect.com.vn/', broker_name='VNDIRECT',
                          opener=None, registry=None) -> dict:
    """At-most-once bounded launch for a completed NEW request, never on rerun.

    Caller must pass actual browser Host from Streamlit context, not LLM data.
    Original model/financial decisions are not changed by this feature.
    """
    mode = response.metadata.get('execution_profile', {})
    if not isinstance(mode, dict):
        mode = {}
    if not enabled:
        return {'status': 'DISABLED', 'opened': [], 'skipped': []}
    if execution_mode == 'OFFLINE' or mode.get('mode') == 'OFFLINE' or mode.get('external_allowed') is False:
        return {'status': 'OFFLINE_BLOCKED', 'opened': [], 'skipped': []}
    if not is_local_browser_host(host):
        return {'status': 'NOT_LOCALHOST', 'opened': [], 'skipped': []}
    identifier = response.metadata.setdefault('browser_action_id', uuid4().hex)
    ledger = dict(state.get('interaction_browser_ledger', {}))
    if identifier in ledger:
        return {**ledger[identifier], 'status': 'ALREADY_HANDLED'}
    limit = max(1, min(5, int(max_tabs)))
    try:
        actions = build_interaction_actions(response, broker_url_template=broker_url_template, broker_name=broker_name)
        registry = registry or SourceRegistry()
    except (ValueError, OSError):
        return {'status': 'INVALID_CONFIGURATION', 'opened': [], 'skipped': []}
    kinds = _requested_kinds(response, request_text)
    selected, skipped, seen_urls, seen_groups = [], [], set(), set()
    for kind in kinds:
        for action in actions:
            if action.kind != kind or action.url in seen_urls:
                continue
            # A tickerless report is not attributed to an arbitrary stock in a list.
            if kind == 'REPORT' and len(response.tickers) > 1 and not action.ticker:
                skipped.append({'label': action.label, 'reason': 'AMBIGUOUS_REPORT_TICKER'})
                continue
            group = (action.ticker, kind)
            if kind in {'STOCK', 'QUOTE', 'CHART', 'BROKER'} and group in seen_groups:
                continue
            if not _safe_for_automatic_open(action, registry, broker_url_template):
                skipped.append({'label': action.label, 'reason': 'UNTRUSTED_AUTOMATIC_DESTINATION'})
                continue
            selected.append(action)
            seen_urls.add(action.url)
            seen_groups.add(group)
    # Mark before opening so render retries/errors never launch tabs repeatedly.
    result = {'status': 'NO_SAFE_ACTIONS', 'opened': [], 'failed': [], 'skipped': skipped,
              'tab_limit': limit, 'remaining_count': max(0, len(selected)-limit)}
    ledger[identifier] = result
    state['interaction_browser_ledger'] = dict(list(ledger.items())[-100:])
    if opener is None:
        import webbrowser
        opener = lambda url: webbrowser.open(url, new=2, autoraise=False)
    for action in selected[:limit]:
        try:
            opened = bool(opener(action.url))
        except Exception:
            opened = False
        record = {'label': action.label, 'url': action.url, 'kind': action.kind, 'ticker': action.ticker}
        result['opened' if opened else 'failed'].append(record)
    if result['opened']:
        result['status'] = 'OPENED' if not result['failed'] else 'PARTIAL'
    elif result['failed']:
        result['status'] = 'BROWSER_UNAVAILABLE'
    # Streamlit session state is not a normal nested mutable-dict proxy.
    ledger[identifier] = result
    state['interaction_browser_ledger'] = dict(list(ledger.items())[-100:])
    return result
