"""Source-scoped, conservative claim checks. These do NOT prove semantic truth.

Quotes must come from one explicitly linked observation or research excerpt.
Unscoped financial totals, source metadata and other issuers cannot lend numbers
to a claim merely because they share an evidence pack.
"""
from __future__ import annotations

import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation


def plain(value):
    return ''.join(ch for ch in unicodedata.normalize('NFD', str(value).casefold().replace('đ', 'd'))
                   if unicodedata.category(ch) != 'Mn')


_ALIASES = {
    'revenue': ('revenue', 'doanh thu'),
    'net_profit': ('net profit', 'net_profit', 'loi nhuan sau thue', 'loi nhuan rong'),
    'operating_cash_flow': ('operating_cash_flow', 'operating cash flow', 'dong tien kinh doanh', 'dong tien tu hoat dong kinh doanh'),
    'roe': ('roe', 'loi nhuan tren von chu'), 'eps': ('eps', 'thu nhap moi co phieu'),
    'pe': ('p/e', 'pe', 'price_to_earnings'), 'pb': ('p/b', 'pb', 'price_to_book'),
    'debt_to_equity': ('debt_to_equity', 'debt to equity', 'no tren von chu', 'no/von chu'),
    'volatility': ('volatility', 'bien dong'), 'max_drawdown': ('max_drawdown', 'drawdown', 'sut giam'),
    'rsi': ('rsi', 'rsi14'), 'close': ('close', 'gia dong cua'),
    'gross_margin': ('gross_margin', 'gross margin', 'bien loi nhuan gop'),
    'net_margin': ('net_margin', 'net margin', 'bien loi nhuan rong'),
}


def canonical_metric(value):
    value = plain(value).strip()
    for name, aliases in _ALIASES.items():
        if value == name or value in aliases:
            return name
    return value.replace(' ', '_')


def metric_present(metric, text):
    name = canonical_metric(metric)
    aliases = _ALIASES.get(name, (name, name.replace('_', ' ')))
    return any(re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', plain(text)) for alias in aliases)


def numeric_tokens(text):
    """Normalize formatting only: 20.0%=20%; 1,000=1000; 1,5=1.5.

    Never silently convert a fraction into a percentage or a monetary scale.
    Single dot is a decimal separator (the structured payload uses JSON floats).
    """
    result = set()
    pattern = r'(?<![\w])[-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?(?:\s*%)?'
    for match in re.finditer(pattern, str(text).replace('−', '-')):
        token = match.group().strip()
        percent = token.endswith('%')
        token = token.rstrip('%').strip()
        if ',' in token and '.' in token:
            if token.rfind(',') > token.rfind('.'):
                token = token.replace('.', '').replace(',', '.')
            else:
                token = token.replace(',', '')
        elif ',' in token:
            token = token.replace(',', '') if re.fullmatch(r'[-+]?\d{1,3}(?:,\d{3})+', token) else token.replace(',', '.')
        try:
            number = Decimal(token)
            if number.is_finite():
                result.add((number, '%' if percent else ''))
        except InvalidOperation:
            continue
    return result


def unsupported_numbers(claim, source):
    return bool(numeric_tokens(claim).difference(numeric_tokens(source)))


def _unit(value):
    value = plain(value).strip()
    return {'%': 'percent', 'phan tram': 'percent', 'ty le phan tram': 'percent',
            'dong': 'vnd', 'lan': 'ratio', 'usd': 'usd'}.get(value, value)


def build_evidence_pack(facts, allowed_source_ids):
    """Build a bounded list of source-bound snippets for exact-copy citations."""
    allowed = set(allowed_source_ids)
    ticker = facts.get('ticker')
    output, seen = [], set()

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            ids = [value.get('chunk_id'), value.get('source_id'), *value.get('source_ids', [])]
            ids = [item for item in ids if isinstance(item, str) and item in allowed]
            # Metadata alone is not evidence of a measured financial value.
            content = value.get('quote') or value.get('exact_quote') or value.get('evidence_quote')
            if not content and 'value' in value and ('metric' in value or 'label' in value):
                structured = {key: value[key] for key in ('metric', 'label', 'value', 'unit', 'period_end', 'as_of') if key in value}
                if value.get('unit') == 'percent' and isinstance(value['value'], (int, float)):
                    structured['display_value'] = f"{value['value']}%"
                content = json.dumps(structured, ensure_ascii=False, default=str)
            for source_id in ids:
                if not isinstance(content, str) or not content.strip():
                    continue
                source_tickers = value.get('tickers') or [value.get('ticker') or ticker]
                entry = {'source_id': source_id, 'quote': content[:2000],
                         'tickers': [item for item in source_tickers if item],
                         'metric': value.get('metric'), 'unit': value.get('unit'),
                         'period_end': str(value.get('period_end') or value.get('as_of') or '')[:10] or None}
                key = (source_id, entry['quote'])
                if key not in seen and len(output) < 32:
                    if len(json.dumps([*output, entry], ensure_ascii=False).encode('utf-8')) <= 12000:
                        output.append(entry)
                        seen.add(key)
            for item in value.values():
                if isinstance(item, (list, dict)):
                    visit(item)
    visit(facts)
    return output


def audit_structured_claim(claim, evidence_pack, expected_ticker=None):
    errors = []
    quote = ' '.join(claim.exact_quote.split())
    candidates = [item for item in evidence_pack if item['source_id'] == claim.source_id
                  and quote in ' '.join(item['quote'].split())]
    if not candidates:
        return ['quote_not_in_cited_source']
    if unsupported_numbers(claim.claim_vi, quote):
        errors.append('numbers_not_in_cited_quote')
    if claim.value is not None and Decimal(str(claim.value)) not in {number for number, _ in numeric_tokens(quote)}:
        errors.append('value_not_in_cited_quote')
    if numeric_tokens(claim.claim_vi):
        if not claim.metric:
            errors.append('numeric_claim_missing_metric_anchor')
        if not claim.period_end:
            errors.append('numeric_claim_missing_period_anchor')
        if not claim.unit:
            errors.append('numeric_claim_missing_unit_anchor')
        if expected_ticker and not claim.ticker:
            errors.append('numeric_claim_missing_ticker_anchor')
    if claim.metric and (not metric_present(claim.metric, quote) or not metric_present(claim.metric, claim.claim_vi)):
        errors.append('metric_not_anchored_in_quote_and_claim')
    if claim.metric and not any(not item['metric'] or canonical_metric(item['metric']) == canonical_metric(claim.metric) for item in candidates):
        errors.append('metric_metadata_mismatch')
    for metric in _ALIASES:
        if metric_present(metric, claim.claim_vi) and not metric_present(metric, quote):
            errors.append('additional_metric_not_in_quote')
    if claim.period_end and not any(item['period_end'] == claim.period_end or claim.period_end in quote for item in candidates):
        errors.append('period_mismatch')
    # Explicit year in narrative must be anchored by the same quoted snippet.
    years = set(re.findall(r'\b(?:19|20)\d{2}\b', claim.claim_vi))
    if years.difference(re.findall(r'\b(?:19|20)\d{2}\b', quote)):
        errors.append('year_not_in_cited_quote')
    target = claim.ticker or expected_ticker
    if expected_ticker and claim.ticker and claim.ticker != expected_ticker:
        errors.append('wrong_analysis_ticker')
    if target and not any(not item['tickers'] or target in item['tickers'] for item in candidates):
        errors.append('issuer_mismatch')
    symbols = set(re.findall(r'\b[A-Z][A-Z0-9]{2,4}\b', claim.claim_vi))
    acronyms = {'ROE', 'ROA', 'EPS', 'RSI', 'RSI14', 'MACD', 'ATR', 'ADX', 'VND', 'USD', 'FCF', 'CFO', 'TTM', 'YOY', 'EBIT', 'NIM', 'LSTM', 'LLM', 'ETF', 'GDP'}
    if target and symbols.difference(acronyms | {target}):
        errors.append('unverified_issuer_symbol_in_claim')
    if claim.unit:
        unit = _unit(claim.unit)
        metadata = any(item['unit'] and _unit(item['unit']) == unit for item in candidates)
        patterns = {'percent': r'%', 'ratio': r'\b(?:ratio|lan)\b', 'vnd': r'\b(?:vnd|dong)\b', 'usd': r'\busd\b'}
        in_quote = bool(re.search(patterns.get(unit, re.escape(unit)), plain(quote)))
        if not metadata and not in_quote:
            errors.append('unit_not_anchored')
        if any(item['unit'] for item in candidates) and not any(not item['unit'] or _unit(item['unit']) == unit for item in candidates):
            errors.append('unit_metadata_mismatch')
        if unit == 'vnd' and re.search(r'\b(?:ty|trieu|nghin|billion|million|thousand)\b', plain(quote)):
            errors.append('monetary_scale_not_anchored')
    # Financial quantities require a unit, preventing use of ratio as percent.
    if claim.value is not None and not claim.unit:
        errors.append('value_missing_unit_anchor')
    return list(dict.fromkeys(errors))


def audit_review(review, evidence_pack, expected_ticker=None):
    if not review.claims:
        return {'status': 'LEGACY_REVIEW_LEVEL_ONLY', 'verified_claim_count': 0,
                'issues': ['No claim-level quotes supplied; review-level citations do not establish claim support.'],
                'semantic_truth_verified': False}
    checks = [{'index': index, 'source_id': claim.source_id,
               'issues': audit_structured_claim(claim, evidence_pack, expected_ticker)}
              for index, claim in enumerate(review.claims)]
    narrative = [review.assessment, *review.supporting_observations, *review.counterarguments]
    # A valid claim cannot launder unrelated numbers in the surrounding review.
    quoted_numbers = set().union(*(numeric_tokens(claim.exact_quote) for claim in review.claims))
    narrative_issues = ['unanchored_narrative_number'] if any(numeric_tokens(text).difference(quoted_numbers) for text in narrative) else []
    for text in narrative:
        for sentence in re.split(r'(?<=[.!?;])\s+(?=[A-ZĐÀ-Ỹ])|\n+', text):
            numbers = numeric_tokens(sentence)
            if not numbers:
                continue
            metrics = [metric for metric in _ALIASES if metric_present(metric, sentence)]
            if not any(numbers.issubset(numeric_tokens(claim.exact_quote))
                       and all(metric_present(metric, claim.exact_quote) for metric in metrics)
                       for claim in review.claims):
                narrative_issues.append('narrative_metric_number_not_source_scoped')
    narrative_issues = list(dict.fromkeys(narrative_issues))
    rejected = any(item['issues'] for item in checks) or bool(narrative_issues)
    return {'status': 'REJECTED' if rejected else 'CLAIM_ANCHORS_VERIFIED',
            'verified_claim_count': sum(not item['issues'] for item in checks), 'claims': checks,
            'issues': narrative_issues, 'semantic_truth_verified': False}
