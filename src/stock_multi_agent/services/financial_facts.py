"""Conservative financial facts, compatible periods and reproducible derived evidence.

No magnitude heuristics, implicit FX, YTD-to-quarter conversion or missing-value
imputation. A reported value remains useful for display even if not comparable.
"""
from __future__ import annotations

import math
import re
import calendar
from collections import defaultdict
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stock_multi_agent.models.data import FinancialStatementObservation


class CanonicalFinancialFact(BaseModel):
    metric: str
    statement: str | None = None
    value: float
    normalized_value: float | None = None
    unit: str
    reported_unit: str | None = None
    currency: str | None = None
    scale: float | None = None
    period_label: str
    period_start: date | None = None
    period_end: date | None = None
    frequency: str = 'reported'
    reporting_basis: str = 'UNKNOWN'
    consolidation: str = 'UNKNOWN'
    assurance: str = 'UNKNOWN'
    published_at: datetime | None = None
    source_id: str
    source_url: str | None = None
    evidence_quote: str | None = None
    status: str = 'PARTIAL'
    limitations: list[str] = Field(default_factory=list)


class DerivedFinancialMetric(BaseModel):
    metric: str
    value: float
    unit: str = 'ratio'
    period_end: date
    previous_period_end: date | None = None
    formula: str
    input_fields: list[str]
    source_ids: list[str]
    evidence: dict
    note: str


_CURRENCY_UNITS = {
    'vnd': ('VND', 1.0), 'đồng': ('VND', 1.0), 'dong': ('VND', 1.0),
    'thousand vnd': ('VND', 1_000.0), 'nghìn đồng': ('VND', 1_000.0),
    'million vnd': ('VND', 1_000_000.0), 'triệu đồng': ('VND', 1_000_000.0),
    'billion vnd': ('VND', 1_000_000_000.0), 'tỷ đồng': ('VND', 1_000_000_000.0),
    'usd': ('USD', 1.0),
}


def normalize_observation(item) -> CanonicalFinancialFact:
    raw = item.model_dump()
    fact = CanonicalFinancialFact(**{key: value for key, value in raw.items()
        if key in CanonicalFinancialFact.model_fields})
    fact.reported_unit = fact.unit
    unit = fact.unit.strip().casefold()
    declaration = _CURRENCY_UNITS.get(unit)
    if declaration:
        currency, scale = declaration
        if fact.currency and fact.currency.upper() != currency:
            fact.limitations.append('Currency conflicts with the declared unit.')
            return fact
        if unit in {'vnd', 'usd'} and fact.scale is not None:
            scale = fact.scale  # explicit numeric scale alongside the base-currency code
        elif fact.scale is not None and fact.scale != scale:
            fact.limitations.append('Scale conflicts with the declared unit.')
            return fact
        fact.currency, fact.scale = currency, scale
        fact.unit = currency
        fact.normalized_value = fact.value * scale
    elif unit in {'ratio', 'multiple', 'percent', '%'}:
        fact.normalized_value = fact.value / 100 if unit in {'percent', '%'} else fact.value
        fact.unit = 'ratio' if unit in {'percent', '%'} else unit
        fact.scale = 1.0
    elif unit in {'currency', 'currency_per_share'} and fact.currency and fact.scale is not None:
        fact.normalized_value = fact.value * fact.scale
        fact.unit = fact.currency.upper() + ('/share' if unit == 'currency_per_share' else '')
    else:
        fact.limitations.append('Unit/currency/scale is not declared; value kept in source units only.')
    if fact.period_end is None:
        fact.limitations.append('Reporting period end is unknown.')
    if fact.period_start and fact.period_end and fact.period_start > fact.period_end:
        fact.limitations.append('Reporting period start is after period end.')
        fact.normalized_value = None
    if fact.reporting_basis == 'UNKNOWN':
        fact.limitations.append('Standalone, cumulative or annual accounting basis is unknown.')
    if fact.consolidation == 'UNKNOWN':
        fact.limitations.append('Consolidated versus separate reporting is unknown.')
    if fact.normalized_value is not None and not math.isfinite(fact.normalized_value):
        fact.normalized_value = None
        fact.limitations.append('Normalized value overflowed.')
    fact.status = 'NORMALIZED' if not fact.limitations else 'PARTIAL'
    return fact


def comparability_reason(left: CanonicalFinancialFact, right: CanonicalFinancialFact,
                         *, same_period: bool = False, same_metric: bool = True) -> str | None:
    """None means comparable; otherwise the exact reason the calculation is withheld."""
    if same_metric and left.metric != right.metric:
        return 'Different metrics.'
    if left.normalized_value is None or right.normalized_value is None:
        return 'Unknown or conflicting unit/currency/scale.'
    if (left.unit, left.currency) != (right.unit, right.currency):
        return 'Different units or currencies; no implicit FX conversion.'
    if left.period_end is None or right.period_end is None:
        return 'Unknown reporting period.'
    if same_period and (left.period_start, left.period_end) != (right.period_start, right.period_end):
        return 'Different reporting periods.'
    if left.reporting_basis == 'UNKNOWN' or right.reporting_basis == 'UNKNOWN':
        return 'Unknown standalone/YTD/annual accounting basis.'
    if left.reporting_basis != right.reporting_basis or left.frequency != right.frequency:
        return 'Different reporting bases or frequencies.'
    if left.consolidation == 'UNKNOWN' or right.consolidation == 'UNKNOWN':
        return 'Unknown consolidation basis.'
    if left.consolidation != right.consolidation:
        return 'Consolidated and separate statements must not be mixed.'
    if left.assurance != right.assurance:
        return 'Different assurance levels; do not silently compare audited and unaudited values.'
    if not same_period and left.period_start and right.period_start:
        duration_left = (left.period_end - left.period_start).days
        duration_right = (right.period_end - right.period_start).days
        if abs(duration_left - duration_right) > 3:
            return 'Different reporting period durations.'
    return None


def _visible(observation, as_of, fetched_at):
    return (observation.period_end is not None and observation.period_end <= as_of
        and (observation.published_at is None or observation.published_at <= fetched_at))


def derive_financial_metrics(observations, *, as_of: date, fetched_at: datetime):
    """Return (derived metrics, limitations), never score unverified statement data."""
    facts = [normalize_observation(item) for item in observations if _visible(item, as_of, fetched_at)]
    grouped = defaultdict(list)
    for fact in facts:
        grouped[(fact.metric, fact.period_end)].append(fact)
    # Conflicting restatements/sources are held, not resolved by list order.
    unique, limitations = [], []
    for (metric, period), values in grouped.items():
        signatures = {(v.normalized_value if v.normalized_value is not None else v.value,
            v.unit, v.currency, v.scale, v.frequency, v.reporting_basis, v.consolidation, v.assurance) for v in values}
        if len(signatures) != 1:
            limitations.append(f'{metric} {period}: conflicting observations; withheld.')
        else:
            unique.append(values[0])
    derived = []

    def add(metric, value, left, right, formula, note):
        sources = sorted({left.source_id, right.source_id})
        fields = [f'{item.metric}@{item.period_end}' for item in (left, right)]
        evidence = {'label': metric, 'value': f'{value:.8g}', 'input_fields': fields,
            'source_ids': sources, 'as_of': right.period_end.isoformat(), 'confidence': 1.0,
            'note': note}
        derived.append(DerivedFinancialMetric(metric=metric, value=value, period_end=right.period_end,
            previous_period_end=left.period_end if left.period_end != right.period_end else None,
            formula=formula, input_fields=fields, source_ids=sources, evidence=evidence, note=note))

    for metric in ('revenue', 'net_profit'):
        series = sorted([f for f in unique if f.metric == metric], key=lambda f: f.period_end)
        if len(series) < 2:
            continue
        latest = series[-1]
        previous = next((f for f in reversed(series[:-1]) if f.period_end.year == latest.period_end.year - 1
            and (f.period_end.month, f.period_end.day) == (latest.period_end.month, latest.period_end.day)), None)
        if previous is None:
            limitations.append(f'{metric}: no matching prior-year reporting period; YoY withheld.')
            continue
        reason = comparability_reason(previous, latest)
        if reason:
            limitations.append(f'{metric}: {reason}')
        elif previous.normalized_value <= 0:
            limitations.append(f'{metric}: non-positive prior value; percentage growth would be misleading.')
        else:
            value = latest.normalized_value / previous.normalized_value - 1
            add('revenue_growth' if metric == 'revenue' else 'profit_growth', value, previous, latest,
                'current / prior_year - 1', 'Year-on-year, same reporting basis, consolidation, unit and duration.')
    profits = sorted([f for f in unique if f.metric == 'net_profit'], key=lambda f: f.period_end, reverse=True)
    if profits:
        profit = profits[0]
        cash = next((f for f in unique if f.metric == 'operating_cash_flow' and f.period_end == profit.period_end), None)
        if cash:
            reason = comparability_reason(profit, cash, same_period=True, same_metric=False)
            if reason:
                limitations.append(f'cash_conversion: {reason}')
            elif profit.normalized_value <= 0:
                limitations.append('cash_conversion: non-positive net profit; ratio withheld.')
            else:
                add('cash_conversion', cash.normalized_value / profit.normalized_value, profit, cash,
                    'operating_cash_flow / net_profit', 'Same period and accounting basis; not a standalone cash-quality verdict.')
    return derived, list(dict.fromkeys(limitations))


def build_financial_evidence(data) -> dict:
    observations = [*data.financial_statement_observations, *data.financial_observations]
    normalized = [normalize_observation(item) for item in observations if _visible(item, data.as_of, data.fetched_at)]
    derived, limitations = derive_financial_metrics(data.financial_statement_observations,
        as_of=data.as_of, fetched_at=data.fetched_at)
    limitations += [f'{fact.metric} {fact.period_label}: {note}' for fact in normalized for note in fact.limitations]
    return {'schema_version': 'canonical-financial-v1',
        'normalized_facts': [fact.model_dump(mode='json') for fact in normalized],
        'derived_metrics': [item.model_dump(mode='json') for item in derived],
        'limitations': list(dict.fromkeys(limitations)),
        'policy': 'No guessing scale, FX, accounting basis or missing periods; no causal claim from growth alone.'}


class ResearchFinancialFactProposal(BaseModel):
    """Explicit, typed source claim. Acceptance validates extraction, not issuer truth."""
    model_config = ConfigDict(extra='forbid')
    ticker: str = Field(pattern=r'^[A-Z][A-Z0-9]{1,9}$')
    statement: Literal['income_statement', 'balance_sheet', 'cash_flow']
    metric: Literal['revenue', 'net_profit', 'operating_cash_flow', 'total_assets', 'total_equity', 'total_liabilities']
    value: float = Field(allow_inf_nan=False)
    unit: str
    period_label: str
    period_end: date
    period_start: date | None = None
    currency: str
    scale: float = Field(gt=0, allow_inf_nan=False)
    reporting_basis: Literal['STANDALONE', 'YTD', 'ANNUAL', 'POINT_IN_TIME', 'TTM']
    consolidation: Literal['CONSOLIDATED', 'SEPARATE']
    source_id: str
    evidence_quote: str = Field(min_length=10)

    @model_validator(mode='after')
    def valid_period(self):
        if self.period_start and self.period_start > self.period_end:
            raise ValueError('period_start must not follow period_end')
        return self


def validate_research_fact(proposal: ResearchFinancialFactProposal, citation) -> FinancialStatementObservation:
    """Only accept exact quoted figures with period/unit declarations from same chunk.

    Conservative rejection is deliberate: proposals are not interpreted as proof
    of an audited fact, and are never automatically added to provider observations.
    """
    if proposal.source_id != citation.chunk_id or proposal.evidence_quote not in citation.quote:
        raise ValueError('Exact quote and matching citation chunk are required.')
    quote = proposal.evidence_quote.casefold()
    if proposal.ticker not in getattr(citation, 'tickers', []):
        raise ValueError('Issuer ticker is not declared by the citation metadata.')
    if len(set(citation.tickers)) > 1 and not re.search(r'\b' + re.escape(proposal.ticker.casefold()) + r'\b', quote):
        raise ValueError('A multi-issuer citation must identify the issuer inside the exact quote.')
    metric_words = {'revenue': ('doanh thu', 'revenue', 'net sales'), 'net_profit': ('lợi nhuận sau thuế', 'net profit', 'profit after tax'),
        'operating_cash_flow': ('dòng tiền kinh doanh', 'lưu chuyển tiền từ hoạt động kinh doanh', 'operating cash flow'),
        'total_assets': ('tổng tài sản', 'total assets'), 'total_equity': ('vốn chủ sở hữu', 'total equity'),
        'total_liabilities': ('tổng nợ phải trả', 'total liabilities')}[proposal.metric]
    if not any(word in quote for word in metric_words):
        raise ValueError('Metric identity is not explicit in the exact quote.')
    amount_pattern = r'([+-]?\d+(?:[.,]\d+)?)'
    labelled_values = [match.group(1) for word in metric_words for match in
        re.finditer(re.escape(word) + r'\s*[:=]?\s*' + amount_pattern + r'\s*' +
            re.escape(proposal.unit.casefold()) + r'(?!\w)', quote)]
    if not any(float(value.replace(',', '.')) == proposal.value for value in labelled_values):
        raise ValueError('The value and unit must immediately follow their exact metric label; complex table mapping requires review.')
    numbers = re.findall(r'(?<!\w)[+-]?\d+(?:[.,]\d+)?(?!\w)', quote)
    if not any(float(token.replace(',', '.')) == proposal.value for token in numbers):
        raise ValueError('Exact numeric value is not present in the quote; ambiguous grouped numbers are not accepted.')
    if str(proposal.period_end.year) not in quote or proposal.unit.casefold() not in quote:
        raise ValueError('The quote must explicitly contain the period year and unit.')
    declared_unit = _CURRENCY_UNITS.get(proposal.unit.strip().casefold())
    if declared_unit != (proposal.currency.upper(), proposal.scale):
        raise ValueError('Currency and scale must exactly match the explicit quoted unit.')
    end = proposal.period_end
    from stock_multi_agent.services.reporting_period import infer_reporting_period
    detected = infer_reporting_period(proposal.evidence_quote, proposal.evidence_quote)
    if detected.status == 'AMBIGUOUS' or detected.end and detected.end != end:
        raise ValueError('The exact quote has ambiguous or inconsistent reporting periods.')
    explicit_end = any(token in quote for token in (end.isoformat(), end.strftime('%d/%m/%Y'), end.strftime('%d-%m-%Y')))
    quarter = (end.month - 1) // 3 + 1
    quarter_match = re.search(rf'\b(?:q\s*{quarter}|quý\s*{quarter}|quarter\s*{quarter})\b', quote)
    annual_end = proposal.reporting_basis == 'ANNUAL' and (end.month, end.day) == (12, 31)
    if not (explicit_end or annual_end or quarter_match and proposal.reporting_basis in {'STANDALONE', 'YTD'}
            and end.month in (3, 6, 9, 12) and end.day == calendar.monthrange(end.year, end.month)[1]):
        raise ValueError('Exact reporting period end or matching quarter must appear in the quote.')
    if citation.published_at and end > citation.published_at.date():
        raise ValueError('A reported fact cannot end after its publication date.')
    consolidation = ('hợp nhất', 'consolidated') if proposal.consolidation == 'CONSOLIDATED' else ('riêng lẻ', 'separate')
    if not any(word in quote for word in consolidation):
        raise ValueError('Consolidation declaration is missing from exact evidence.')
    basis_words = {'ANNUAL': ('cả năm', 'annual', 'year ended'), 'YTD': ('lũy kế', 'luỹ kế', 'year to date', 'ytd'),
        'STANDALONE': ('quý riêng', 'standalone quarter'), 'POINT_IN_TIME': ('tại ngày', 'as of'),
        'TTM': ('ttm', 'trailing twelve months')}[proposal.reporting_basis]
    if not any(word in quote for word in basis_words):
        raise ValueError('Accounting basis is missing from exact evidence.')
    observation = FinancialStatementObservation(**proposal.model_dump(exclude={'ticker'}), frequency='annual' if proposal.reporting_basis == 'ANNUAL' else 'quarter',
        source_url=str(citation.source_url), published_at=citation.published_at)
    if normalize_observation(observation).normalized_value is None:
        raise ValueError('Proposal unit/currency/scale cannot be normalized.')
    return observation
