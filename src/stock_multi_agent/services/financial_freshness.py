"""Scoring eligibility is separate from preserving raw historical observations."""
from datetime import date
from dataclasses import dataclass
import math

from stock_multi_agent.models.analysis import FinancialMetrics, MarketData

MAX_FINANCIAL_AGE_DAYS = 550


def financial_age_weight(age_days: int) -> float:
    """Evidence age affects confidence, never changes the reported period/value."""
    if age_days <= MAX_FINANCIAL_AGE_DAYS:
        return 1.0
    if age_days <= 3 * 365:
        return 0.65
    if age_days <= 5 * 365:
        return 0.45
    return 0.25


@dataclass
class FundamentalInputs:
    metrics: FinancialMetrics
    statements: dict[str, float]
    metric_periods: dict[str, date]
    fallback_metrics: list[str]
    notes: list[str]
    age_weight: float


def fundamental_inputs(data: MarketData) -> FundamentalInputs:
    """Latest usable observation per metric, with an explicit historical fallback.

    Risk/current scoring retains its strict freshness filter below. Fundamental
    can assess older evidence, but cannot turn unknown units or dates into facts.
    """
    reference = data.as_of or data.bars[-1].date
    requires_period = bool(data.financial_observations or any(
        source.data_kind.value == 'FINANCIAL_RATIOS' for source in data.source_refs))
    values = {} if requires_period else data.financials.model_dump()
    periods, notes = {}, []
    for metric in FinancialMetrics.model_fields:
        observations = [item for item in data.financial_observations
            if item.metric == metric and item.period_end and item.period_end <= reference
            and (item.published_at is None or item.published_at <= data.fetched_at)]
        # An unusable latest period may fall back to an earlier verified one.
        for period in sorted({item.period_end for item in observations}, reverse=True):
            candidates = [item for item in observations if item.period_end == period]
            signatures = {(item.value, item.unit, item.currency, item.scale,
                item.frequency, item.reporting_basis, item.consolidation) for item in candidates}
            if len(signatures) != 1:
                notes.append(f'{metric} {period}: conflicting observations; withheld.')
                continue
            item = candidates[0]
            # Provider ratios are already normalized to decimal/multiple. In
            # particular, unknown_ratio must never be rescued from raw history.
            expected_unit = ('multiple' if metric in {'pe', 'pb'} else
                'currency_per_share' if metric == 'eps' else
                'currency' if metric == 'operating_cash_flow' else 'ratio')
            if item.unit != expected_unit or not math.isfinite(item.value):
                notes.append(f'{metric} {period}: unverified unit/scale; withheld.')
                continue
            values[metric], periods[metric] = item.value, period
            break
    values = {name: value if value is None or math.isfinite(value) else None
              for name, value in values.items()}

    from stock_multi_agent.services.financial_facts import build_financial_evidence
    statements = {}
    for item in build_financial_evidence(data)['derived_metrics']:
        metric, period = item['metric'], date.fromisoformat(item['period_end'])
        # A newer statement calculation wins over an older reported ratio.
        if values.get(metric) is not None and period <= periods.get(metric, date.min):
            continue
        statements[metric] = item['value']
        values[metric] = None if metric in FinancialMetrics.model_fields else values.get(metric)
        periods[metric] = period
    metrics = FinancialMetrics(**{key: value for key, value in values.items()
                                if key in FinancialMetrics.model_fields})
    fallback = sorted(metric for metric, period in periods.items()
        if (reference - period).days > MAX_FINANCIAL_AGE_DAYS)
    notes.extend(f'{metric}: using nearest available reporting period {periods[metric]} '
        f'({(reference - periods[metric]).days} days old); historical fallback, reduced confidence.'
        for metric in fallback)
    return FundamentalInputs(metrics, statements, periods, fallback, notes,
        min((financial_age_weight((reference - period).days) for period in periods.values()), default=1.0))


def scoring_financials(data):
    values = data.financials.model_dump()
    notes = []
    reference = data.as_of or data.bars[-1].date
    observed = data.financial_observations
    requires_period = bool(observed or any(s.data_kind.value == 'FINANCIAL_RATIOS' for s in data.source_refs))
    for metric, value in values.items():
        if value is None:
            continue
        matches = [o for o in observed if o.metric == metric and abs(o.value - value) <= 1e-8]
        eligible = [o for o in matches if o.period_end and 0 <= (reference - o.period_end).days <= MAX_FINANCIAL_AGE_DAYS]
        if requires_period and not eligible:
            periods = ', '.join(sorted({str(o.period_end) for o in matches})) or 'unknown'
            values[metric] = None
            notes.append(f'{metric}: reporting period {periods} is stale, future or unverified; excluded from current scoring.')
    return data.financials.model_copy(update=values), notes


def current_statement_metrics(data):
    from stock_multi_agent.services.financial_facts import build_financial_evidence
    reference = data.as_of or data.bars[-1].date
    return {item['metric']: item['value'] for item in build_financial_evidence(data)['derived_metrics']
            if item.get('period_end') and 0 <= (reference - date.fromisoformat(str(item['period_end']))).days <= MAX_FINANCIAL_AGE_DAYS}
