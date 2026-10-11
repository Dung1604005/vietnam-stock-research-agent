from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from stock_multi_agent.models.analysis import MarketData
from stock_multi_agent.models.data import (
    DataQualityIssue,
    DataQualityReport,
    PriceAdjustment,
    QualitySeverity,
    QualityStatus,
)


class DataQualityChecker:
    """Deterministic fail-closed checks for normalized market/company data."""

    def evaluate(
        self,
        data: MarketData,
        require_financials: bool,
        today: date | None = None,
    ) -> DataQualityReport:
        today = today or date.today()
        issues: list[DataQualityIssue] = []
        checks = [
            "minimum_history", "chronology", "freshness", "business_day_gaps",
            "extreme_returns", "volume_coverage", "source_provenance",
            "price_adjustment", "financial_periods",
        ]
        if len(data.bars) < 50:
            issues.append(self._issue("INSUFFICIENT_HISTORY", QualitySeverity.ERROR, f"Only {len(data.bars)} valid sessions", "bars", len(data.bars)))
        dates = [bar.date for bar in data.bars]
        if dates != sorted(dates) or len(dates) != len(set(dates)):
            issues.append(self._issue("INVALID_CHRONOLOGY", QualitySeverity.ERROR, "Price dates are not unique and ascending", "bars.date"))
        if data.as_of and data.as_of > today:
            issues.append(self._issue("FUTURE_AS_OF", QualitySeverity.ERROR, "Data as-of date is in the future", "as_of", data.as_of.isoformat()))
        stale_days = self._business_days_between(data.bars[-1].date, today)
        if stale_days > 3:
            issues.append(self._issue("STALE_PRICE", QualitySeverity.WARNING, f"Latest price is {stale_days} business days old", "bars.date", data.bars[-1].date.isoformat()))

        large_gaps = 0
        for previous, current in zip(data.bars, data.bars[1:]):
            if self._business_days_between(previous.date, current.date) > 5:
                large_gaps += 1
        if large_gaps:
            issues.append(self._issue("PRICE_GAPS", QualitySeverity.WARNING, f"Found {large_gaps} gaps longer than five business days", "bars.date", large_gaps))

        extreme_returns = 0
        for previous, current in zip(data.bars, data.bars[1:]):
            if abs(current.close / previous.close - 1) > 0.5:
                extreme_returns += 1
        if extreme_returns:
            issues.append(self._issue("EXTREME_RETURNS", QualitySeverity.WARNING, f"Found {extreme_returns} moves above 50%; corporate actions may be unaccounted for", "bars.close", extreme_returns))

        zero_volume_ratio = sum(bar.volume == 0 for bar in data.bars) / len(data.bars)
        if zero_volume_ratio > 0.2:
            issues.append(self._issue("LOW_VOLUME_COVERAGE", QualitySeverity.WARNING, "More than 20% of sessions have zero volume", "bars.volume", round(zero_volume_ratio, 4)))
        if not data.source_refs:
            issues.append(self._issue("MISSING_SOURCE_REF", QualitySeverity.ERROR, "No structured source reference is attached", "source_refs"))
        if data.price_adjustment is PriceAdjustment.UNKNOWN:
            issues.append(self._issue("UNKNOWN_PRICE_ADJUSTMENT", QualitySeverity.WARNING, "Provider does not declare whether prices are corporate-action adjusted", "price_adjustment"))

        available = data.financials.available_count()
        if require_financials and available < 3:
            issues.append(self._issue("INSUFFICIENT_FINANCIALS", QualitySeverity.WARNING, f"Only {available}/10 financial metrics are available", "financials", available))
        has_statement_periods = any(item.period_end is not None for item in data.financial_statement_observations)
        if available and not data.financial_observations and not has_statement_periods:
            issues.append(self._issue("MISSING_FINANCIAL_PERIOD", QualitySeverity.WARNING, "Financial metrics have no reporting-period observations", "financial_observations"))
        period_dates = [item.period_end for item in data.financial_observations if item.period_end]
        statement_period_dates = [item.period_end for item in data.financial_statement_observations if item.period_end]
        latest_ratio_period = max(period_dates) if period_dates else None
        latest_statement_period = max(statement_period_dates) if statement_period_dates else None
        latest_financial_period = max([item for item in (latest_ratio_period, latest_statement_period) if item is not None], default=None)
        if require_financials and latest_financial_period is not None and (today - latest_financial_period).days > 550:
            issues.append(self._issue(
                "STALE_FINANCIALS",
                QualitySeverity.WARNING,
                f"Latest financial reporting period is stale ({latest_financial_period.isoformat()})",
                "financial_periods",
                latest_financial_period.isoformat(),
            ))
        elif require_financials and latest_ratio_period is not None and (today - latest_ratio_period).days > 550:
                issues.append(self._issue(
                    "STALE_FINANCIAL_RATIOS",
                    QualitySeverity.WARNING,
                    f"Latest ratio reporting period is stale ({latest_ratio_period.isoformat()}); current statement observations remain available",
                    "financial_observations.period_end",
                    latest_ratio_period.isoformat(),
                ))

        status = QualityStatus.FAIL if any(item.severity is QualitySeverity.ERROR for item in issues) else (
            QualityStatus.WARN if any(item.severity is QualitySeverity.WARNING for item in issues) else QualityStatus.PASS
        )
        return DataQualityReport(
            status=status,
            checked_at=datetime.now(timezone.utc),
            checks_run=checks,
            issues=issues,
        )

    @staticmethod
    def _issue(code, severity, message, field=None, observed=None) -> DataQualityIssue:
        return DataQualityIssue(code=code, severity=severity, message=message, field=field, observed=observed)

    @staticmethod
    def _business_days_between(start: date, end: date) -> int:
        if end <= start:
            return 0
        current = start + timedelta(days=1)
        count = 0
        while current <= end:
            if current.weekday() < 5:
                count += 1
            current += timedelta(days=1)
        return count
