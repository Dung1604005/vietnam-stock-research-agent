from __future__ import annotations

from stock_multi_agent.models.analysis import FinancialMetrics


class FinancialToolkit:
    """Financial metric capabilities without recommendation policy."""

    @staticmethod
    def completeness(metrics: FinancialMetrics) -> float:
        return metrics.available_count() / len(type(metrics).model_fields)

    @staticmethod
    def format_percent(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.1f}%"

    @staticmethod
    def format_number(value: float | None) -> str:
        return "N/A" if value is None else f"{value:,.2f}"
