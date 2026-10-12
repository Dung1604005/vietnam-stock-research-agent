from __future__ import annotations

from stock_multi_agent.agents.base_agent import BaseAgent
from stock_multi_agent.models.agent_contracts import AgentSpec, FundamentalAgentContext
from stock_multi_agent.models.analysis import EvidenceItem, FundamentalAnalysis, Signal
from stock_multi_agent.models.messages import AgentName, MessageType
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.tools.financial_data import FinancialToolkit


class FundamentalAgent(BaseAgent):
    name = AgentName.FUNDAMENTAL
    goal = "Assess business quality and valuation independently from price trends."
    spec = AgentSpec(
        role=name,
        goal=goal,
        required_context=["market_data.financials", "financial_statements", "research_evidence"],
        allowed_tools=["financial_metric_normalization", "sector_template", "quality_weighting"],
        output_type="FundamentalAnalysis",
        can_use_llm=False,
        decision_constraints=[
            "Never infer a missing financial value from prose",
            "Reduce confidence when evidence is stale or low quality",
            "Research citations provide context but do not directly change the numerical score",
        ],
    )

    def __init__(self, toolkit: FinancialToolkit, llm_advisor=None) -> None:
        self.tools = toolkit
        self.llm_advisor = llm_advisor

    def run(self, state: AgentState) -> dict[str, object]:
        context = FundamentalAgentContext.from_state(state)
        metrics = context.market_data.financials
        research_context = self._research_context(state)
        evidence_reliability, quality_notes = self._financial_reliability(state)
        statement_metrics = self._statement_metrics(state)
        stale_ratio_metrics = any(
            issue.code == "STALE_FINANCIAL_RATIOS"
            for issue in state.market_data.quality_report.issues
        )
        if stale_ratio_metrics:
            # Current, period-labelled statements take precedence for metrics
            # they can calculate. Other stale ratios remain usable at reduced weight.
            metrics = metrics.model_copy(update={
                **({"revenue_growth": None} if statement_metrics.get("revenue_growth") is not None else {}),
                **({"profit_growth": None} if statement_metrics.get("profit_growth") is not None else {}),
            })
        available = metrics.available_count()
        statement_coverage = sum(
            statement_metrics.get(name) is not None
            for name in ("revenue_growth", "profit_growth", "cash_conversion")
        )
        traces = [self.event("OBSERVE", "READ_FINANCIALS", f"Observe {available}/10 normalized financial metrics")]
        if available == 0 and statement_coverage == 0:
            result = FundamentalAnalysis(
                signal=Signal.UNKNOWN, score=5, confidence=0.1,
                strengths=[], weaknesses=["No usable financial metric or multi-period statement evidence is available"],
                evidence=[self._evidence(state, "Metric coverage", "No usable financial metric is available", ["financials"])],
                reasoning_summary="Fundamental evidence is absent; no valuation conclusion is made.",
                sector_template=self._sector_template(state),
                policy_version="fundamental-v3-quality-weighted",
                research_context_count=len(research_context),
            )
            result.evidence.extend(research_context)
        else:
            score = 5.0
            template = self._sector_template(state)
            thresholds = self._thresholds(template)
            strengths: list[str] = []
            weaknesses: list[str] = []
            evidence: list[EvidenceItem] = []
            evidence.extend(research_context)
            contributions: dict[str, float] = {}

            def assess(value: float | None, good: float, bad: float, label: str) -> None:
                nonlocal score
                if value is None:
                    return
                evidence.append(self._evidence(state, label, self.tools.format_percent(value), [label.casefold().replace(" ", "_")]))
                if value >= good:
                    score += 0.7
                    contributions[label] = 0.7
                    strengths.append(f"{label} is strong")
                elif value < bad:
                    score -= 0.7
                    contributions[label] = -0.7
                    weaknesses.append(f"{label} is weak/negative")
                else:
                    contributions[label] = 0.0

            assess(metrics.revenue_growth, *thresholds["revenue_growth"], "Revenue growth")
            assess(metrics.profit_growth, *thresholds["profit_growth"], "Profit growth")
            assess(metrics.roe, *thresholds["roe"], "ROE")
            assess(metrics.roa, *thresholds["roa"], "ROA")
            if template not in {"BANK", "SECURITIES", "INSURANCE"}:
                assess(metrics.profit_margin, *thresholds["profit_margin"], "Profit margin")
            if metrics.pe is not None:
                evidence.append(self._evidence(state, "P/E", f"{metrics.pe:.2f}", ["financials.pe"]))
                if 0 < metrics.pe <= thresholds["pe_good"]:
                    score += 0.7
                    contributions["P/E"] = 0.7
                    strengths.append("P/E is moderate")
                elif metrics.pe > thresholds["pe_bad"] or metrics.pe <= 0:
                    score -= 0.8
                    contributions["P/E"] = -0.8
                    weaknesses.append("P/E suggests valuation or earnings risk")
                else:
                    contributions["P/E"] = 0.0
            if metrics.pb is not None:
                evidence.append(self._evidence(state, "P/B", f"{metrics.pb:.2f}", ["financials.pb"]))
                if metrics.pb > thresholds["pb_bad"]:
                    score -= 0.4
                    contributions["P/B"] = -0.4
                    weaknesses.append("P/B is relatively high")
                else:
                    contributions["P/B"] = 0.0
            if metrics.debt_to_equity is not None and template not in {"BANK", "SECURITIES", "INSURANCE"}:
                evidence.append(self._evidence(state, "Debt/equity", f"{metrics.debt_to_equity:.2f}", ["financials.debt_to_equity"]))
                if metrics.debt_to_equity > thresholds["debt_to_equity_bad"]:
                    score -= 0.8
                    contributions["Debt/equity"] = -0.8
                    weaknesses.append("Leverage is high")
                else:
                    contributions["Debt/equity"] = 0.0
            if metrics.operating_cash_flow is not None:
                evidence.append(self._evidence(state, "Operating cash flow", f"{metrics.operating_cash_flow:,.0f}", ["financials.operating_cash_flow"]))
                if metrics.operating_cash_flow > 0:
                    score += 0.4
                    contributions["Operating cash flow"] = 0.4
                    strengths.append("Operating cash flow is positive")
                else:
                    score -= 0.6
                    contributions["Operating cash flow"] = -0.6
                    weaknesses.append("Operating cash flow is non-positive")
            if metrics.eps is not None:
                evidence.append(self._evidence(state, "EPS", f"{metrics.eps:,.2f}", ["financials.eps"]))

            # Prefer the normalised ratio when available, then fall back to a
            # directly reported two-period statement calculation. This avoids
            # inventing a growth rate when only one reporting period exists.
            if metrics.revenue_growth is None and statement_metrics.get("revenue_growth") is not None:
                value = statement_metrics["revenue_growth"]
                evidence.append(self._evidence(state, "Reported revenue growth", self.tools.format_percent(value), ["financial_statement_observations"]))
                contributions["reported_revenue_growth"] = 0.7 if value >= thresholds["revenue_growth"][0] else -0.7 if value < thresholds["revenue_growth"][1] else 0.0
                score += contributions["reported_revenue_growth"]
            if metrics.profit_growth is None and statement_metrics.get("profit_growth") is not None:
                value = statement_metrics["profit_growth"]
                evidence.append(self._evidence(state, "Reported profit growth", self.tools.format_percent(value), ["financial_statement_observations"]))
                contributions["reported_profit_growth"] = 0.7 if value >= thresholds["profit_growth"][0] else -0.7 if value < thresholds["profit_growth"][1] else 0.0
                score += contributions["reported_profit_growth"]
            if statement_metrics.get("cash_conversion") is not None:
                value = statement_metrics["cash_conversion"]
                evidence.append(self._evidence(state, "Operating cash conversion", f"{value:.2f}x", ["financial_statement_observations"]))
                contributions["cash_conversion"] = 0.4 if value >= 0.8 else -0.6 if value < 0 else 0.0
                score += contributions["cash_conversion"]

            raw_score = max(0.0, min(10.0, score))
            quality_penalty = 0.75 * (1.0 - evidence_reliability)
            score = 5.0 + (raw_score - 5.0) * evidence_reliability - quality_penalty
            score = max(0.0, min(10.0, score))
            if quality_notes:
                weaknesses.extend(quality_notes)
                contributions["data_quality_penalty"] = round(-quality_penalty, 3)
            signal = Signal.BUY if score >= 7.2 else Signal.HOLD if score >= 4.2 else Signal.SELL
            coverage_confidence = min(0.9, 0.35 + self.tools.completeness(metrics) * 0.6 + 0.05 * statement_coverage)
            confidence = max(0.1, coverage_confidence * evidence_reliability)
            result = FundamentalAnalysis(
                signal=signal, score=round(score, 2), confidence=round(confidence, 2),
                strengths=strengths, weaknesses=weaknesses, evidence=evidence,
                reasoning_summary=(
                    f"Available growth, quality, valuation, leverage and cash-flow evidence imply {signal.value}; "
                    f"quality-weighted score {score:.1f}/10 (raw {raw_score:.1f}, evidence reliability {evidence_reliability:.0%})."
                ),
                sector_template=template,
                policy_version="fundamental-v3-quality-weighted",
                feature_contributions=contributions,
                unavailable_metrics=[
                    name for name, value in metrics.model_dump().items() if value is None
                ] + [
                    *([] if statement_metrics.get("cash_conversion") is not None else ["cash conversion ratio"]),
                    "TTM statement series", "valuation history", "dilution/dividend history"
                ],
                research_context_count=len(research_context),
            )
            traces.append(self.event(
                "REASON", "SCORE_FUNDAMENTALS",
                f"Quality-weighted fundamental score={score:.1f}/10; raw={raw_score:.1f}; reliability={evidence_reliability:.0%}",
            ))

        source_ids = [
            *[source.source_id for source in state.market_data.source_refs],
            *[citation.chunk_id for citation in state.research_evidence],
        ]
        llm_review = self.llm_review(
            "Interpret business quality, valuation tensions, and missing fundamental evidence",
            {
                "ticker": state.ticker,
                "horizon": state.investment_horizon,
                "deterministic_result": result.model_dump(mode="json"),
                "quality_status": state.market_data.quality_report.status.value,
                "data_warnings": state.market_data.warnings,
            },
            source_ids,
        )
        if llm_review is not None:
            traces.append(self.event(
                "REASON", "LLM_FUNDAMENTAL_REVIEW", f"Structured LLM review={llm_review.status}", "local_llm",
            ))
        traces.append(self.event("ACT", "SEND_RESULT", f"Fundamental signal={result.signal.value}, confidence={result.confidence:.2f}"))
        return {
            "fundamental_result": result,
            "messages": [self.message(AgentName.MANAGER, MessageType.RESULT, "Fundamental analysis complete", result.model_dump(mode="json"), result.confidence)],
            "llm_reviews": [llm_review] if llm_review is not None else [],
            "trace": traces,
            "completed_tasks": ["fundamental_analysis"],
            "agent_reports": [self.report(
                stage="fundamental_analysis",
                status="UNAVAILABLE" if result.signal is Signal.UNKNOWN else "COMPLETED",
                summary=result.reasoning_summary,
                confidence=result.confidence,
                output_key="fundamental_result",
                tools_used=["sector_template", "quality_weighting", "financial_metric_normalization"],
                assumptions=[f"Sector template={result.sector_template}"],
                missing_information=result.unavailable_metrics,
                evidence_count=len(result.evidence),
            )],
        }

    @staticmethod
    def _research_context(state: AgentState) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        for citation in state.research_evidence[:3]:
            quote = citation.quote if len(citation.quote) <= 500 else citation.quote[:497] + "..."
            evidence.append(EvidenceItem(
                label=f"Research context: {citation.title}"[:160],
                value=quote,
                input_fields=["research_evidence"],
                source_ids=[citation.document_id, citation.chunk_id],
                as_of=citation.published_at.date(),
                confidence=min(0.75, citation.retrieval_score),
                note=f"Publisher: {citation.publisher}; contextual evidence only, excluded from numerical score",
            ))
        return evidence

    @staticmethod
    def _sector_template(state: AgentState) -> str:
        sector = state.market_data.instrument.sector if state.market_data and state.market_data.instrument else None
        if not sector:
            return "NON_FINANCIAL_GENERIC_UNVERIFIED_SECTOR"
        normalized = sector.casefold()
        if "bank" in normalized or "ngân hàng" in normalized:
            return "BANK"
        if "securit" in normalized or "chứng khoán" in normalized:
            return "SECURITIES"
        if "insur" in normalized or "bảo hiểm" in normalized:
            return "INSURANCE"
        if "real estate" in normalized or "bất động sản" in normalized:
            return "REAL_ESTATE"
        return "NON_FINANCIAL_GENERIC"

    @staticmethod
    def _thresholds(template: str) -> dict[str, tuple[float, float] | float]:
        """Small explicit sector templates; no peer percentile is invented."""
        generic: dict[str, tuple[float, float] | float] = {
            "revenue_growth": (0.10, 0.0), "profit_growth": (0.10, 0.0),
            "roe": (0.15, 0.08), "roa": (0.08, 0.03), "profit_margin": (0.12, 0.05),
            "pe_good": 18.0, "pe_bad": 30.0, "pb_bad": 5.0, "debt_to_equity_bad": 1.5,
        }
        if template == "BANK":
            return {**generic, "roe": (0.15, 0.08), "roa": (0.01, 0.004), "pe_good": 12.0, "pe_bad": 20.0, "pb_bad": 2.5}
        if template == "SECURITIES":
            return {**generic, "roe": (0.12, 0.06), "roa": (0.04, 0.015), "pe_good": 15.0, "pe_bad": 25.0, "pb_bad": 3.0}
        if template == "INSURANCE":
            return {**generic, "roe": (0.12, 0.06), "roa": (0.025, 0.01), "pe_good": 15.0, "pe_bad": 25.0, "pb_bad": 2.5}
        if template == "REAL_ESTATE":
            return {**generic, "roe": (0.12, 0.06), "roa": (0.04, 0.015), "profit_margin": (0.18, 0.08), "pe_good": 15.0, "pe_bad": 25.0, "pb_bad": 3.0, "debt_to_equity_bad": 2.0}
        return generic

    @staticmethod
    def _statement_metrics(state: AgentState) -> dict[str, float | None]:
        """Calculate only same-unit, period-labelled reported values."""
        if state.market_data is None:
            return {}
        grouped: dict[str, list[tuple[object, float]]] = {}
        for observation in state.market_data.financial_statement_observations:
            if observation.period_end is None or observation.frequency != "annual":
                continue
            grouped.setdefault(observation.metric, []).append((observation.period_end, observation.value))
        values: dict[str, float | None] = {}
        for metric, series in grouped.items():
            ordered = sorted(series, key=lambda item: item[0])
            if len(ordered) >= 2 and ordered[-2][1] != 0:
                values[f"{metric}_growth"] = ordered[-1][1] / ordered[-2][1] - 1
        net_profit = grouped.get("net_profit", [])
        operating_cash_flow = grouped.get("operating_cash_flow", [])
        if net_profit and operating_cash_flow:
            latest_profit_period, latest_profit = max(net_profit, key=lambda item: item[0])
            latest_cash_period, latest_cash = max(operating_cash_flow, key=lambda item: item[0])
            if latest_profit_period == latest_cash_period and latest_profit != 0:
                values["cash_conversion"] = latest_cash / latest_profit
        return values

    @staticmethod
    def _evidence(state: AgentState, label: str, value: str, fields: list[str]) -> EvidenceItem:
        data = state.market_data
        confidence, quality_notes = FundamentalAgent._financial_reliability(state)
        financial_periods = [
            observation.period_end
            for observation in [
                *(data.financial_observations if data else []),
                *(data.financial_statement_observations if data else []),
            ]
            if observation.period_end is not None
        ]
        observation_source_ids = {
            observation.source_id
            for observation in [
                *(data.financial_observations if data else []),
                *(data.financial_statement_observations if data else []),
            ]
        }
        return EvidenceItem(
            label=label,
            value=value,
            input_fields=fields,
            source_ids=sorted(observation_source_ids | ({
                source.source_id for source in data.source_refs
                if source.data_kind.value in {"FINANCIAL_RATIOS", "INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"}
            } if data else set())),
            as_of=max(financial_periods) if financial_periods else None,
            confidence=confidence,
            note="; ".join(quality_notes) if quality_notes else None,
        )

    @staticmethod
    def _financial_reliability(state: AgentState) -> tuple[float, list[str]]:
        """Map declared data-quality limitations to evidence weight without inventing values."""
        if state.market_data is None:
            return 0.1, ["Financial data is unavailable"]
        reliability_by_code = {
            "MISSING_SOURCE_REF": 0.35,
            "STALE_FINANCIALS": 0.50,
            "MISSING_FINANCIAL_PERIOD": 0.55,
            "INSUFFICIENT_FINANCIALS": 0.60,
            "STALE_FINANCIAL_RATIOS": 0.65,
        }
        relevant_issues = [
            issue for issue in state.market_data.quality_report.issues
            if issue.code in reliability_by_code
        ]
        reliability = min(
            [1.0, *(reliability_by_code[issue.code] for issue in relevant_issues)],
        )
        notes = [f"Lower-quality evidence ({issue.code}): {issue.message}" for issue in relevant_issues]
        return reliability, notes
