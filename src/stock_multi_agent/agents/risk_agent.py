from __future__ import annotations

from stock_multi_agent.agents.base_agent import BaseAgent
import os

from stock_multi_agent.models.analysis import EvidenceItem, RiskAnalysis, RiskLevel, RiskSection, Signal
from stock_multi_agent.models.data import DataKind
from stock_multi_agent.models.messages import AgentName, MessageType
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.tools.technical_indicators import TechnicalIndicatorToolkit


class RiskAgent(BaseAgent):
    name = AgentName.RISK
    goal = "Challenge optimistic conclusions and prevent unjustified confidence."

    def __init__(self, toolkit: TechnicalIndicatorToolkit, llm_advisor=None) -> None:
        self.tools = toolkit
        self.llm_advisor = llm_advisor

    def run(self, state: AgentState) -> dict[str, object]:
        if state.market_data is None or state.technical_result is None or state.fundamental_result is None:
            raise ValueError("Risk Agent requires data and both specialist results")
        closes = [bar.close for bar in state.market_data.bars]
        volumes = [bar.volume for bar in state.market_data.bars]
        volatility = self.tools.annualized_volatility(closes[-60:])
        drawdown = self.tools.max_drawdown(closes[-252:])
        average_traded_value = self.tools.average_traded_value(closes, volumes)
        technical = state.technical_result
        fundamental = state.fundamental_result
        score = 1.0
        criticisms: list[str] = []
        warnings = list(state.market_data.warnings)

        if volatility > 0.5:
            score += 3.0
            criticisms.append(f"Annualized recent volatility is very high ({volatility:.1%}).")
        elif volatility > 0.35:
            score += 2.0
            criticisms.append(f"Annualized recent volatility is elevated ({volatility:.1%}).")
        elif volatility > 0.25:
            score += 1.0
        if drawdown > 0.4:
            score += 3.0
            criticisms.append(f"One-year maximum drawdown is severe ({drawdown:.1%}).")
        elif drawdown > 0.25:
            score += 2.0
            criticisms.append(f"One-year maximum drawdown is material ({drawdown:.1%}).")
        elif drawdown > 0.15:
            score += 1.0

        has_unknown_signal = Signal.UNKNOWN in (technical.signal, fundamental.signal)
        if not has_unknown_signal and technical.signal != fundamental.signal:
            score += 1.0
            criticisms.append(f"Agents disagree: Technical={technical.signal.value}, Fundamental={fundamental.signal.value}.")
        if has_unknown_signal:
            # Missing evidence is uncertainty, not proof that the issuer itself is risky.
            # Keep a small prudence premium and express the larger effect through confidence.
            score += 0.75
            warnings.append("At least one specialist has insufficient evidence (UNKNOWN).")
        elif min(technical.confidence, fundamental.confidence) < 0.5:
            score += 0.5
            warnings.append("At least one specialist confidence is below 0.50.")
        min_traded_value = self._minimum_traded_value()
        if average_traded_value < min_traded_value:
            score += 1.0
            criticisms.append(f"Liquidity is below the configured average traded-value filter ({average_traded_value:,.0f} VND).")
        quality_issues = state.market_data.quality_report.issues
        if quality_issues:
            # Quality issues lower confidence below; counting every issue in the risk
            # score duplicated stale/missing-data penalties and made HIGH nearly automatic.
            warnings.append(f"Data-quality warnings affect confidence ({len(quality_issues)} issue(s)).")
        debt_to_equity = state.market_data.financials.debt_to_equity
        sector = state.market_data.instrument.sector if state.market_data.instrument else None
        financial_sector = self._is_financial_sector(sector)
        if debt_to_equity is not None and not financial_sector:
            if debt_to_equity > 2.0:
                score += 1.0
                criticisms.append("Leverage is high according to the available debt/equity metric.")
            elif debt_to_equity > 1.5:
                score += 0.5
                criticisms.append("Leverage is elevated according to the available debt/equity metric.")
        event_refs = [source for source in state.market_data.source_refs if source.data_kind is DataKind.ISSUER_EVENTS]
        current_as_of = state.market_data.as_of or state.market_data.bars[-1].date
        recent_events = sorted(
            [event for event in state.market_data.issuer_events if event.event_date <= current_as_of],
            key=lambda event: event.event_date,
            reverse=True,
        )[:3]
        has_trading_suspension = any(event.event_code == "SUSP" for event in recent_events)
        if has_trading_suspension:
            score += 2.0
            criticisms.append("An issuer event reports a trading suspension; verify the official disclosure before acting.")
        position_total = sum(position.weight for position in state.portfolio_positions)
        normalized_weights = [position.weight / position_total for position in state.portfolio_positions] if position_total else []
        concentration = sum(weight ** 2 for weight in normalized_weights)
        max_weight = max(normalized_weights, default=0.0)
        if max_weight >= 0.5:
            score += 1.0
            criticisms.append(f"Portfolio concentration is high (largest normalized position {max_weight:.0%}).")
        elif max_weight >= 0.3:
            score += 0.5
        score = min(10.0, score)
        level = RiskLevel.HIGH if score >= 7.5 else RiskLevel.MEDIUM if score >= 3.5 else RiskLevel.LOW
        if has_unknown_signal and level is RiskLevel.LOW:
            # Do not claim LOW risk when evidence is incomplete; uncertainty alone is
            # MEDIUM, but it no longer escalates the issuer to HIGH or triggers a veto.
            level = RiskLevel.MEDIUM
        veto = has_trading_suspension or score >= 8.5
        if veto:
            warnings.append("Risk veto: the Manager must not emit BUY from this evidence set.")
        source_ids = [source.source_id for source in state.market_data.source_refs]
        as_of = state.market_data.as_of
        sections = [
            RiskSection(
                name="market", status="ASSESSED",
                observations=[f"Volatility {volatility:.1%}", f"Maximum drawdown {drawdown:.1%}"],
                evidence=[
                    EvidenceItem(label="Volatility", value=f"{volatility:.1%}", input_fields=["bars.close"], source_ids=source_ids, as_of=as_of),
                    EvidenceItem(label="Maximum drawdown", value=f"{drawdown:.1%}", input_fields=["bars.close"], source_ids=source_ids, as_of=as_of),
                ],
            ),
            RiskSection(
                name="liquidity", status="ASSESSED",
                observations=[f"Average traded value {average_traded_value:,.0f} VND", f"Configured filter {min_traded_value:,.0f} VND"],
                evidence=[EvidenceItem(
                    label="Average traded value", value=f"{average_traded_value:,.0f} VND",
                    input_fields=["bars.close", "bars.volume"], source_ids=source_ids, as_of=as_of,
                )],
            ),
            RiskSection(
                name="data", status="ASSESSED",
                observations=[issue.message for issue in quality_issues] or ["No data-quality issue reported"],
                evidence=[EvidenceItem(
                    label="Data-quality status", value=state.market_data.quality_report.status.value,
                    input_fields=["quality_report"], source_ids=source_ids, as_of=as_of,
                )],
            ),
            RiskSection(
                name="business_and_accounting", status="ASSESSED",
                observations=[
                    f"Debt/equity: {state.market_data.financials.debt_to_equity:.2f}"
                    if state.market_data.financials.debt_to_equity is not None else "Debt/equity is unavailable",
                    *fundamental.weaknesses[:2],
                ],
                evidence=[EvidenceItem(
                    label="Fundamental confidence", value=f"{fundamental.confidence:.0%}",
                    input_fields=["financials", "fundamental_result"], source_ids=source_ids, as_of=as_of,
                    confidence=fundamental.confidence,
                )],
            ),
            RiskSection(
                name="event_news", status="ASSESSED" if event_refs else "UNKNOWN",
                observations=(
                    [f"{event.event_date.isoformat()} · {event.event_code or 'EVENT'} · {event.title}" for event in recent_events]
                    or ["No issuer events were returned by the verified event source."]
                    if event_refs else ["No verified issuer-event or news feed is attached to this analysis run."]
                ),
                evidence=[EvidenceItem(
                    label="Verified issuer events", value=f"{len(recent_events)} event(s) in current source window",
                    input_fields=["issuer_events"], source_ids=[source.source_id for source in event_refs], as_of=as_of,
                )] if event_refs else [],
            ),
            RiskSection(
                name="portfolio", status="ASSESSED" if normalized_weights else "UNKNOWN",
                observations=(
                    [
                        f"Input weight total {position_total:.1%}; normalized largest position {max_weight:.1%}",
                        f"Concentration HHI {concentration:.3f}; cross-asset correlation is unavailable without return series for every holding.",
                    ] if normalized_weights else ["No user holdings or weights were supplied."]
                ),
                evidence=[EvidenceItem(
                    label="Portfolio concentration", value=f"HHI {concentration:.3f}; largest {max_weight:.1%}",
                    input_fields=["portfolio_positions.weight"], source_ids=[], as_of=as_of,
                )] if normalized_weights else [],
            ),
        ]
        result = RiskAnalysis(
            risk_level=level,
            risk_score=round(score, 2),
            criticisms=criticisms,
            warnings=warnings,
            veto=veto,
            reasoning_summary=(
                f"Market volatility, drawdown, liquidity, data quality and specialist agreement produce {level.value} risk ({score:.1f}/10). "
                "Event/news and portfolio risk are marked UNKNOWN only when their verified inputs are unavailable."
            ),
            sections=sections,
            confidence=round(max(0.1, 0.8 - 0.08 * len(quality_issues) - (0.15 if fundamental.signal is Signal.UNKNOWN else 0)), 2),
            policy_version="risk-v3-balanced",
        )
        llm_review = self.llm_review(
            "Adversarially challenge specialist conclusions and identify unmeasured risks",
            {
                "ticker": state.ticker,
                "horizon": state.investment_horizon,
                "technical_result": technical.model_dump(mode="json"),
                "fundamental_result": fundamental.model_dump(mode="json"),
                "deterministic_risk_result": result.model_dump(mode="json"),
                "quantitative_forecast": state.price_forecast.model_dump(mode="json") if state.price_forecast else None,
                "prior_llm_reviews": [
                    review.model_dump(mode="json") for review in state.llm_reviews
                    if review.status == "GROUNDED"
                ],
            },
            source_ids,
        )
        return {
            "risk_result": result,
            "messages": [self.message(AgentName.MANAGER, MessageType.CRITIQUE, "Risk critique complete", result.model_dump(mode="json"), max(0.1, 1 - score / 12))],
            "llm_reviews": [llm_review] if llm_review is not None else [],
            "trace": [
                self.event("OBSERVE", "READ_SPECIALIST_RESULTS", f"Technical={technical.signal.value}; Fundamental={fundamental.signal.value}"),
                self.event("ACT", "CALL_TOOL", "Measure recent volatility", "annualized_volatility"),
                self.event("ACT", "CALL_TOOL", "Measure one-year peak-to-trough loss", "max_drawdown"),
                self.event("ACT", "SEND_CRITIQUE", f"Risk={level.value}; veto={veto}"),
                *([self.event("REASON", "LLM_RISK_REVIEW", f"Structured LLM review={llm_review.status}", "local_llm")] if llm_review is not None else []),
            ],
            "completed_tasks": ["risk_reanalysis" if state.iteration else "risk_analysis"],
        }

    @staticmethod
    def _minimum_traded_value() -> float:
        try:
            return max(0.0, float(os.getenv("STOCK_AGENT_MIN_AVG_TRADED_VALUE_VND", "1000000000")))
        except ValueError:
            return 1_000_000_000.0

    @staticmethod
    def _is_financial_sector(sector: str | None) -> bool:
        normalized = (sector or "").casefold()
        return any(token in normalized for token in (
            "bank", "ngân hàng", "securities", "chứng khoán", "insurance", "bảo hiểm", "financial",
        ))

