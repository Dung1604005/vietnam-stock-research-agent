from __future__ import annotations

from stock_multi_agent.agents.base_agent import BaseAgent
import os
import unicodedata
from datetime import date

from stock_multi_agent.models.agent_contracts import AgentSpec, RiskAgentContext
from stock_multi_agent.models.analysis import EvidenceItem, RiskAnalysis, RiskLevel, RiskSection, Signal
from stock_multi_agent.models.data import DataKind
from stock_multi_agent.models.messages import AgentName, MessageType
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.tools.technical_indicators import TechnicalIndicatorToolkit
from stock_multi_agent.services.financial_freshness import scoring_financials


class RiskAgent(BaseAgent):
    name = AgentName.RISK
    goal = "Challenge optimistic conclusions and prevent unjustified confidence."
    spec = AgentSpec(
        role=name,
        goal=goal,
        required_context=["market_data", "technical_result", "fundamental_result", "portfolio_positions", "research_evidence"],
        allowed_tools=["annualized_volatility", "max_drawdown", "average_traded_value", 'structured_llm_review'],
        output_type="RiskAnalysis",
        can_use_llm=True,
        decision_constraints=[
            "Separate issuer/portfolio risk from missing-data uncertainty",
            "Only verified severe events or a high measured score may trigger veto",
            "Research prose cannot alter the deterministic risk score or veto",
        ],
    )

    def __init__(self, toolkit: TechnicalIndicatorToolkit, llm_advisor=None) -> None:
        self.tools = toolkit
        self.llm_advisor = llm_advisor

    def run(self, state: AgentState, allow_llm_review: bool = True) -> dict[str, object]:
        context = RiskAgentContext.from_state(state)
        closes = [bar.close for bar in state.market_data.bars]
        volumes = [bar.volume for bar in state.market_data.bars]
        volatility = self.tools.annualized_volatility(closes[-60:])
        drawdown = self.tools.max_drawdown(closes[-252:])
        average_traded_value = self.tools.average_traded_value(closes, volumes)
        # OHLCV providers do not agree on VND vs thousand VND. A missing scale
        # is uncertainty, never proof of illiquidity.
        turnover_scale = {"VND": 1.0, "THOUSAND_VND": 1000.0}.get(state.market_data.price_unit)
        turnover_vnd = average_traded_value * turnover_scale if turnover_scale is not None else None
        technical = state.technical_result
        fundamental = state.fundamental_result
        score = 1.0
        contributions = {"base": 1.0, "volatility": 0.0, "drawdown": 0.0,
                         "liquidity": 0.0, "leverage": 0.0, "issuer_event": 0.0, "portfolio_concentration": 0.0}
        uncertainty_score = 0.5
        criticisms: list[str] = []
        warnings = list(state.market_data.warnings)

        if volatility > 0.5:
            score += 3.0
            contributions["volatility"] = 3.0
            criticisms.append(f"Annualized recent volatility is very high ({volatility:.1%}).")
        elif volatility > 0.35:
            score += 2.0
            contributions["volatility"] = 2.0
            criticisms.append(f"Annualized recent volatility is elevated ({volatility:.1%}).")
        elif volatility > 0.25:
            score += 1.0
            contributions["volatility"] = 1.0
        if drawdown > 0.4:
            score += 3.0
            contributions["drawdown"] = 3.0
            criticisms.append(f"One-year maximum drawdown is severe ({drawdown:.1%}).")
        elif drawdown > 0.25:
            score += 2.0
            contributions["drawdown"] = 2.0
            criticisms.append(f"One-year maximum drawdown is material ({drawdown:.1%}).")
        elif drawdown > 0.15:
            score += 1.0
            contributions["drawdown"] = 1.0

        has_unknown_signal = Signal.UNKNOWN in (technical.signal, fundamental.signal)
        if not has_unknown_signal and technical.signal != fundamental.signal:
            uncertainty_score += 1.25
            criticisms.append(f"Agents disagree: Technical={technical.signal.value}, Fundamental={fundamental.signal.value}.")
        if has_unknown_signal:
            uncertainty_score += 2.0
            warnings.append("At least one specialist has insufficient evidence (UNKNOWN).")
        elif min(technical.confidence, fundamental.confidence) < 0.5:
            uncertainty_score += 1.0
            warnings.append("At least one specialist confidence is below 0.50.")
        min_traded_value = self._minimum_traded_value()
        if turnover_vnd is not None and turnover_vnd < min_traded_value:
            score += 1.0
            contributions["liquidity"] = 1.0
            criticisms.append(f"Liquidity is below the configured average traded-value filter ({turnover_vnd:,.0f} VND).")
        elif turnover_vnd is None:
            warnings.append("Price scale is unverified; native turnover is not compared with a VND liquidity threshold.")
        quality_issues = state.market_data.quality_report.issues
        if quality_issues:
            # Quality issues lower confidence below; counting every issue in the risk
            # score duplicated stale/missing-data penalties and made HIGH nearly automatic.
            warnings.append(f"Data-quality warnings affect confidence ({len(quality_issues)} issue(s)).")
            uncertainty_score += min(3.0, 0.75 * len(quality_issues))
        eligible_financials, financial_limits = scoring_financials(state.market_data)
        debt_to_equity = eligible_financials.debt_to_equity
        warnings.extend(financial_limits)
        sector = state.market_data.instrument.sector if state.market_data.instrument else None
        financial_sector = self._is_financial_sector(sector)
        sector_template = self._sector_template(sector)
        if debt_to_equity is not None and not financial_sector:
            if debt_to_equity > 2.0:
                score += 1.0
                contributions["leverage"] = 1.0
                criticisms.append("Leverage is high according to the available debt/equity metric.")
            elif debt_to_equity > 1.5:
                score += 0.5
                contributions["leverage"] = 0.5
                criticisms.append("Leverage is elevated according to the available debt/equity metric.")
        event_refs = [source for source in state.market_data.source_refs if source.data_kind is DataKind.ISSUER_EVENTS]
        event_source_ids = {source.source_id for source in event_refs}
        current_as_of = state.market_data.as_of or state.market_data.bars[-1].date
        recent_events = sorted(
            [event for event in state.market_data.issuer_events if event.source_id in event_source_ids
             and event.event_date <= current_as_of
             and (event.published_date is None or event.published_date <= current_as_of)],
            key=lambda event: event.event_date,
            reverse=True,
        )
        has_trading_suspension = any(event.event_code == "SUSP" for event in recent_events)
        if has_trading_suspension:
            score += 2.0
            contributions["issuer_event"] = 2.0
            criticisms.append("An issuer event reports a trading suspension; verify the official disclosure before acting.")
        issuer_score = min(10.0, score)
        position_total = sum(position.weight for position in state.portfolio_positions)
        normalized_weights = [position.weight / position_total for position in state.portfolio_positions] if position_total else []
        concentration = sum(weight ** 2 for weight in normalized_weights)
        max_weight = max(normalized_weights, default=0.0)
        if max_weight >= 0.5:
            score += 1.0
            contributions["portfolio_concentration"] = 1.0
            criticisms.append(f"Portfolio concentration is high (largest normalized position {max_weight:.0%}).")
        elif max_weight >= 0.3:
            score += 0.5
            contributions["portfolio_concentration"] = 0.5
        contributions["score_cap_adjustment"] = min(10.0, score) - score
        score = min(10.0, score)
        level = RiskLevel.HIGH if score >= 7.5 else RiskLevel.MEDIUM if score >= 3.5 else RiskLevel.LOW
        issuer_level = RiskLevel.HIGH if issuer_score >= 7.5 else RiskLevel.MEDIUM if issuer_score >= 3.5 else RiskLevel.LOW
        uncertainty_score = min(10.0, uncertainty_score)
        uncertainty_level = RiskLevel.HIGH if uncertainty_score >= 7.0 else RiskLevel.MEDIUM if uncertainty_score >= 2.5 else RiskLevel.LOW
        veto = has_trading_suspension or issuer_score >= 8.5
        if veto:
            warnings.append("Risk veto: the Manager must not emit BUY from this evidence set.")
        source_ids = [source.source_id for source in state.market_data.source_refs]
        price_sources = [source.source_id for source in state.market_data.source_refs if source.data_kind is DataKind.OHLCV]
        financial_sources = [source.source_id for source in state.market_data.source_refs if source.data_kind in (
            DataKind.FINANCIAL_RATIOS, DataKind.INCOME_STATEMENT, DataKind.BALANCE_SHEET, DataKind.CASH_FLOW)]
        as_of = state.market_data.as_of
        business = self._business_section(state, sector_template, financial_sources)
        market_coverage = min(len(closes), 252) / 252
        liquidity_coverage = (0.5 if len(closes) >= 20 else 0.0) + (0.5 if turnover_vnd is not None else 0.0)
        sections = [
            RiskSection(
                name="market", status="ASSESSED" if market_coverage == 1 else "PARTIAL", coverage=market_coverage,
                missing_information=[] if market_coverage == 1 else ["252 price sessions for a full one-year drawdown window"],
                observations=[f"Volatility {volatility:.1%}", f"Maximum drawdown {drawdown:.1%}",
                              f"Measured over {min(len(closes), 60)} / {min(len(closes), 252)} available sessions, not a forward probability."],
                evidence=[
                    EvidenceItem(label="Volatility", value=f"{volatility:.1%}", input_fields=["bars.close"], source_ids=price_sources, as_of=as_of),
                    EvidenceItem(label="Maximum drawdown", value=f"{drawdown:.1%}", input_fields=["bars.close"], source_ids=price_sources, as_of=as_of),
                ],
            ),
            RiskSection(
                name="liquidity", status="ASSESSED" if liquidity_coverage == 1 else "PARTIAL", coverage=liquidity_coverage,
                missing_information=[] if turnover_vnd is not None else ["Verified OHLCV price scale (VND or THOUSAND_VND)"],
                observations=[f"Average traded value {turnover_vnd:,.0f} VND" if turnover_vnd is not None else
                              f"Native turnover {average_traded_value:,.0f}; price scale UNKNOWN, VND filter not applied",
                              f"Configured filter {min_traded_value:,.0f} VND"],
                evidence=[EvidenceItem(
                    label="Average traded value", value=f"{turnover_vnd:,.0f} VND" if turnover_vnd is not None else f"{average_traded_value:,.0f} native units (unverified scale)",
                    input_fields=["bars.close", "bars.volume", "price_unit"], source_ids=price_sources, as_of=as_of,
                )],
            ),
            RiskSection(
                name="data", status="ASSESSED", coverage=1.0,
                observations=[
                    f"Data uncertainty {uncertainty_level.value} ({uncertainty_score:.1f}/10)",
                    *([issue.message for issue in quality_issues] or ["No data-quality issue reported"]),
                ],
                evidence=[EvidenceItem(
                    label="Data-quality status", value=state.market_data.quality_report.status.value,
                    input_fields=["quality_report"], source_ids=source_ids, as_of=as_of,
                )],
            ),
            business,
            RiskSection(
                name="event_news", status="ASSESSED" if event_refs else "UNKNOWN",
                coverage=1.0 if event_refs else 0.0,
                missing_information=[] if event_refs else ["Verified issuer-event feed; absence is not proof of no event"],
                observations=(
                    [f"{event.event_date.isoformat()} · {event.event_code or 'EVENT'} · {event.title}" for event in recent_events[:3]]
                    or ["No issuer events were returned by the verified event source."]
                    if event_refs else ["No verified issuer-event or news feed is attached to this analysis run."]
                ),
                evidence=[EvidenceItem(
                    label="Verified issuer events", value=f"{len(recent_events)} event(s) in current source window",
                    input_fields=["issuer_events"], source_ids=[source.source_id for source in event_refs], as_of=as_of,
                )] if event_refs else [],
            ),
            RiskSection(
                name="portfolio", status="PARTIAL" if normalized_weights else "UNKNOWN",
                coverage=0.5 if normalized_weights else 0.0,
                missing_information=["Aligned return series for every holding; covariance and stress scenarios"],
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
            RiskSection(
                name="research_context", status="PARTIAL" if state.research_evidence else "UNKNOWN",
                coverage=0.5 if state.research_evidence else 0.0,
                missing_information=["Independent claim verification; a retrieved passage is not a comprehensive risk assessment"],
                observations=(
                    [f"{item.publisher}: {item.title}" for item in state.research_evidence[:3]]
                    if state.research_evidence else ["No verified local research citation matched this ticker."]
                ),
                evidence=[EvidenceItem(
                    label=f"Research context: {item.title}"[:160],
                    value=item.quote if len(item.quote) <= 500 else item.quote[:497] + "...",
                    input_fields=["research_evidence"],
                    source_ids=[item.document_id, item.chunk_id],
                    as_of=item.published_at.date(),
                    confidence=min(0.75, item.retrieval_score),
                    note="Contextual evidence; it does not directly set the risk score.",
                ) for item in state.research_evidence[:3]],
            ),
        ]
        result = RiskAnalysis(
            risk_level=level,
            risk_score=round(score, 2),
            criticisms=criticisms,
            warnings=warnings,
            veto=veto,
            reasoning_summary=(
                f"Measured issuer and portfolio factors produce {level.value} risk ({score:.1f}/10); "
                f"issuer-only risk is {issuer_level.value} ({issuer_score:.1f}/10), while data uncertainty is "
                f"{uncertainty_level.value} ({uncertainty_score:.1f}/10)."
            ),
            sections=sections,
            confidence=round(max(0.1, 0.8 - 0.08 * len(quality_issues) - (0.15 if fundamental.signal is Signal.UNKNOWN else 0)), 2),
            policy_version="risk-v4-decomposed",
            feature_contributions=contributions,
            coverage={section.name: section.coverage for section in sections},
            sector_template=sector_template,
            issuer_risk_level=issuer_level,
            issuer_risk_score=round(issuer_score, 2),
            data_uncertainty_level=uncertainty_level,
            data_uncertainty_score=round(uncertainty_score, 2),
        )
        llm_review = self.review_result(state, result) if allow_llm_review else None
        return {
            "risk_result": result,
            "messages": [self.message(AgentName.MANAGER, MessageType.CRITIQUE, "Risk critique complete", result.model_dump(mode="json"), max(0.1, 1 - score / 12))],
            "llm_reviews": [llm_review] if llm_review is not None else [],
            "trace": [
                self.event("OBSERVE", "READ_SPECIALIST_RESULTS", f"Technical={technical.signal.value}; Fundamental={fundamental.signal.value}"),
                self.event("ACT", "CALL_TOOL", "Measure recent volatility", "annualized_volatility"),
                self.event("ACT", "CALL_TOOL", "Measure one-year peak-to-trough loss", "max_drawdown"),
                self.event("ACT", "SEND_CRITIQUE", f"Risk={level.value}; veto={veto}"),
                *([self.event("REASON", "LLM_RISK_REVIEW", f"Structured LLM review={llm_review.status}", "configured_llm")] if llm_review is not None else []),
            ],
            "completed_tasks": ["risk_reanalysis" if state.iteration else "risk_analysis"],
            "agent_reports": [self.report(
                stage="risk_reanalysis" if state.iteration else "risk_analysis",
                status="PARTIAL" if uncertainty_level is RiskLevel.HIGH else "COMPLETED",
                summary=result.reasoning_summary,
                confidence=result.confidence,
                output_key="risk_result",
                tools_used=["annualized_volatility", "max_drawdown", "average_traded_value"],
                assumptions=["Data uncertainty is reported separately from measured risk"],
                missing_information=[f"{section.name}: {item}" for section in sections for item in section.missing_information],
                evidence_count=sum(len(section.evidence) for section in sections),
            )],
        }

    def review_result(self, state, result):
        if not state.allow_llm_planning:
            return None
        return self.llm_review('Phản biện các kết luận và giải thích nguồn gốc rủi ro, kịch bản bất lợi có điều kiện', {
            'ticker': state.ticker, 'horizon': state.investment_horizon,
            'source_provenance': [{key: item.model_dump(mode='json')[key] for key in ('source_id', 'provider', 'data_kind', 'as_of', 'fetched_at', 'source_url')} for item in state.market_data.source_refs],
            'research_context': [item.model_dump(mode='json') for item in state.research_evidence[:2]],
            'technical_result': state.technical_result.model_dump(mode='json'),
            'fundamental_result': state.fundamental_result.model_dump(mode='json'),
            'deterministic_risk_result': result.model_dump(mode='json'),
            'quantitative_forecast': state.price_forecast.model_dump(mode='json') if state.price_forecast else None,
        }, [*[item.source_id for item in state.market_data.source_refs], *[item.chunk_id for item in state.research_evidence]])

    @staticmethod
    def _minimum_traded_value() -> float:
        try:
            return max(0.0, float(os.getenv("STOCK_AGENT_MIN_AVG_TRADED_VALUE_VND", "1000000000")))
        except ValueError:
            return 1_000_000_000.0

    @staticmethod
    def _is_financial_sector(sector: str | None) -> bool:
        normalized = RiskAgent._normalize_sector(sector)
        return any(token in normalized for token in (
            "bank", "ngan hang", "securities", "chung khoan", "insurance", "bao hiem", "financial",
        ))

    @staticmethod
    def _normalize_sector(sector: str | None) -> str:
        return "".join(char for char in unicodedata.normalize("NFKD", (sector or "").casefold())
                       if not unicodedata.combining(char)).replace("đ", "d")

    @classmethod
    def _sector_template(cls, sector: str | None) -> str:
        normalized = cls._normalize_sector(sector)
        for label, tokens in (
            ("BANK", ("bank", "ngan hang")),
            ("SECURITIES", ("securities", "chung khoan")),
            ("INSURANCE", ("insurance", "bao hiem")),
            ("REAL_ESTATE", ("real estate", "bat dong san")),
        ):
            if any(token in normalized for token in tokens):
                return label
        return "NON_FINANCIAL_GENERIC" if normalized and not cls._is_financial_sector(sector) else "UNKNOWN"

    @classmethod
    def _business_section(cls, state: AgentState, template: str, financial_sources: list[str]) -> RiskSection:
        """Describe measured fields and explicitly unmeasured sector risks.

        Sector checklists are not evidence that the issuer actually has a risk.
        NPL/CAR/etc are observed only when the structured source supplies them;
        no generic bank/insurance thresholds or made-up values are introduced.
        """
        required = {
            "BANK": ("roe", "roa", "profit_growth", "npl_ratio", "capital_adequacy_ratio", "loan_to_deposit_ratio"),
            "SECURITIES": ("roe", "profit_growth", "margin_lending", "proprietary_trading_exposure", "capital_adequacy_ratio"),
            "INSURANCE": ("roe", "profit_growth", "combined_ratio", "solvency_ratio", "reserve_adequacy"),
            "REAL_ESTATE": ("debt_to_equity", "operating_cash_flow", "profit_growth", "inventory", "project_legal_status"),
            "NON_FINANCIAL_GENERIC": ("debt_to_equity", "operating_cash_flow", "profit_margin", "revenue_growth", "profit_growth", "roe"),
            "UNKNOWN": ("roe", "profit_growth", "operating_cash_flow", "verified_sector_classification"),
        }[template]
        hints = {
            "BANK": "Bank-specific credit quality, capital and funding risks require NPL/CAR/LDR; high debt/equity is not scored like industrial leverage.",
            "SECURITIES": "Margin lending, proprietary trading and regulatory capital exposures need separate evidence; bank/industrial leverage thresholds are not applied.",
            "INSURANCE": "Underwriting, reserves, investment exposure and solvency require insurer-specific metrics; generic debt/equity does not establish insurance risk.",
            "REAL_ESTATE": "Debt, operating cash flow, inventory and project legal status are separate checks; industry membership alone establishes no adverse event.",
            "NON_FINANCIAL_GENERIC": "Debt, operating cash flow, margins and growth are limited checks; audit opinion, customers and contingent liabilities require reports.",
            "UNKNOWN": "Sector is unverified; industry-specific prudential conclusions are withheld.",
        }
        metrics = scoring_financials(state.market_data)[0].model_dump()
        observations = [f"Sector template: {template}", hints[template]]
        evidence = []
        available = set()
        for metric in required:
            value = metrics.get(metric)
            if value is None:
                continue
            available.add(metric)
            value_text = (f"{value:.1%}" if metric in {"roe", "roa", "profit_growth", "revenue_growth", "profit_margin"}
                          else f"{value:,.2f}" + (" (native financial unit)" if metric == "operating_cash_flow" else ""))
            observations.append(f"{metric}: {value_text}")
            evidence.append(EvidenceItem(label=metric, value=value_text, input_fields=[f"financials.{metric}"],
                source_ids=financial_sources, as_of=state.market_data.as_of,
                note="Provider-normalized metric; period and original source scale must be checked before cross-period comparison."))
        # Preserve source period/unit for specialized fields. No arithmetic is
        # performed across incomparable statements or guessed ratio scales.
        known_sources = {source.source_id for source in state.market_data.source_refs}
        structured = sorted([*state.market_data.financial_observations, *state.market_data.financial_statement_observations],
                            key=lambda item: item.period_end or date.min, reverse=True)
        for item in structured:
            if item.metric not in required or item.metric in available or not item.period_end:
                continue
            if item.period_end > state.market_data.as_of or item.source_id not in known_sources:
                continue
            if item.published_at and item.published_at.date() > state.market_data.as_of:
                continue
            if item.unit.casefold() in {"", "unknown", "unspecified"}:
                continue
            available.add(item.metric)
            text = f"{item.value:,.4g} {item.unit}; period ending {item.period_end}"
            observations.append(f"{item.metric}: {text}")
            evidence.append(EvidenceItem(label=item.metric, value=text, input_fields=["financial_observations", "financial_statement_observations"],
                source_ids=[item.source_id], as_of=item.period_end,
                note="Observed only; no sector threshold or inferred risk-score contribution."))
        missing = [metric for metric in required if metric not in available]
        # Full accounting assurance cannot be inferred from six financial ratios.
        missing.extend(["Verified audit opinion and contingent liabilities", "Comparable reporting basis and financial-period freshness"])
        if available and not financial_sources:
            missing.append("Financial-source references for normalized metrics")
        observations.extend(state.fundamental_result.weaknesses[:2])
        if metrics.get("profit_growth") is not None and metrics["profit_growth"] < 0:
            observations.append("Reported profit growth is negative; this observation alone does not identify its cause.")
        if template not in {"BANK", "SECURITIES", "INSURANCE"} and metrics.get("operating_cash_flow") is not None and metrics["operating_cash_flow"] < 0:
            observations.append("Reported operating cash flow is negative; working-capital timing and period comparability need review.")
        return RiskSection(name="business_and_accounting", status="PARTIAL" if available else "UNKNOWN",
            coverage=len(available) / (len(required) + 2), observations=observations,
            missing_information=missing, evidence=evidence)

