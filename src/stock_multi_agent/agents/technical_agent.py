from __future__ import annotations

import os

from stock_multi_agent.agents.base_agent import BaseAgent
from stock_multi_agent.models.agent_contracts import AgentSpec, TechnicalAgentContext
from stock_multi_agent.models.analysis import EvidenceItem, PriceAdjustment, Signal, TechnicalAnalysis
from stock_multi_agent.models.forecast import ForecastStatus
from stock_multi_agent.models.llm import AgentLLMReview
from stock_multi_agent.models.messages import AgentName, MessageType
from stock_multi_agent.models.state import AgentState, TraceEvent
from stock_multi_agent.tools.technical_indicators import HorizonProfile, LONG_HORIZON, TechnicalIndicatorToolkit


class TechnicalAgent(BaseAgent):
    """Deterministic, horizon-aware technical engine.

    Scores are diagnostics, not calibrated return forecasts. The policy version
    and feature contributions make the result reviewable and evaluable later.
    """

    name = AgentName.TECHNICAL
    goal = "Assess price trend with a versioned indicator profile for the requested horizon."
    spec = AgentSpec(
        role=name,
        goal=goal,
        required_context=["market_data", "investment_horizon", "benchmark_data", "price_forecast"],
        allowed_tools=[
            "SMA", "RSI", "MACD", "annualized_volatility", "volume_ratio",
            "average_traded_value", "BOLLINGER_BANDS", "MAX_DRAWDOWN",
            "SUPPORT_RESISTANCE", "ATR", "ADX", "local_llm_indicator_planner",
            "lstm_price_forecast_blend",
        ],
        output_type="TechnicalAnalysis",
        can_use_llm=True,
        decision_constraints=[
            "LLM may select allowlisted tools but never calculate indicator values",
            "Python owns scores, signals, and numerical evidence",
            "A valid one-session LSTM forecast contributes exactly 20% to the final technical score",
            "An untrained baseline or unavailable forecast contributes 0% to the technical score",
        ],
    )

    def __init__(self, toolkit: TechnicalIndicatorToolkit, llm_advisor=None) -> None:
        self.tools = toolkit
        self.llm_advisor = llm_advisor

    def run(
        self,
        state: AgentState,
        extended: bool = False,
        allow_llm_planning: bool = True,
    ) -> dict[str, object]:
        context = TechnicalAgentContext.from_state(state)
        closes = [bar.close for bar in context.market_data.bars]
        volumes = [bar.volume for bar in context.market_data.bars]
        profile = LONG_HORIZON if extended else self.tools.horizon_profile(context.investment_horizon)
        timeframe = f"extended {context.investment_horizon} ({profile.name} profile)" if extended else f"{context.investment_horizon} ({profile.name} profile)"
        traces: list[TraceEvent] = [self.event(
            "OBSERVE", "READ_MARKET_DATA", f"Observe {len(closes)} OHLCV bars; profile={profile.name}",
        )]
        required_history = max(50, profile.slow_window + profile.macd_signal)
        buy_threshold = 7.0
        source_ids = [source.source_id for source in state.market_data.source_refs]
        indicator_plan = self._existing_indicator_plan(state) if extended else None
        if indicator_plan is None and not state.requested_indicators and allow_llm_planning:
            indicator_plan = self._request_indicator_plan(state, source_ids)
        selected_checks = indicator_plan[0] if indicator_plan is not None else []
        if len(closes) < required_history:
            analysis = TechnicalAnalysis(
                signal=Signal.UNKNOWN, score=5, confidence=0.1,
                evidence=[self._evidence(state, "Price history", f"Only {len(closes)} sessions; {required_history} required", ["bars"])],
                risks=["Insufficient price history for the requested horizon"], tools_used=[],
                reasoning_summary="Technical scoring was withheld because the selected horizon needs more history.",
                timeframe=timeframe, horizon_profile=profile.name,
                diagnostics={"required_history": float(required_history), "available_history": float(len(closes))},
            )
        elif state.requested_indicators and not extended:
            analysis, extra_traces = self._run_requested_indicators(state, closes, volumes, state.requested_indicators, timeframe, profile)
            traces.extend(extra_traces)
            buy_threshold = 6.5
        else:
            analysis, extra_traces = self._run_default(
                state, closes, volumes, timeframe, profile, selected_checks,
            )
            traces.extend(extra_traces)

        analysis, blend_applied = self._apply_lstm_blend(
            state,
            analysis,
            buy_threshold=buy_threshold,
        )
        if blend_applied:
            traces.append(self.event(
                "REASON",
                "BLEND_LSTM_FORECAST",
                "Final technical score = 80% indicator score + 20% LSTM score",
                "lstm_price_forecast_blend",
            ))

        llm_review = None
        if indicator_plan is not None and not extended:
            selected, rationale, cited_source_ids = indicator_plan
            display_labels = {
                "BOLLINGER_BANDS": "dải Bollinger",
                "MAX_DRAWDOWN": "mức sụt giảm tối đa",
                "SUPPORT_RESISTANCE": "hỗ trợ/kháng cự",
                "ATR": "biên độ dao động ATR",
                "ADX": "sức mạnh xu hướng ADX",
            }
            selected_display = ", ".join(display_labels[item] for item in selected)
            llm_review = AgentLLMReview(
                agent=self.name,
                task="Chọn chỉ báo kỹ thuật bổ sung",
                status="GROUNDED",
                assessment=(
                    f"LLM chọn tính bổ sung {selected_display}; Python thực hiện phép tính trên dữ liệu nguồn. "
                    f"{rationale}"
                ),
                supporting_observations=selected,
                cited_source_ids=cited_source_ids,
                model_role="indicator_planner",
                controls_numeric_output=False,
            )
            traces.append(self.event(
                "PLAN", "LLM_INDICATOR_PLAN", f"Selected {', '.join(selected)}; calculations executed by Python", "local_llm",
            ))
        message = self.message(
            AgentName.MANAGER, MessageType.RESULT,
            "Technical re-analysis complete" if extended else "Technical analysis complete",
            analysis.model_dump(mode="json"), analysis.confidence,
        )
        traces.append(self.event("ACT", "SEND_RESULT", f"Technical signal={analysis.signal.value}, confidence={analysis.confidence:.2f}"))
        return {
            "technical_result": analysis,
            "messages": [message],
            "llm_reviews": [llm_review] if llm_review is not None else [],
            "trace": traces,
            "completed_tasks": ["technical_reanalysis" if extended else "technical_analysis"],
            "agent_reports": [self.report(
                stage="technical_reanalysis" if extended else "technical_analysis",
                status="UNAVAILABLE" if analysis.signal is Signal.UNKNOWN else "COMPLETED",
                summary=analysis.reasoning_summary,
                confidence=analysis.confidence,
                output_key="technical_result",
                tools_used=analysis.tools_used,
                assumptions=[f"Horizon profile={analysis.horizon_profile}"],
                missing_information=list(analysis.risks) if analysis.signal is Signal.UNKNOWN else [],
                evidence_count=len(analysis.evidence),
            )],
        }

    def _run_default(
        self,
        state: AgentState,
        closes: list[float],
        volumes: list[float],
        timeframe: str,
        profile: HorizonProfile,
        selected_checks: list[str] | None = None,
    ) -> tuple[TechnicalAnalysis, list[TraceEvent]]:
        selected = [
            f"SMA{profile.fast_window}", f"SMA{profile.slow_window}", f"RSI{profile.rsi_period}",
            f"MACD({profile.macd_fast},{profile.macd_slow},{profile.macd_signal})",
            f"volatility_{profile.volatility_window}", "volume_ratio", "average_traded_value",
        ]
        traces = [self.event("REASON", "SELECT_TOOLS", f"Selected {', '.join(selected)} for {profile.name} horizon")]
        sma_fast = self.tools.sma(closes, profile.fast_window)
        sma_slow = self.tools.sma(closes, profile.slow_window)
        rsi = self.tools.rsi(closes, profile.rsi_period)
        macd = self.tools.macd(closes, profile.macd_fast, profile.macd_slow, profile.macd_signal)
        volatility = self.tools.annualized_volatility(closes[-profile.volatility_window:])
        volume_ratio = self.tools.volume_ratio(volumes)
        average_traded_value = self.tools.average_traded_value(closes, volumes)
        highs = [bar.high for bar in state.market_data.bars]
        lows = [bar.low for bar in state.market_data.bars]
        return_window = min(profile.return_window, len(closes) - 1)
        trailing_return = closes[-1] / closes[-1 - return_window] - 1
        regime = "UPTREND" if closes[-1] > sma_fast > sma_slow else "DOWNTREND" if closes[-1] < sma_fast < sma_slow else "MIXED"

        score = 5.0
        contributions: dict[str, float] = {}
        risks: list[str] = []
        evidence = [
            self._evidence(state, "Trend", f"Close {closes[-1]:.2f}; SMA{profile.fast_window} {sma_fast:.2f}; SMA{profile.slow_window} {sma_slow:.2f}", ["bars.close"]),
            self._evidence(state, f"RSI{profile.rsi_period}", f"{rsi:.1f}", ["bars.close"]),
            self._evidence(state, "MACD histogram", f"{macd['histogram']:.3f}", ["bars.close"]),
            self._evidence(state, "Realized volatility", f"{volatility:.1%} over {profile.volatility_window} sessions", ["bars.close"]),
            self._evidence(state, "Liquidity", f"Average traded value {average_traded_value:,.0f} VND; volume ratio {volume_ratio:.2f}", ["bars.close", "bars.volume"]),
            self._evidence(state, "Trailing return", f"{trailing_return:.1%} over {return_window} sessions; regime={regime}", ["bars.close"]),
        ]
        selected_checks = selected_checks or []
        for check in selected_checks:
            if check == "BOLLINGER_BANDS":
                bands = self.tools.bollinger_bands(closes, min(20, len(closes)))
                evidence.append(self._evidence(
                    state,
                    "LLM-selected Bollinger Bands",
                    f"Lower {bands['lower']:.2f}; middle {bands['middle']:.2f}; upper {bands['upper']:.2f}",
                    ["bars.close"],
                ))
                selected.append("Bollinger Bands")
            elif check == "MAX_DRAWDOWN":
                drawdown = self.tools.max_drawdown(closes[-min(len(closes), profile.return_window):])
                evidence.append(self._evidence(
                    state, "LLM-selected max drawdown", f"{drawdown:.1%}", ["bars.close"],
                ))
                selected.append("Max drawdown")
                if drawdown > 0.3:
                    risks.append("Maximum drawdown over the selected horizon exceeds 30%")
            elif check == "SUPPORT_RESISTANCE":
                levels = self.tools.support_resistance(closes, min(60, len(closes)))
                evidence.append(self._evidence(
                    state,
                    "LLM-selected support/resistance",
                    f"Support {levels['support']:.2f}; resistance {levels['resistance']:.2f}",
                    ["bars.close"],
                ))
                selected.append("Support/resistance")
            elif check == "ATR":
                atr = self.tools.average_true_range(highs, lows, closes, profile.rsi_period)
                atr_percent = atr / closes[-1] if closes[-1] else 0.0
                evidence.append(self._evidence(
                    state, "LLM-selected ATR", f"ATR{profile.rsi_period} {atr:.2f} ({atr_percent:.1%} of close)",
                    ["bars.high", "bars.low", "bars.close"],
                ))
                selected.append(f"ATR{profile.rsi_period}")
                if atr_percent > 0.05:
                    risks.append("ATR exceeds 5% of the latest close, indicating wide daily ranges")
            elif check == "ADX":
                adx = self.tools.adx(highs, lows, closes, profile.rsi_period)
                evidence.append(self._evidence(
                    state, "LLM-selected ADX", f"ADX{profile.rsi_period} {adx:.1f}",
                    ["bars.high", "bars.low", "bars.close"],
                ))
                selected.append(f"ADX{profile.rsi_period}")
                if adx < 20:
                    risks.append("ADX indicates that the current trend is weak")

        contributions["price_vs_fast_sma"] = 1.0 if closes[-1] > sma_fast else -1.0
        contributions["fast_vs_slow_sma"] = 1.25 if sma_fast > sma_slow else -1.25
        score += contributions["price_vs_fast_sma"] + contributions["fast_vs_slow_sma"]
        if 50 <= rsi <= 70:
            contributions["rsi"] = 0.75
        elif rsi > 75:
            contributions["rsi"] = -1.0
            risks.append("RSI indicates a potentially overbought condition")
        elif rsi < 35:
            contributions["rsi"] = -0.75
            risks.append("Momentum is weak (RSI below 35)")
        else:
            contributions["rsi"] = 0.0
        contributions["macd"] = 0.75 if macd["histogram"] > 0 else -0.75
        score += contributions["rsi"] + contributions["macd"]
        contributions["volume_confirmation"] = 0.5 if volume_ratio > 1.2 else 0.0
        score += contributions["volume_confirmation"]
        if volatility > 0.5:
            contributions["volatility"] = -1.25
            risks.append("Very high recent volatility")
        elif volatility > 0.35:
            contributions["volatility"] = -0.5
            risks.append("Elevated recent volatility")
        else:
            contributions["volatility"] = 0.0
        score += contributions["volatility"]

        min_traded_value = self._minimum_traded_value()
        if average_traded_value < min_traded_value:
            contributions["liquidity"] = -0.5
            score += contributions["liquidity"]
            risks.append(f"Average traded value is below the configured liquidity filter ({min_traded_value:,.0f} VND)")
        else:
            contributions["liquidity"] = 0.0
        if state.market_data.price_adjustment is PriceAdjustment.UNKNOWN:
            contributions["price_adjustment_quality"] = -0.25
            score += contributions["price_adjustment_quality"]
            risks.append("Price-adjustment status is unknown; corporate actions may affect indicators")
        else:
            contributions["price_adjustment_quality"] = 0.0

        score = max(0.0, min(10.0, score))
        confidence = min(0.88, 0.5 + min(len(closes), 252) / 1000)
        if average_traded_value < min_traded_value:
            confidence -= 0.08
        if state.market_data.price_adjustment is PriceAdjustment.UNKNOWN:
            confidence -= 0.08
        relative_benchmark_return, benchmark_note = self._relative_benchmark_return(state, return_window)
        if relative_benchmark_return is None:
            contributions["relative_benchmark"] = 0.0
        else:
            contributions["relative_benchmark"] = 0.5 if relative_benchmark_return > 0 else -0.5
            score = max(0.0, min(10.0, score + contributions["relative_benchmark"]))
            evidence.append(self._benchmark_evidence(state, relative_benchmark_return, return_window))
        signal = Signal.BUY if score >= 7.0 else Signal.HOLD if score >= 4.0 else Signal.SELL
        diagnostics: dict[str, str | float | None] = {
            "average_traded_value_vnd": round(average_traded_value, 2),
            "liquidity_filter_vnd": min_traded_value,
            "relative_benchmark_return": round(relative_benchmark_return, 6) if relative_benchmark_return is not None else None,
            "relative_benchmark_note": benchmark_note,
            "regime": regime,
            "trailing_return": round(trailing_return, 6),
            "annualized_volatility": round(volatility, 6),
            "price_adjustment": state.market_data.price_adjustment.value,
        }
        traces.extend(self.event("ACT", "CALL_TOOL", f"Calculated {tool}", tool) for tool in selected)
        return TechnicalAnalysis(
            signal=signal, score=round(score, 2), confidence=round(max(0.1, confidence), 2),
            evidence=evidence, risks=risks, tools_used=selected,
            reasoning_summary=(
                f"{profile.name.capitalize()}-horizon trend, momentum, liquidity and volatility diagnostics imply {signal.value}; "
                f"score {score:.1f}/10. {benchmark_note}"
            ),
            timeframe=timeframe, horizon_profile=profile.name,
            feature_contributions=contributions, diagnostics=diagnostics,
        ), traces

    @staticmethod
    def _apply_lstm_blend(
        state: AgentState,
        analysis: TechnicalAnalysis,
        *,
        buy_threshold: float,
    ) -> tuple[TechnicalAnalysis, bool]:
        """Blend an eligible next-session LSTM signal into the Technical score.

        The model return is mapped deterministically onto a 0-10 score. A return
        of -3% or less maps to 0, 0% maps to 5, and +3% or more maps to 10.
        Keeping the mapping bounded prevents the model from dominating the
        indicator engine even when it produces an extreme estimate.
        """
        forecast = state.price_forecast
        experimental_v4 = bool(
            forecast is not None
            and forecast.experimental_blend_enabled
            and forecast.model_version == 'lstm-one-session-priority-v4'
            and forecast.artifact_status == 'VERIFIED_FILES'
            and forecast.technical_blend_weight == 0.2
        )
        if (
            forecast is None
            or forecast.status is not ForecastStatus.TRAINED_MODEL
            or forecast.horizon_sessions != 1
            or forecast.forecast_price is None
            or forecast.current_price <= 0
            or analysis.signal is Signal.UNKNOWN
            or (forecast.policy_eligibility is False and not experimental_v4)
            or forecast.technical_blend_weight == 0
        ):
            return analysis, False

        # None preserves the previously selected V2 policy. New artifacts must
        # declare eligibility separately from merely being a trained model.
        v3_contract = 'v3' in forecast.model_version.casefold()
        if (v3_contract or forecast.validation_status != 'LEGACY_NOT_REVALIDATED') and forecast.policy_eligibility is not True and not experimental_v4:
            return analysis, False
        if v3_contract and forecast.technical_blend_weight is None:
            return analysis, False
        lstm_weight = 0.2 if forecast.technical_blend_weight is None else min(0.2, forecast.technical_blend_weight)
        indicator_weight = 1.0 - lstm_weight
        return_scale = 0.03
        expected_return = forecast.forecast_price / forecast.current_price - 1.0
        bounded_return = max(-return_scale, min(return_scale, expected_return))
        lstm_score = 5.0 + 5.0 * bounded_return / return_scale
        indicator_score = analysis.score
        blended_score = max(
            0.0,
            min(10.0, indicator_weight * indicator_score + lstm_weight * lstm_score),
        )
        signal = (
            Signal.BUY
            if blended_score >= buy_threshold
            else Signal.HOLD
            if blended_score >= 4.0
            else Signal.SELL
        )

        diagnostics = dict(analysis.diagnostics)
        diagnostics.update({
            "indicator_score_before_lstm": round(indicator_score, 4),
            "lstm_score": round(lstm_score, 4),
            "lstm_expected_return": round(expected_return, 8),
            "lstm_weight": lstm_weight,
            "indicator_weight": indicator_weight,
            "lstm_return_scale": return_scale,
            "lstm_model_version": forecast.model_version,
            "lstm_experimental_blend": experimental_v4,
            "lstm_validation_status": forecast.validation_status,
        })
        contributions = dict(analysis.feature_contributions)
        contributions["lstm_blend_delta"] = round(blended_score - indicator_score, 4)
        evidence = [
            *analysis.evidence,
            EvidenceItem(
                label=f"LSTM {lstm_weight:.0%} blend",
                value=(
                    f"Expected next-session return {expected_return:.2%}; "
                    f"LSTM score {lstm_score:.2f}/10; final blended score {blended_score:.2f}/10"
                ),
                input_fields=["price_forecast.current_price", "price_forecast.forecast_price"],
                source_ids=[source.source_id for source in state.market_data.source_refs]
                if state.market_data else [],
                as_of=forecast.as_of,
                confidence=forecast.confidence,
            ),
        ]
        return analysis.model_copy(update={
            "signal": signal,
            "score": round(blended_score, 2),
            "evidence": evidence,
            "tools_used": list(dict.fromkeys([
                *analysis.tools_used,
                "lstm_price_forecast_blend",
            ])),
            "reasoning_summary": (
                f"{analysis.reasoning_summary} Final score uses {indicator_weight:.0%} indicator evidence "
                f"and {lstm_weight:.0%} LSTM next-session signal, resulting in {blended_score:.2f}/10 ({signal.value})."
            ),
            "policy_version": "technical-v3-lstm20",
            "feature_contributions": contributions,
            "diagnostics": diagnostics,
        }), True

    def _request_indicator_plan(
        self,
        state: AgentState,
        source_ids: list[str],
    ) -> tuple[list[str], str, list[str]] | None:
        if self.llm_advisor is None:
            return None
        bars = state.market_data.bars if state.market_data is not None else []
        facts = {
            "ticker": state.ticker,
            "horizon": state.investment_horizon,
            "data_agent": {
                "bar_count": len(bars),
                "latest_close": bars[-1].close if bars else None,
                "as_of": state.market_data.as_of.isoformat() if state.market_data else None,
                "warnings": state.market_data.warnings if state.market_data else [],
            },
            "forecast_agent": state.price_forecast.model_dump(mode="json") if state.price_forecast else None,
        }
        plan = self.llm_advisor.plan_technical_indicators(facts, source_ids)
        if plan is None:
            return None
        return plan.selected_indicators, plan.rationale, plan.cited_source_ids

    @staticmethod
    def _existing_indicator_plan(state: AgentState) -> tuple[list[str], str, list[str]] | None:
        for review in reversed(state.llm_reviews):
            if review.agent is AgentName.TECHNICAL and review.model_role == "indicator_planner":
                return list(review.supporting_observations), review.assessment, list(review.cited_source_ids)
        return None

    def _run_requested_indicators(
        self,
        state: AgentState,
        closes: list[float],
        volumes: list[float],
        requested: list[str],
        timeframe: str,
        profile: HorizonProfile,
    ) -> tuple[TechnicalAnalysis, list[TraceEvent]]:
        selected = [item for item in requested if item != "CANDLESTICK"]
        score = 5.0
        contributions: dict[str, float] = {}
        evidence: list[EvidenceItem] = []
        risks: list[str] = []
        highs = [bar.high for bar in state.market_data.bars]
        lows = [bar.low for bar in state.market_data.bars]
        traces = [self.event("REASON", "SELECT_TOOLS", f"Query requests only: {', '.join(selected)}")]
        for indicator in selected:
            if indicator == "RSI":
                value = self.tools.rsi(closes, profile.rsi_period)
                evidence.append(self._evidence(state, f"RSI{profile.rsi_period}", f"{value:.1f}", ["bars.close"]))
                contributions["RSI"] = 2.0 if 50 <= value <= 70 else -2.0 if value < 35 else -1.0 if value > 75 else 0.0
                score += contributions["RSI"]
                if value < 35:
                    risks.append("RSI shows weak/oversold momentum")
                elif value > 75:
                    risks.append("RSI is potentially overbought")
            elif indicator == "MACD":
                value = self.tools.macd(closes, profile.macd_fast, profile.macd_slow, profile.macd_signal)
                evidence.append(self._evidence(state, "MACD", f"{value['macd']:.3f}; signal {value['signal']:.3f}; histogram {value['histogram']:.3f}", ["bars.close"]))
                contributions["MACD"] = 2.0 if value["histogram"] > 0 else -2.0
                score += contributions["MACD"]
            elif indicator.startswith("MA") and indicator[2:].isdigit():
                window = int(indicator[2:])
                if len(closes) >= window:
                    value = self.tools.sma(closes, window)
                    evidence.append(self._evidence(state, indicator, f"Close {closes[-1]:.2f}; {indicator} {value:.2f}", ["bars.close"]))
                    contributions[indicator] = 1.0 if closes[-1] > value else -1.0
                    score += contributions[indicator]
                else:
                    risks.append(f"Not enough sessions for {indicator}")
            elif indicator == "VOLUME":
                value = self.tools.volume_ratio(volumes)
                evidence.append(self._evidence(state, "Volume ratio", f"{value:.2f}", ["bars.volume"]))
                contributions["VOLUME"] = 0.5 if value > 1.2 else 0.0
                score += contributions["VOLUME"]
            elif indicator == "ATR":
                value = self.tools.average_true_range(highs, lows, closes, profile.rsi_period)
                percent = value / closes[-1] if closes[-1] else 0.0
                evidence.append(self._evidence(
                    state, f"ATR{profile.rsi_period}", f"{value:.2f} ({percent:.1%} of close)",
                    ["bars.high", "bars.low", "bars.close"],
                ))
                contributions["ATR"] = 0.0
                if percent > 0.05:
                    risks.append("ATR exceeds 5% of the latest close, indicating wide daily ranges")
            elif indicator == "ADX":
                value = self.tools.adx(highs, lows, closes, profile.rsi_period)
                evidence.append(self._evidence(
                    state, f"ADX{profile.rsi_period}", f"{value:.1f}",
                    ["bars.high", "bars.low", "bars.close"],
                ))
                contributions["ADX"] = 0.0
                if value < 20:
                    risks.append("ADX indicates that the current trend is weak")
            traces.append(self.event("ACT", "CALL_TOOL", f"Calculated {indicator}", indicator))
        if not selected:
            selected = ["RSI"]
            value = self.tools.rsi(closes, profile.rsi_period)
            evidence.append(self._evidence(state, f"RSI{profile.rsi_period}", f"{value:.1f}", ["bars.close"]))
            traces.append(self.event("ACT", "CALL_TOOL", "Calculated RSI", "RSI"))
        score = max(0.0, min(10.0, score))
        signal = Signal.BUY if score >= 6.5 else Signal.HOLD if score >= 4.0 else Signal.SELL
        return TechnicalAnalysis(
            signal=signal, score=round(score, 2), confidence=0.7,
            evidence=evidence, risks=risks, tools_used=selected,
            reasoning_summary=f"The requested indicator set ({', '.join(selected)}) implies {signal.value}; score {score:.1f}/10.",
            timeframe=timeframe, horizon_profile=profile.name,
            feature_contributions=contributions,
            diagnostics={"price_adjustment": state.market_data.price_adjustment.value if state.market_data else None},
        ), traces

    @staticmethod
    def _minimum_traded_value() -> float:
        try:
            return max(0.0, float(os.getenv("STOCK_AGENT_MIN_AVG_TRADED_VALUE_VND", "1000000000")))
        except ValueError:
            return 1_000_000_000.0

    @staticmethod
    def _relative_benchmark_return(state: AgentState, window: int) -> tuple[float | None, str]:
        if state.market_data is None or state.benchmark_data is None:
            return None, "Relative strength is UNKNOWN because no benchmark data was supplied."
        issuer_prices = {bar.date: bar.close for bar in state.market_data.bars}
        benchmark_prices = {bar.date: bar.close for bar in state.benchmark_data.bars}
        common_dates = sorted(set(issuer_prices).intersection(benchmark_prices))
        if len(common_dates) <= window:
            return None, "Relative strength is UNKNOWN because benchmark dates do not cover the selected window."
        start, end = common_dates[-1 - window], common_dates[-1]
        issuer_return = issuer_prices[end] / issuer_prices[start] - 1
        benchmark_return = benchmark_prices[end] / benchmark_prices[start] - 1
        return issuer_return - benchmark_return, f"Relative return versus VNINDEX is based on {window} aligned sessions."

    @staticmethod
    def _benchmark_evidence(state: AgentState, relative_return: float, window: int) -> EvidenceItem:
        benchmark = state.benchmark_data
        return EvidenceItem(
            label="Relative return versus VNINDEX",
            value=f"{relative_return:.1%} over {window} aligned sessions",
            input_fields=["bars.close", "benchmark_data.bars.close"],
            source_ids=[source.source_id for source in benchmark.source_refs if source.data_kind.value == "OHLCV"] if benchmark else [],
            as_of=benchmark.as_of if benchmark else None,
            confidence=1.0,
        )

    @staticmethod
    def _evidence(state: AgentState, label: str, value: str, fields: list[str]) -> EvidenceItem:
        data = state.market_data
        return EvidenceItem(
            label=label, value=value, input_fields=fields,
            source_ids=[source.source_id for source in data.source_refs if source.data_kind.value == "OHLCV"] if data else [],
            as_of=data.as_of if data else None, confidence=1.0,
        )
