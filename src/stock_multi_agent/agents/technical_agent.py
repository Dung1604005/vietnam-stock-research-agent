from __future__ import annotations

import os

from stock_multi_agent.agents.base_agent import BaseAgent
from stock_multi_agent.models.analysis import EvidenceItem, PriceAdjustment, Signal, TechnicalAnalysis
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

    def __init__(self, toolkit: TechnicalIndicatorToolkit) -> None:
        self.tools = toolkit

    def run(self, state: AgentState, extended: bool = False) -> dict[str, object]:
        if state.market_data is None:
            raise ValueError("Technical Agent requires market data")
        closes = [bar.close for bar in state.market_data.bars]
        volumes = [bar.volume for bar in state.market_data.bars]
        profile = LONG_HORIZON if extended else self.tools.horizon_profile(state.investment_horizon)
        timeframe = f"extended {state.investment_horizon} ({profile.name} profile)" if extended else f"{state.investment_horizon} ({profile.name} profile)"
        traces: list[TraceEvent] = [self.event(
            "OBSERVE", "READ_MARKET_DATA", f"Observe {len(closes)} OHLCV bars; profile={profile.name}",
        )]
        required_history = max(50, profile.slow_window + profile.macd_signal)
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
        else:
            analysis, extra_traces = self._run_default(state, closes, volumes, timeframe, profile)
            traces.extend(extra_traces)

        message = self.message(
            AgentName.MANAGER, MessageType.RESULT,
            "Technical re-analysis complete" if extended else "Technical analysis complete",
            analysis.model_dump(mode="json"), analysis.confidence,
        )
        traces.append(self.event("ACT", "SEND_RESULT", f"Technical signal={analysis.signal.value}, confidence={analysis.confidence:.2f}"))
        return {
            "technical_result": analysis,
            "messages": [message],
            "trace": traces,
            "completed_tasks": ["technical_reanalysis" if extended else "technical_analysis"],
        }

    def _run_default(
        self,
        state: AgentState,
        closes: list[float],
        volumes: list[float],
        timeframe: str,
        profile: HorizonProfile,
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
