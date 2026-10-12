from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HorizonProfile:
    """Deterministic technical settings selected from the requested horizon."""

    name: str
    fast_window: int
    slow_window: int
    rsi_period: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    volatility_window: int
    return_window: int


SHORT_HORIZON = HorizonProfile("short", 10, 20, 7, 6, 19, 5, 20, 20)
MEDIUM_HORIZON = HorizonProfile("medium", 20, 50, 14, 12, 26, 9, 60, 60)
LONG_HORIZON = HorizonProfile("long", 50, 150, 14, 12, 26, 9, 120, 120)


class TechnicalIndicatorToolkit:
    @staticmethod
    def _array(values: Sequence[float], minimum: int = 1) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if len(array) < minimum or not np.isfinite(array).all():
            raise ValueError(f"Need at least {minimum} finite values")
        return array

    def sma(self, values: Sequence[float], window: int) -> float:
        array = self._array(values, window)
        return float(np.mean(array[-window:]))

    def sma_series(self, values: Sequence[float], window: int) -> np.ndarray:
        array = self._array(values, window)
        result = np.full(len(array), np.nan, dtype=float)
        cumulative = np.cumsum(np.insert(array, 0, 0.0))
        result[window - 1:] = (cumulative[window:] - cumulative[:-window]) / window
        return result

    def ema_series(self, values: Sequence[float], span: int) -> np.ndarray:
        array = self._array(values, span)
        alpha = 2.0 / (span + 1)
        result = np.empty_like(array)
        result[0] = array[0]
        for index in range(1, len(array)):
            result[index] = alpha * array[index] + (1 - alpha) * result[index - 1]
        return result

    def rsi(self, values: Sequence[float], period: int = 14) -> float:
        array = self._array(values, period + 1)
        deltas = np.diff(array[-(period + 1):])
        gains = float(np.mean(np.clip(deltas, 0, None)))
        losses = float(np.mean(np.clip(-deltas, 0, None)))
        if losses == 0:
            return 100.0 if gains > 0 else 50.0
        return float(100 - 100 / (1 + gains / losses))

    def rsi_series(self, values: Sequence[float], period: int = 14) -> np.ndarray:
        array = self._array(values, period + 1)
        result = np.full(len(array), np.nan, dtype=float)
        for index in range(period, len(array)):
            result[index] = self.rsi(array[index - period:index + 1], period)
        # Neutral RSI is explicit only for the warm-up span, matching chart
        # behavior while keeping the scalar calculation unchanged.
        return np.nan_to_num(result, nan=50.0)

    def macd(self, values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, float]:
        array = self._array(values, slow + signal)
        line = self.ema_series(array, fast) - self.ema_series(array, slow)
        signal_line = self.ema_series(line, signal)
        return {
            "macd": float(line[-1]),
            "signal": float(signal_line[-1]),
            "histogram": float(line[-1] - signal_line[-1]),
        }

    def macd_series(
        self,
        values: Sequence[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        array = self._array(values, slow + signal)
        line = self.ema_series(array, fast) - self.ema_series(array, slow)
        signal_line = self.ema_series(line, signal)
        return line, signal_line, line - signal_line

    def bollinger_bands(self, values: Sequence[float], window: int = 20, deviations: float = 2.0) -> dict[str, float]:
        array = self._array(values, window)[-window:]
        middle = float(np.mean(array))
        width = deviations * float(np.std(array, ddof=0))
        return {"lower": middle - width, "middle": middle, "upper": middle + width}

    def annualized_volatility(self, values: Sequence[float], trading_days: int = 252) -> float:
        array = self._array(values, 3)
        returns = np.diff(np.log(array))
        return float(np.std(returns, ddof=1) * sqrt(trading_days))

    def max_drawdown(self, values: Sequence[float]) -> float:
        array = self._array(values, 2)
        peaks = np.maximum.accumulate(array)
        drawdowns = array / peaks - 1
        return float(abs(np.min(drawdowns)))

    def average_true_range(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        period: int = 14,
    ) -> float:
        high = self._array(highs, period + 1)
        low = self._array(lows, period + 1)
        close = self._array(closes, period + 1)
        if not (len(high) == len(low) == len(close)):
            raise ValueError("High, low, and close series must have equal length")
        true_ranges = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ])
        return float(np.mean(true_ranges[-period:]))

    def adx(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        period: int = 14,
    ) -> float:
        high = self._array(highs, period * 2 + 1)
        low = self._array(lows, period * 2 + 1)
        close = self._array(closes, period * 2 + 1)
        if not (len(high) == len(low) == len(close)):
            raise ValueError("High, low, and close series must have equal length")
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        true_ranges = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ])
        dx_values: list[float] = []
        for endpoint in range(period, len(true_ranges) + 1):
            atr_sum = float(np.sum(true_ranges[endpoint - period:endpoint]))
            if atr_sum <= 0:
                continue
            plus_di = 100.0 * float(np.sum(plus_dm[endpoint - period:endpoint])) / atr_sum
            minus_di = 100.0 * float(np.sum(minus_dm[endpoint - period:endpoint])) / atr_sum
            denominator = plus_di + minus_di
            dx_values.append(0.0 if denominator == 0 else 100.0 * abs(plus_di - minus_di) / denominator)
        return float(np.mean(dx_values[-period:])) if dx_values else 0.0

    def support_resistance(self, values: Sequence[float], lookback: int = 60) -> dict[str, float]:
        array = self._array(values, min(lookback, 20))[-lookback:]
        return {
            "support": float(np.quantile(array, 0.1)),
            "resistance": float(np.quantile(array, 0.9)),
        }

    def volume_ratio(self, volumes: Sequence[float], recent: int = 5, baseline: int = 20) -> float:
        array = self._array(volumes, baseline)
        average = float(np.mean(array[-baseline:]))
        return float(np.mean(array[-recent:]) / average) if average > 0 else 0.0

    def average_traded_value(self, closes: Sequence[float], volumes: Sequence[float], window: int = 20) -> float:
        close_array = self._array(closes, window)
        volume_array = self._array(volumes, window)
        return float(np.mean(close_array[-window:] * volume_array[-window:]))

    @staticmethod
    def horizon_profile(horizon: str) -> HorizonProfile:
        """Map a human horizon to one versioned parameter set.

        This intentionally has no market forecast semantics. It only controls
        the input windows used by deterministic indicators.
        """
        normalized = horizon.casefold().strip()
        match = re.search(r"(\d+)\s*(day|days|d|week|weeks|w|month|months|m|year|years|y)", normalized)
        if not match:
            return MEDIUM_HORIZON
        amount = int(match.group(1))
        unit = match.group(2)
        approximate_days = amount * (365 if unit.startswith("y") or "year" in unit else 30 if unit.startswith("m") or "month" in unit else 7 if unit.startswith("w") or "week" in unit else 1)
        if approximate_days <= 45:
            return SHORT_HORIZON
        if approximate_days >= 270:
            return LONG_HORIZON
        return MEDIUM_HORIZON

