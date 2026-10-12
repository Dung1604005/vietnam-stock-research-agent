import pytest

from stock_multi_agent.tools.technical_indicators import TechnicalIndicatorToolkit


def test_sma_and_rsi() -> None:
    tools = TechnicalIndicatorToolkit()
    values = list(range(1, 31))
    assert tools.sma(values, 5) == pytest.approx(28.0)
    assert tools.rsi(values, 14) == pytest.approx(100.0)


def test_macd_volatility_drawdown() -> None:
    tools = TechnicalIndicatorToolkit()
    values = [100 + index * 0.5 + (index % 3) for index in range(80)]
    macd = tools.macd(values)
    assert set(macd) == {"macd", "signal", "histogram"}
    assert tools.annualized_volatility(values) >= 0
    assert tools.max_drawdown([100, 120, 90, 110]) == pytest.approx(0.25)


def test_indicator_rejects_insufficient_data() -> None:
    with pytest.raises(ValueError):
        TechnicalIndicatorToolkit().sma([1, 2], 3)


def test_atr_and_adx_are_finite_deterministic_values() -> None:
    tools = TechnicalIndicatorToolkit()
    closes = [100.0 + index * 0.4 + (0.2 if index % 2 else -0.2) for index in range(40)]
    highs = [value + 1.0 for value in closes]
    lows = [value - 1.0 for value in closes]

    atr = tools.average_true_range(highs, lows, closes, 14)
    adx = tools.adx(highs, lows, closes, 14)

    assert atr > 0
    assert 0 <= adx <= 100

