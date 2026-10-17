from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import InMemoryProvider, conflict_market_data
from stock_multi_agent.agents.technical_agent import TechnicalAgent
from stock_multi_agent.chat.response import ResponseComposer
from stock_multi_agent.models.forecast import ForecastStatus, PriceForecast
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.price_forecast import PriceForecastService
from stock_multi_agent.services.lstm_forecast import LstmV2ForecastModel
from stock_multi_agent.tools.technical_indicators import TechnicalIndicatorToolkit


class _FakeKerasModel:
    def predict(self, inputs, verbose=0):
        del verbose
        sequence, ticker_id = inputs
        assert sequence.shape == (1, 30, 22)
        assert ticker_id.shape == (1,)
        return np.zeros((1, 2), dtype=np.float32)


def test_untrained_forecast_baseline_is_explicit_and_has_current_price() -> None:
    data = conflict_market_data("FPT")
    forecast = PriceForecastService().forecast_next_session(data)

    assert forecast.status is ForecastStatus.BASELINE_UNTRAINED
    assert forecast.current_price == data.bars[-1].close
    assert forecast.forecast_price is not None
    assert forecast.lower_bound <= forecast.forecast_price <= forecast.upper_bound
    assert forecast.confidence == 0
    assert "No trained forecasting model" in forecast.warnings[0]


def test_full_analysis_includes_price_and_forecast_contract() -> None:
    service = AnalysisService(provider=InMemoryProvider(conflict_market_data("FPT")))
    state = service.analyze("FPT", max_iterations=0)
    response = ResponseComposer.full_analysis(state)
    assert "độ tin cậy" not in response.message.lower()

    assert state.price_forecast is not None
    assert "Giá hiện tại và ước tính phiên kế tiếp" in response.message
    assert response.metadata["price_forecast"]["status"] == "BASELINE_UNTRAINED"
    assert response.metadata["price_forecast"]["current_price"] == pytest.approx(state.market_data.bars[-1].close)


def test_selective_single_ticker_analysis_also_includes_price_forecast() -> None:
    service = AnalysisService(provider=InMemoryProvider(conflict_market_data("FPT")))
    state = service.analyze_technical("FPT")
    response = ResponseComposer.technical(state)

    assert "Giá hiện tại và ước tính phiên kế tiếp" in response.message
    assert response.metadata["price_forecast"]["ticker"] == "FPT"


def test_finalized_v2_adapter_builds_declared_features_and_trained_forecast() -> None:
    artifact_dir = Path(__file__).resolve().parents[1] / "notebooks" / "trainResult"
    adapter = LstmV2ForecastModel(artifact_dir, keras_model=_FakeKerasModel())

    forecast = PriceForecastService(trained_model=adapter).forecast_next_session(
        conflict_market_data("FPT")
    )

    assert forecast.status is ForecastStatus.TRAINED_MODEL
    assert forecast.model_version == "lstm-multihorizon-v2"
    assert forecast.horizon_sessions == 1
    assert forecast.forecast_price is not None
    assert forecast.lower_bound <= forecast.forecast_price <= forecast.upper_bound
    assert forecast.confidence == pytest.approx(0.10)
    assert any("train-mean proxies" in warning for warning in forecast.warnings)


def test_trained_lstm_contributes_exactly_twenty_percent_to_technical_score() -> None:
    data = conflict_market_data("FPT")
    agent = TechnicalAgent(TechnicalIndicatorToolkit())
    baseline = agent.run(
        AgentState(ticker="FPT", investment_horizon="3 months", user_goal="Analyze FPT", market_data=data),
        allow_llm_planning=False,
    )["technical_result"]
    forecast = PriceForecast(
        ticker="FPT",
        as_of=data.bars[-1].date,
        current_price=data.bars[-1].close,
        forecast_price=data.bars[-1].close * 1.03,
        status=ForecastStatus.TRAINED_MODEL,
        method="test model",
        model_version="test-lstm",
        confidence=0.10,
    )
    blended = agent.run(
        AgentState(
            ticker="FPT",
            investment_horizon="3 months",
            user_goal="Analyze FPT",
            market_data=data,
            price_forecast=forecast,
        ),
        allow_llm_planning=False,
    )["technical_result"]

    assert blended.score == pytest.approx(round(0.80 * baseline.score + 0.20 * 10.0, 2))
    assert blended.diagnostics["indicator_score_before_lstm"] == pytest.approx(baseline.score)
    assert blended.diagnostics["lstm_weight"] == pytest.approx(0.20)
    assert blended.feature_contributions["lstm_blend_delta"] == pytest.approx(
        blended.score - baseline.score
    )
    assert blended.policy_version == "technical-v3-lstm20"
    assert "lstm_price_forecast_blend" in blended.tools_used


def test_untrained_forecast_does_not_change_technical_score() -> None:
    data = conflict_market_data("FPT")
    agent = TechnicalAgent(TechnicalIndicatorToolkit())
    state = AgentState(
        ticker="FPT",
        investment_horizon="3 months",
        user_goal="Analyze FPT",
        market_data=data,
    )
    without_forecast = agent.run(state, allow_llm_planning=False)["technical_result"]
    state.price_forecast = PriceForecastService().forecast_next_session(data)
    with_baseline = agent.run(state, allow_llm_planning=False)["technical_result"]

    assert with_baseline.score == without_forecast.score
    assert with_baseline.signal == without_forecast.signal
    assert "lstm_weight" not in with_baseline.diagnostics
    assert "lstm_price_forecast_blend" not in with_baseline.tools_used
