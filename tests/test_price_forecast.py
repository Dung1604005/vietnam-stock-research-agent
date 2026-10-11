from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import InMemoryProvider, conflict_market_data
from stock_multi_agent.chat.response import ResponseComposer
from stock_multi_agent.models.forecast import ForecastStatus
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.price_forecast import PriceForecastService
from stock_multi_agent.services.lstm_forecast import LstmV2ForecastModel


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
