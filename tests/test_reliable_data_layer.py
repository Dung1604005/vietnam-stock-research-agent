from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from conftest import conflict_market_data
from stock_multi_agent.agents.fundamental_agent import FundamentalAgent
from stock_multi_agent.models.analysis import FinancialMetrics, Signal
from stock_multi_agent.models.data import FinancialMetricObservation, PriceAdjustment, QualityStatus
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.providers.data_store import RawSnapshotStore, SQLiteMarketDataCache
from stock_multi_agent.providers.market_provider import VnstockMarketDataProvider
from stock_multi_agent.providers.resilience import CircuitOpenError, RateLimiter, ResiliencePolicy, ResilientExecutor
from stock_multi_agent.services.data_quality import DataQualityChecker
from stock_multi_agent.tools.financial_data import FinancialToolkit


TEST_ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / ".runtime" / "test-artifacts"


def test_quality_checker_fails_closed_without_provenance() -> None:
    data = conflict_market_data().model_copy(update={"source_refs": []})
    report = DataQualityChecker().evaluate(data, require_financials=True, today=data.bars[-1].date)
    assert report.status is QualityStatus.FAIL
    assert any(issue.code == "MISSING_SOURCE_REF" for issue in report.issues)


def test_quality_checker_records_unknown_adjustment_without_inventing_policy() -> None:
    data = conflict_market_data().model_copy(update={
        "price_adjustment": PriceAdjustment.UNKNOWN,
    })
    report = DataQualityChecker().evaluate(data, require_financials=True, today=data.bars[-1].date)
    assert report.status is QualityStatus.WARN
    assert any(issue.code == "UNKNOWN_PRICE_ADJUSTMENT" for issue in report.issues)


def test_stale_financial_period_is_used_as_explicit_low_confidence_fallback() -> None:
    data = conflict_market_data().model_copy(update={
        "financial_observations": [FinancialMetricObservation(
            metric="pe",
            value=18.0,
            unit="multiple",
            period_label="2018",
            period_end=date(2018, 12, 31),
            source_id="VCI:Finance.ratio:FPT",
        )],
    })
    report = DataQualityChecker().evaluate(data, require_financials=True, today=date(2026, 10, 9))
    assert any(issue.code == "STALE_FINANCIALS" for issue in report.issues)
    data = data.model_copy(update={"quality_report": report})
    state = AgentState(ticker="FPT", investment_horizon="3 months", user_goal="Analyze FPT", market_data=data)
    result = FundamentalAgent(FinancialToolkit()).run(state)["fundamental_result"]
    assert result.signal is not Signal.UNKNOWN
    assert result.data_mode == 'LATEST_AVAILABLE'
    assert result.metric_periods == {'pe': date(2018, 12, 31)}
    assert result.confidence < 0.5
    assert result.policy_version == "fundamental-v3-quality-weighted"
    assert result.feature_contributions['P/E'] == pytest.approx(0.7)
    assert data.financials.available_count() > 0  # Historical raw values are retained.
    assert all(item.confidence <= 0.25 for item in result.evidence)
    assert any("STALE_FINANCIALS" in weakness for weakness in result.weaknesses)


def test_sparse_unperioded_metrics_are_scored_with_low_confidence() -> None:
    data = conflict_market_data().model_copy(update={
        "financials": FinancialMetrics(pe=14.0, roe=0.18),
        "financial_observations": [],
        "financial_statement_observations": [],
    })
    report = DataQualityChecker().evaluate(data, require_financials=True, today=data.as_of)
    assert {issue.code for issue in report.issues}.issuperset({
        "INSUFFICIENT_FINANCIALS", "MISSING_FINANCIAL_PERIOD",
    })

    result = FundamentalAgent(FinancialToolkit()).run(AgentState(
        ticker="FPT", investment_horizon="3 months", user_goal="Analyze FPT",
        market_data=data.model_copy(update={"quality_report": report}),
    ))["fundamental_result"]

    assert result.signal is not Signal.UNKNOWN
    assert result.confidence < 0.3
    assert 5.0 < result.score < 6.0
    assert {item.label for item in result.evidence} == {"P/E", "ROE"}


def test_sqlite_cache_round_trip_and_expiry() -> None:
    cache = SQLiteMarketDataCache(TEST_ARTIFACT_ROOT / "cache" / "market.sqlite3")
    data = conflict_market_data()
    cache.set("fresh", data, ttl_seconds=60)
    assert cache.get("fresh").ticker == "FPT"
    cache.set("expired", data, ttl_seconds=0)
    assert cache.get("expired") is None


def test_raw_snapshots_are_content_addressed_and_immutable() -> None:
    store = RawSnapshotStore(TEST_ARTIFACT_ROOT / "snapshots")
    fetched_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    first = store.save("VCI", "OHLCV", "FPT", {"rows": [1, 2]}, fetched_at, {"interval": "1D"})
    second = store.save("VCI", "OHLCV", "FPT", {"rows": [1, 2]}, fetched_at, {"interval": "1D"})
    assert first == second
    assert (TEST_ARTIFACT_ROOT / "snapshots" / first[1]).exists()


def test_resilient_executor_retries_then_succeeds() -> None:
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("temporary")
        return "ok"

    executor = ResilientExecutor(
        ResiliencePolicy(max_attempts=3, base_delay_seconds=0, circuit_failure_threshold=2, min_request_interval_seconds=0),
        sleeper=lambda _: None,
    )
    assert executor.run(operation) == "ok"
    assert attempts == 3
    assert executor.health().retries_total == 2
    assert executor.health().attempts_total == 3


def test_circuit_opens_after_repeated_provider_failure() -> None:
    executor = ResilientExecutor(
        ResiliencePolicy(max_attempts=1, base_delay_seconds=0, circuit_failure_threshold=1, circuit_cooldown_seconds=60, min_request_interval_seconds=0),
        sleeper=lambda _: None,
    )
    with pytest.raises(RuntimeError, match="down"):
        executor.run(lambda: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(CircuitOpenError):
        executor.run(lambda: "unreachable")


def test_explicit_provider_rate_limit_is_not_retried() -> None:
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("Rate Limit Exceeded: wait 59 seconds")

    executor = ResilientExecutor(
        ResiliencePolicy(max_attempts=3, base_delay_seconds=0, min_request_interval_seconds=0),
        sleeper=lambda _: None,
    )
    with pytest.raises(RuntimeError, match="Rate Limit"):
        executor.run(operation)
    assert attempts == 1
    assert executor.health().retries_total == 0


def test_rate_limiter_paces_attempts_deterministically() -> None:
    now = 10.0
    waits: list[float] = []

    def clock() -> float:
        return now

    def sleeper(seconds: float) -> None:
        nonlocal now
        waits.append(seconds)
        now += seconds

    limiter = RateLimiter(0.5, sleeper=sleeper, clock=clock)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()
    assert waits == [0.5, 0.5]
    assert limiter.total_wait_seconds == pytest.approx(1.0)


def test_provider_health_tracks_cache_hits_without_external_calls() -> None:
    cache = SQLiteMarketDataCache(TEST_ARTIFACT_ROOT / "health" / "market.sqlite3")
    provider = VnstockMarketDataProvider(cache=cache)
    key = provider._cache_key("FPT", include_financials=False)
    cache.set(key, conflict_market_data(), ttl_seconds=60)
    provider.get_market_data("FPT", include_financials=False)
    health = provider.health()
    assert health["cache_hits"] == 1
    assert health["cache_misses"] == 0
    assert health["last_success_at"] is not None


def test_provider_uses_community_pacing_when_api_key_is_configured(monkeypatch) -> None:
    monkeypatch.setenv("VNSTOCK_API_KEY", "community-test-key")
    monkeypatch.delenv("STOCK_AGENT_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS", raising=False)
    provider = VnstockMarketDataProvider()
    assert provider.api_key_configured is True
    assert provider.executor.policy.min_request_interval_seconds == pytest.approx(1.05)
    assert provider.health()["api_key_configured"] is True


def test_provider_keeps_guest_pacing_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("VNSTOCK_API_KEY", raising=False)
    monkeypatch.delenv("STOCK_AGENT_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS", raising=False)
    provider = VnstockMarketDataProvider()
    assert provider.api_key_configured is False
    assert provider.executor.policy.min_request_interval_seconds == pytest.approx(3.2)


def test_financial_normalization_keeps_period_unit_and_source() -> None:
    frame = pd.DataFrame(
        [
            ["P/E", "P/E", "pe_ratio", 20.0, 18.0],
            ["ROE", "ROE (%)", "roe", 16.0, 20.0],
        ],
        columns=["item", "item_en", "item_id", "2024", "2025"],
    )
    metrics, observations = VnstockMarketDataProvider._normalize_ratios_with_observations(
        frame, "VCI:Finance.ratio:FPT"
    )
    assert metrics.pe == pytest.approx(18.0)
    pe = next(item for item in observations if item.metric == "pe")
    assert pe.period_label == "2025"
    assert pe.period_end == date(2025, 12, 31)
    assert pe.unit == "multiple"
    assert pe.source_id == "VCI:Finance.ratio:FPT"


def test_financial_normalization_selects_latest_period_not_last_column() -> None:
    frame = pd.DataFrame(
        [["P/E", "P/E", "pe_ratio", 18.0, 20.0]],
        columns=["item", "item_en", "item_id", "2025", "2024"],
    )
    metrics, observations = VnstockMarketDataProvider._normalize_ratios_with_observations(frame, "VCI:FPT")
    assert metrics.pe == pytest.approx(18.0)
    assert observations[0].period_label == "2025"
