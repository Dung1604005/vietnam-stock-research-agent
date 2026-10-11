from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from stock_multi_agent.nlu.evaluation import assert_no_query_overlap, evaluate_dataset
from stock_multi_agent.nlu.telemetry import SemanticTelemetryEvent, SQLiteSemanticTelemetryStore


TEST_ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / ".runtime" / "test-artifacts"


def test_holdout_overlap_is_rejected_after_normalization() -> None:
    regression = [{"query": "Phân tích FPT", "expected_intent": "STOCK_ANALYSIS"}]
    holdout = [{"query": "phan tich fpt", "expected_intent": "STOCK_ANALYSIS"}]
    with pytest.raises(ValueError, match="overlaps"):
        assert_no_query_overlap(regression, holdout)


def test_evaluation_reports_calibration_without_using_provider_or_telemetry() -> None:
    report = evaluate_dataset(
        [{"query": "Phân tích FPT", "expected_intent": "STOCK_ANALYSIS", "expected_tickers": ["FPT"]}],
        "unit", "v1",
    )
    assert report["metrics"]["intent"] == 1.0
    assert report["calibration"]["expected_calibration_error"] is not None
    assert len(report["calibration"]["bins"]) == 4


def test_semantic_telemetry_is_durable_and_contains_no_query_column() -> None:
    database = TEST_ARTIFACT_ROOT / "telemetry" / "semantic.sqlite3"
    store = SQLiteSemanticTelemetryStore(database)
    store.record(SemanticTelemetryEvent(
        parser_source="semantic_similarity",
        parser_latency_ms=4,
        llm_attempted=False,
        llm_latency_ms=None,
        fallback_reason="no_configured_llm",
        ambiguity_count=0,
        needs_clarification=False,
        prompt_version="semantic-system-v1",
        example_bank_version="intent-examples-v1",
        occurred_at=datetime(2026, 10, 9, tzinfo=timezone.utc),
    ))
    summary = store.summary()
    assert summary["events"] >= 1
    assert summary["fallback_reasons"]["no_configured_llm"] >= 1
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(semantic_parse_events)")}
    assert "query" not in columns
    assert "raw_query" not in columns
