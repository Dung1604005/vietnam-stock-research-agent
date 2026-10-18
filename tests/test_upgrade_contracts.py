from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from stock_multi_agent.models.research_contracts import AgentFinding, Evidence, FinalAnswer, RequestBudget
from stock_multi_agent.services.request_telemetry import BudgetExceeded, current_metrics, observed, request_scope


def test_quantitative_finding_requires_known_evidence():
    with pytest.raises(ValidationError, match="evidence IDs"):
        AgentFinding(agent="technical", claim="RSI 60", confidence=.8, data_quality="GOOD", quantitative=True)
    finding = AgentFinding(agent="technical", claim="RSI 60", evidence_ids=["missing"], confidence=.8, data_quality="GOOD", quantitative=True)
    with pytest.raises(ValidationError, match="unknown evidence"):
        FinalAnswer(status="PARTIAL", text="Test", findings=[finding])
    with pytest.raises(ValidationError, match="require evidence"):
        FinalAnswer(status="GROUNDED", text="Test")


def test_evidence_rejects_naive_timestamp():
    with pytest.raises(ValidationError, match="timezone"):
        Evidence(evidence_id="E1", source_url="https://example.invalid", publisher="Test", fetched_at=datetime(2025, 1, 1),
                 quote="Test", content_sha256="a" * 64)


def test_request_budget_blocks_before_provider_and_resets_context(monkeypatch):
    monkeypatch.setenv("STOCK_AGENT_REQUEST_TELEMETRY_ENABLED", "false")
    calls = []
    @observed("test.provider", "llm")
    def provider(secret):
        calls.append(secret)
        return "done"
    with request_scope(RequestBudget(max_llm_calls=1)) as metrics:
        assert provider("do-not-log-this") == "done"
        with pytest.raises(BudgetExceeded):
            provider("not-called")
        assert metrics.llm_calls == 1
        assert "do-not-log-this" not in json.dumps(metrics.snapshot())
    assert current_metrics() is None
    assert calls == ["do-not-log-this"]


def test_telemetry_persists_only_allowlisted_metadata(monkeypatch):
    root = Path(__file__).resolve().parents[1] / ".runtime/test-artifacts/upgrade" / str(uuid4())
    monkeypatch.setenv("STOCK_AGENT_RUNTIME_DIR", str(root))
    monkeypatch.setenv("STOCK_AGENT_REQUEST_TELEMETRY_ENABLED", "true")
    with request_scope() as metrics:
        metrics.source_count = 2
    with sqlite3.connect(root / "telemetry/requests.sqlite3") as connection:
        value = json.loads(connection.execute("SELECT metrics_json FROM requests").fetchone()[0])
    assert value["source_count"] == 2
    assert "query" not in value
    assert "token_usage" in value and value["token_usage"] is None
