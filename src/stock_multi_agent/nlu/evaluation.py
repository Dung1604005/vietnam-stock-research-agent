from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from stock_multi_agent.chat.query_parser import QueryParser
from stock_multi_agent.models.chat import ConversationState, QueryIntent
from stock_multi_agent.nlu.normalizer import VietnameseTextNormalizer
from stock_multi_agent.nlu.telemetry import NullSemanticTelemetryStore


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload, "unversioned-list"
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"Dataset {path} must be a JSON list or an object with a cases list.")
    return payload["cases"], str(payload.get("version", "unversioned-object"))


def assert_no_query_overlap(regression: list[dict[str, Any]], holdout: list[dict[str, Any]]) -> None:
    normalizer = VietnameseTextNormalizer()
    regression_queries = {normalizer.normalize(str(item["query"])) for item in regression}
    overlap = sorted(
        {normalizer.normalize(str(item["query"])) for item in holdout}.intersection(regression_queries)
    )
    if overlap:
        sample = ", ".join(overlap[:3])
        raise ValueError(f"Holdout overlaps with regression queries after normalization: {sample}")


def _context(payload: dict[str, Any]) -> ConversationState | None:
    raw = payload.get("context")
    if not raw:
        return None
    values = dict(raw)
    if isinstance(values.get("last_intent"), str):
        values["last_intent"] = QueryIntent(values["last_intent"])
    return ConversationState(**values)


def _horizon_code(value: str | None) -> str | None:
    if not value:
        return None
    replacements = {" months": "M", " month": "M", " years": "Y", " year": "Y"}
    lowered = value.lower()
    for suffix, code in replacements.items():
        if lowered.endswith(suffix):
            return lowered.removesuffix(suffix).strip() + code
    return value.upper()


def _calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
    bins = [(0.0, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.01)]
    report: list[dict[str, Any]] = []
    ece = 0.0
    for lower, upper in bins:
        selected = [record for record in records if lower <= record["confidence"] < upper]
        if not selected:
            report.append({"range": f"[{lower:.2f}, {upper:.2f})", "cases": 0, "accuracy": None, "mean_confidence": None})
            continue
        accuracy = sum(record["intent_correct"] for record in selected) / len(selected)
        confidence = sum(record["confidence"] for record in selected) / len(selected)
        ece += abs(accuracy - confidence) * len(selected) / len(records)
        report.append({
            "range": f"[{lower:.2f}, {upper:.2f})",
            "cases": len(selected),
            "accuracy": round(accuracy, 4),
            "mean_confidence": round(confidence, 4),
        })
    return {"bins": report, "expected_calibration_error": round(ece, 4) if records else None}


def evaluate_dataset(cases: list[dict[str, Any]], dataset_name: str, version: str) -> dict[str, Any]:
    parser = QueryParser(use_configured_provider=False, telemetry_store=NullSemanticTelemetryStore())
    counters: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    for case in cases:
        parsed = parser.parse(case["query"], _context(case))
        checks: dict[str, bool] = {
            "intent": parsed.intent.value == case["expected_intent"],
            "ticker": parsed.tickers == case.get("expected_tickers", []),
        }
        if "expected_period" in case:
            checks["period"] = parsed.chart_period == case["expected_period"]
        if "expected_horizon" in case:
            checks["period"] = _horizon_code(parsed.investment_horizon) == case["expected_horizon"]
        if "context" in case or "expected_follow_up" in case:
            checks["follow_up"] = parsed.follow_up == case.get("expected_follow_up", True)
        if "expected_needs_clarification" in case:
            checks["clarification"] = parsed.needs_clarification == case["expected_needs_clarification"]
        if "expected_secondary_intents" in case:
            actual_secondary = [value.value for value in parsed.secondary_intents]
            checks["multi_intent"] = set(case["expected_secondary_intents"]).issubset(actual_secondary)
        for metric, passed in checks.items():
            totals[metric] += 1
            counters[metric] += int(passed)
        calibration_records.append({"confidence": parsed.confidence, "intent_correct": checks["intent"]})
        if not all(checks.values()):
            errors.append({
                "query": case["query"], "expected": case["expected_intent"],
                "actual": parsed.intent.value, "confidence": parsed.confidence,
                "failed": [name for name, passed in checks.items() if not passed],
            })
    return {
        "dataset": dataset_name,
        "version": version,
        "cases": len(cases),
        "metrics": {name: counters[name] / totals[name] for name in totals},
        "counts": {name: {"passed": counters[name], "total": totals[name]} for name in totals},
        "calibration": _calibration(calibration_records),
        "errors": errors,
    }


def write_report(report: dict[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination
