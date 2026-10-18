"""One-command, network-free upgrade evaluation; synthetic results are labelled."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import gc
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/upgrade/benchmark-v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/evaluations/upgrade-latest.json")
    parser.add_argument("--source-root", type=Path, help="Use src from a frozen checkout to measure baseline")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any measured gate fails")
    args = parser.parse_args()
    if args.source_root:
        sys.path.insert(0, str(args.source_root.resolve() / "src"))
    os.environ["STOCK_AGENT_SEMANTIC_TELEMETRY_ENABLED"] = "false"
    os.environ["STOCK_AGENT_REQUEST_TELEMETRY_ENABLED"] = "false"
    from stock_multi_agent.chat.query_parser import QueryParser
    from stock_multi_agent.chat.router import IntentRouter, RoutingPlanValidator
    from stock_multi_agent.models.analysis import TechnicalAnalysis, FundamentalAnalysis, RiskAnalysis
    from stock_multi_agent.models.state import AgentState
    from stock_multi_agent.agents.manager_agent import ManagerAgent
    from stock_multi_agent.models.research import ResearchDocumentType
    from stock_multi_agent.services.embeddings import LocalHashEmbeddingProvider
    from stock_multi_agent.services.research_store import ResearchStore

    payload_bytes = args.dataset.read_bytes()
    data = json.loads(payload_bytes)
    nlu = QueryParser(use_configured_provider=False)
    router = IntentRouter()
    errors, latencies = [], []
    counts = {name: [0, 0] for name in ["intent", "ticker", "horizon", "requested_count", "valid_plan", "single_stock_route", "conflict_action"]}
    def record(metric, correct, case):
        counts[metric][1] += 1
        counts[metric][0] += bool(correct)
        if not correct:
            errors.append({"metric": metric, "case": case})

    for case in data["nlu"]:
        started = time.perf_counter()
        parsed = nlu.parse(case["query"])
        latencies.append((time.perf_counter() - started) * 1000)
        record("intent", parsed.intent.value == case["expected_intent"], {"query": case["query"], "actual": parsed.intent.value})
        record("ticker", parsed.tickers == case["expected_tickers"], {"query": case["query"], "actual": parsed.tickers})
        if "expected_horizon" in case:
            record("horizon", parsed.investment_horizon == case["expected_horizon"], case["query"])
        try:
            plan = router.plan(parsed)
            record("valid_plan", not RoutingPlanValidator.validate(plan, parsed), case["query"])
        except Exception as exc:
            record("valid_plan", False, {"query": case["query"], "error": type(exc).__name__})

    for case in data["screening"]:
        parsed = nlu.parse(case["query"])
        record("requested_count", parsed.requested_result_count == case["expected_count"], case["query"])
        record("horizon", parsed.investment_horizon == case["expected_horizon"], case["query"])
    for case in data["single_stock"]:
        parsed = nlu.parse(case["query"])
        try:
            steps = [step.value for step in router.plan(parsed).steps]
        except ValueError:
            steps = []
        record("single_stock_route", parsed.tickers == [case["expected_ticker"]] and "TECHNICAL" in steps and "RISK" in steps, case["query"])
    for case in data["conflicts"]:
        state = AgentState(ticker=case["ticker"], investment_horizon="3 months", user_goal="Synthetic conflict evaluation",
            technical_result=TechnicalAnalysis(signal=case["technical"], confidence=case["technical_confidence"], score=5,
                evidence=[], risks=[], tools_used=[], reasoning_summary="Synthetic", timeframe="3 months"),
            fundamental_result=FundamentalAnalysis(signal=case["fundamental"], confidence=case["fundamental_confidence"], score=5,
                strengths=[], weaknesses=[], evidence=[], reasoning_summary="Synthetic"),
            risk_result=RiskAnalysis(risk_level=case["risk"], risk_score=5, criticisms=[], warnings=[], veto=False, reasoning_summary="Synthetic"))
        matches = [item for item in ManagerAgent.assess_conflicts(state) if item.kind.value == case["expected_kind"]]
        record("conflict_action", bool(matches) and matches[0].suggested_action.value == case["expected_action"], case["id"])

    runtime = ROOT / ".runtime/evaluations"
    runtime.mkdir(parents=True, exist_ok=True)
    hits, reciprocal, answerable, abstained, negatives = 0, 0.0, 0, 0, 0
    with tempfile.TemporaryDirectory(prefix="upgrade-", dir=runtime) as temporary:
        store = ResearchStore(Path(temporary) / "research.sqlite3", embedding_provider=LocalHashEmbeddingProvider())
        ids = {}
        for document in data["documents"]:
            ingested = store.ingest_text(text=document["text"], title=document["title"], document_type=ResearchDocumentType.OTHER,
                source_url=document["source_url"], publisher=document["publisher"], published_at=datetime.fromisoformat(document["published_at"]))
            ids[document["key"]] = ingested.document_id
        for case in data["rag"]:
            bundle = store.search(case["query"], ticker=case.get("ticker"), limit=5)
            expected = {ids[key] for key in case["expected_document_keys"]}
            if expected:
                answerable += 1
                ranks = [index for index, item in enumerate(bundle.citations, 1) if item.document_id in expected]
                hits += bool(ranks)
                reciprocal += 1 / min(ranks) if ranks else 0
                if not ranks:
                    errors.append({"metric": "recall_at_5", "case": case["id"]})
            else:
                negatives += 1
                abstained += not bundle.citations
        # Frozen legacy stores used connection context managers without close().
        # Collect their cycles before TemporaryDirectory cleanup on Windows.
        gc.collect()
    metrics = {name: passed / total if total else None for name, (passed, total) in counts.items()}
    metrics.update(recall_at_5=hits / max(1, answerable), mrr=reciprocal / max(1, answerable),
                   retrieval_abstention=abstained / max(1, negatives))
    thresholds = {"intent": .92, "ticker": .98, "horizon": .97, "requested_count": .97, "valid_plan": 1,
                  "recall_at_5": .85, "mrr": .65, "retrieval_abstention": .95, "single_stock_route": 1, "conflict_action": 1}
    gates = {key: metrics[key] >= threshold for key, threshold in thresholds.items()}
    report = {"schema_version": "upgrade-evaluation-v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": data["version"], "dataset_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "provenance": data["provenance"], "source_root": str(args.source_root or ROOT),
        "coverage": {"nlu": len(data["nlu"]), "nlu_templates": len({item["template_group"] for item in data["nlu"]}),
            "rag": len(data["rag"]), "rag_documents": len(data["documents"]), "screening": len(data["screening"]),
            "single_stock": len(data["single_stock"]), "conflicts": len(data["conflicts"])},
        "scope": "Synthetic routing/retrieval/conflict checks. Single-stock dates are scenario metadata, not historical price validation. No live LLM citation precision or independent holdout measured.",
        "metrics": metrics, "thresholds": thresholds, "gates": gates, "all_measured_gates_passed": all(gates.values()),
        "nlu_latency_ms": {"p95": sorted(latencies)[int(.95 * (len(latencies) - 1))]},
        "citation_precision_live_llm": None, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "gates": gates, "report": str(args.output)}, indent=2))
    if args.strict and not all(gates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
