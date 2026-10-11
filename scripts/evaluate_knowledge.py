from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_multi_agent.chat.query_parser import QueryParser
from stock_multi_agent.services.knowledge_service import KnowledgeService


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the AUTHOR-CREATED Knowledge QA set.")
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("evaluation/knowledge_qa/author-created-v1.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases = payload["cases"]
    nlu = QueryParser(use_configured_provider=False)
    knowledge = KnowledgeService()
    records = []
    for case in cases:
        parsed = nlu.parse(case["query"])
        result = knowledge.retrieve(case["query"])
        hit = bool(result.citations)
        topic_hit = case["expected_topic"] is None or any(
            case["expected_topic"] in citation.topics for citation in result.citations
        )
        citation_complete = all(
            citation.document_id and citation.chunk_id and citation.publisher and citation.source_url
            and citation.retrieved_at and citation.content_sha256
            for citation in result.citations
        )
        records.append({
            "id": case["id"],
            "intent": parsed.intent.value == case["expected_intent"],
            "retrieval": hit == case["expect_hit"],
            "topic": topic_hit if case["expect_hit"] else not hit,
            "citation": citation_complete,
            "freshness": (
                result.requires_freshness_check == case.get("expect_freshness", False)
                if "expect_freshness" in case else True
            ),
        })
    metrics = {
        key: sum(bool(record[key]) for record in records) / len(records)
        for key in ("intent", "retrieval", "topic", "citation", "freshness")
    }
    print(json.dumps({
        "dataset": payload["dataset_name"], "cases": len(cases), "metrics": metrics,
        "failures": [record for record in records if not all(record[key] for key in metrics)],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
