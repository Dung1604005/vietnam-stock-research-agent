from __future__ import annotations

import argparse
import json
from datetime import datetime

from stock_multi_agent.config import load_workspace_env
from stock_multi_agent.models.research import ResearchDocumentType
from stock_multi_agent.services.research_sync import ResearchSyncService


def main():
    load_workspace_env()
    parser = argparse.ArgumentParser(description="Review persistent research candidates without editing a manifest.")
    parser.add_argument("--status", default="QUARANTINED")
    parser.add_argument("--source")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--candidate")
    parser.add_argument("--action", choices=["approve", "reject", "retry"])
    parser.add_argument("--note", default="")
    parser.add_argument("--published-at", help="Actual source publication time, ISO-8601 with timezone")
    parser.add_argument("--title")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--document-type", choices=[item.value for item in ResearchDocumentType])
    args = parser.parse_args()
    sync = ResearchSyncService()
    if args.action:
        if not args.candidate:
            parser.error("--candidate is required for a review action")
        result = sync.repository.review(args.candidate, args.action, note=args.note,
            published_at=datetime.fromisoformat(args.published_at.replace("Z", "+00:00")) if args.published_at else None,
            title=args.title, tickers=args.tickers, document_type=ResearchDocumentType(args.document_type) if args.document_type else None)
        print(result.model_dump_json(indent=2))
    else:
        print(json.dumps([item.model_dump(mode="json") for item in sync.repository.list(
            status=args.status, source_id=args.source, limit=args.limit)], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
