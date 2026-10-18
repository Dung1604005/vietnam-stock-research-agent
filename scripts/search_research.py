from __future__ import annotations

import argparse
from stock_multi_agent.config import load_workspace_env

from stock_multi_agent.services.research_store import ResearchStore
from stock_multi_agent.services.reporting_period import period_display
from stock_multi_agent.services.research_freshness import research_scope


def main() -> None:
    load_workspace_env()
    parser = argparse.ArgumentParser(description="Search verified local research documents and print citations.")
    parser.add_argument("query")
    parser.add_argument("--ticker")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--current", action="store_true", help="Apply current-data freshness gates")
    args = parser.parse_args()
    bundle = ResearchStore().search(args.query, ticker=args.ticker, limit=args.limit,
        scope=research_scope(args.query, ticker=args.ticker, current=True) if args.current else None)
    print(bundle.retrieval_note)
    if bundle.freshness_note:
        print(bundle.freshness_note)
    for citation in bundle.citations:
        print(f"\n[{citation.title}] {citation.source_url}")
        print(f"published={citation.published_at.isoformat()} chunk={citation.chunk_id[:12]} sha256={citation.content_sha256[:12]}")
        print(period_display(citation))
        print(citation.quote)


if __name__ == "__main__":
    main()
