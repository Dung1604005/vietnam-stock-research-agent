from __future__ import annotations

import argparse

from stock_multi_agent.services.research_store import ResearchStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Search verified local research documents and print citations.")
    parser.add_argument("query")
    parser.add_argument("--ticker")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    bundle = ResearchStore().search(args.query, ticker=args.ticker, limit=args.limit)
    print(bundle.retrieval_note)
    for citation in bundle.citations:
        print(f"\n[{citation.title}] {citation.source_url}")
        print(f"published={citation.published_at.isoformat()} chunk={citation.chunk_id[:12]} sha256={citation.content_sha256[:12]}")
        print(citation.quote)


if __name__ == "__main__":
    main()
