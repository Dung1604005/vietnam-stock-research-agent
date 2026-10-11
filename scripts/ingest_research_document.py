from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from stock_multi_agent.models.research import ResearchDocumentType
from stock_multi_agent.services.research_store import ResearchStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a verified local research document with mandatory provenance.")
    parser.add_argument("file", type=Path, help="UTF-8 .txt/.md/.html document")
    parser.add_argument("--title", required=True)
    parser.add_argument("--type", required=True, choices=[item.value for item in ResearchDocumentType])
    parser.add_argument("--source-url", required=True, help="Original public/authorized document URL")
    parser.add_argument("--publisher", required=True)
    parser.add_argument("--published-at", required=True, help="ISO-8601 timestamp with timezone, e.g. 2026-10-09T09:00:00+07:00")
    parser.add_argument("--ticker", action="append", default=[])
    parser.add_argument("--language", default="vi")
    args = parser.parse_args()

    published_at = datetime.fromisoformat(args.published_at)
    store = ResearchStore()
    document = store.ingest_text(
        text=store.read_text_file(args.file), title=args.title,
        document_type=ResearchDocumentType(args.type), source_url=args.source_url,
        publisher=args.publisher, published_at=published_at, tickers=args.ticker,
        language=args.language,
    )
    print(f"Ingested {document.document_id} · sha256={document.content_sha256} · source={document.source_url}")


if __name__ == "__main__":
    main()
