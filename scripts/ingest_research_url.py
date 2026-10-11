from __future__ import annotations

import argparse
import re
from datetime import datetime
from urllib.request import Request, urlopen

from stock_multi_agent.models.research import ResearchDocumentType
from stock_multi_agent.services.research_store import ResearchIngestionError, ResearchStore


MAX_DOCUMENT_BYTES = 3_000_000


def fetch_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": "StockResearchAssistant/0.1 (citation-first ingestion)"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - explicit user-supplied/approved official URL
        content_type = response.headers.get_content_type()
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise ResearchIngestionError(f"Only HTML sources are supported, got {content_type}")
        payload = response.read(MAX_DOCUMENT_BYTES + 1)
        if len(payload) > MAX_DOCUMENT_BYTES:
            raise ResearchIngestionError(f"Document exceeds {MAX_DOCUMENT_BYTES} byte ingestion limit")
        charset = response.headers.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and ingest one approved HTML research source with mandatory provenance.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--type", required=True, choices=[item.value for item in ResearchDocumentType])
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--publisher", required=True)
    parser.add_argument("--published-at", required=True, help="ISO-8601 timestamp with timezone")
    parser.add_argument("--ticker", action="append", default=[])
    parser.add_argument("--language", default="vi")
    args = parser.parse_args()

    published_at = datetime.fromisoformat(args.published_at)
    html = fetch_html(args.source_url)
    store = ResearchStore()
    document = store.ingest_text(
        text=store.normalize_html(html), title=args.title,
        document_type=ResearchDocumentType(args.type), source_url=args.source_url,
        publisher=args.publisher, published_at=published_at, tickers=args.ticker,
        language=args.language,
    )
    print(f"Ingested {document.document_id} · sha256={document.content_sha256} · source={document.source_url}")


if __name__ == "__main__":
    main()
