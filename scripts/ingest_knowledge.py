from __future__ import annotations

import argparse
from pathlib import Path

from stock_multi_agent.services.knowledge_store import KnowledgeStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest provenance-declared documents into the local Knowledge Base.")
    parser.add_argument("--manifest", required=True, type=Path, help="JSON manifest; paths are relative to the manifest.")
    parser.add_argument("--database", type=Path, help="Optional SQLite path; keep it inside the D: workspace.")
    args = parser.parse_args()

    store = KnowledgeStore(args.database) if args.database else KnowledgeStore()
    documents = store.ingest_manifest(args.manifest.resolve())
    print(f"Knowledge ingestion complete: {len(documents)} manifest entries; {store.count_documents()} active documents.")
    for document in documents:
        print(f"- {document.document_id} · {document.title} · {document.source_url}")


if __name__ == "__main__":
    main()
