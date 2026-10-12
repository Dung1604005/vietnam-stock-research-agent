from __future__ import annotations

import argparse
import sys

from stock_multi_agent.services.research_store import ResearchStore


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build vector embeddings for the verified research corpus.")
    parser.add_argument("--force", action="store_true", help="Recompute vectors for the configured embedding model.")
    args = parser.parse_args()
    store = ResearchStore()
    count = store.reindex_embeddings(force=args.force)
    model_id = getattr(store.embedding_provider, "model_id", "disabled")
    print(f"Indexed {count} research chunks with {model_id}")
    if store.last_embedding_error:
        raise RuntimeError(store.last_embedding_error)


if __name__ == "__main__":
    main()
