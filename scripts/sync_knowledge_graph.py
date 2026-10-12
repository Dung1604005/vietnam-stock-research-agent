from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_multi_agent.config import load_workspace_env
from stock_multi_agent.services.knowledge_store import KnowledgeStore


def main() -> None:
    load_workspace_env()
    parser = argparse.ArgumentParser(description="Project the local Knowledge Base into Neo4j.")
    parser.add_argument("--database", type=Path, help="Optional local KB SQLite path.")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the stock-kb Neo4j namespace.")
    args = parser.parse_args()

    store = KnowledgeStore(args.database) if args.database else KnowledgeStore()
    status = store.sync_knowledge_graph(rebuild=args.rebuild)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if status.get("available") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
