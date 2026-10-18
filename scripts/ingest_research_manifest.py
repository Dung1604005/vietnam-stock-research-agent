from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_multi_agent.services.research_ingestion import ResearchIngestionService
from stock_multi_agent.services.research_store import ResearchStore


def main():
    parser = argparse.ArgumentParser(description="Incrementally ingest approved reports (no paid API).")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    results = ResearchIngestionService(ResearchStore()).ingest_manifest(args.manifest.resolve(), force=args.force)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if any(item["status"] == "ERROR" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
