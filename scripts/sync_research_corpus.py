from __future__ import annotations

import argparse
from pathlib import Path
from stock_multi_agent.config import load_workspace_env
from stock_multi_agent.services.research_sync import ResearchSyncService
from stock_multi_agent.services.source_registry import SourceRegistry


def main():
    load_workspace_env()
    parser = argparse.ArgumentParser(description="Discover and incrementally ingest permitted public research reports.")
    parser.add_argument("--source", help="Only this source_id")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--due", action="store_true", help="Check only sources whose discovery TTL expired (default)")
    parser.add_argument("--force-discovery", action="store_true", help="Refresh listings now; does not bypass robots, approval or checksum")
    parser.add_argument("--queue-only", action="store_true", help="Process queued/approved candidates without crawling listings")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--pages", type=int, help="Override per-source HTTP request budget for this run")
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()
    sync = ResearchSyncService(registry=SourceRegistry(args.registry))
    if args.pages is not None:
        if not 2 <= args.pages <= 500:
            parser.error("--pages must be between 2 and 500")
        for source in sync.registry.sources:
            source.max_pages_per_run = args.pages
    if args.health:
        import json
        print(json.dumps(sync.repository.health(), ensure_ascii=False, indent=2))
        return
    report = sync.sync(source_id=args.source, force_discovery=args.force_discovery, seconds=args.seconds, queue_only=args.queue_only)
    print(report.model_dump_json(indent=2))
    if report.status == "COMPLETED_WITH_ISSUES":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
