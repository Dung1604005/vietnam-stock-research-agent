"""Export frozen real analyses and summarize explicit human judgments, offline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from stock_multi_agent.services.review_evaluation import export_review_pack, summarize_review_pack


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=root / '.runtime/research/research.sqlite3')
    parser.add_argument('--output', type=Path, default=root / '.runtime/evaluations/real-analysis-review.json')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--summarize', type=Path, help='Summarize an already reviewed pack, without LLM grading')
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize_review_pack(json.loads(args.summarize.read_text(encoding='utf-8'))), ensure_ascii=False, indent=2))
        return
    if not args.database.exists():
        parser.error('No research database found. Run and save a real stock analysis first.')
    destination = args.output.resolve()
    if not destination.is_relative_to(root.resolve()):
        parser.error('Keep review packs inside the project (on D: for this workspace).')
    if destination.exists():
        parser.error('Output exists; choose a new --output to preserve prior human judgments.')
    pack = export_review_pack(args.database, limit=args.limit)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Exported {len(pack["cases"])} real snapshots: {destination}')
    print('PENDING_HUMAN_REVIEW; no financial accuracy claim has been measured.')


if __name__ == '__main__':
    main()
