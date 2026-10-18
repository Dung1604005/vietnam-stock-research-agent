"""Recompute saved V3 predictions locally; no model loading, extraction or training."""
from argparse import ArgumentParser
import json
from pathlib import Path

from stock_multi_agent.services.ml_result_review import review_saved_v3_archive, render_numerical_review


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New review JSON path; companion Markdown is created too')
    args = parser.parse_args()
    target = args.output
    markdown = target.with_suffix('.md')
    if target.suffix.lower() != '.json' or target.exists() or markdown.exists():
        parser.error('Choose a NEW .json output path; existing JSON/Markdown will not be overwritten')
    review = review_saved_v3_archive(args.archive)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as handle:
        json.dump(review, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with markdown.open('x', encoding='utf-8') as handle:
        handle.write(render_numerical_review(review))
    print(json.dumps({'report': str(target.resolve()), 'markdown': str(markdown.resolve()),
        'gates_by_horizon': review['gates_by_horizon']}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
