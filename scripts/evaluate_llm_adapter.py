"""Score previously captured raw A/B predictions locally without calling models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_multi_agent.services.llm_evaluation import evaluate_adapter_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='Locked dataset folder')
    parser.add_argument('--baseline', type=Path, required=True, help='JSONL: example_id, raw_output (no fallback)')
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--identities', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('config/llm-training-v1.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import hashlib
    manifest = json.loads((args.dataset / 'manifest.json').read_text(encoding='utf-8'))
    test_bytes = (args.dataset / 'test.jsonl').read_bytes()
    if hashlib.sha256(test_bytes).hexdigest() != manifest['split_hashes']['test']:
        raise ValueError('Frozen test split checksum mismatch')
    cases = [json.loads(line) for line in test_bytes.decode().splitlines() if line.strip()]
    def load_predictions(path):
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len({row['example_id'] for row in rows}) != len(rows):
            raise ValueError('Duplicate prediction IDs')
        return {row['example_id']: row for row in rows}
    report = evaluate_adapter_pairs(cases, load_predictions(args.baseline), load_predictions(args.candidate),
        json.loads(args.protocol.read_text(encoding='utf-8')), dataset_manifest=manifest,
        identities=json.loads(args.identities.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Evaluation report already exists; use a new experiment output')
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'reason_codes': report['reason_codes'], 'output': str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
