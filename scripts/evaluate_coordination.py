"""Run frozen snapshot coordination ablations with no external access."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, help='JSON object containing cases compatible with CoordinationCase')
    parser.add_argument('--output', type=Path, default=ROOT / '.runtime/evaluations/coordination-latest.json')
    parser.add_argument('--max-iterations', type=int, default=2)
    args = parser.parse_args()
    os.environ['STOCK_AGENT_REQUEST_TELEMETRY_ENABLED'] = 'false'
    from stock_multi_agent.services.coordination_benchmark import CoordinationBenchmarkService, CoordinationCase, synthetic_coordination_cases
    cases = ([CoordinationCase.model_validate(case) for case in json.loads(args.dataset.read_text(encoding='utf-8'))['cases']]
             if args.dataset else synthetic_coordination_cases())
    report = CoordinationBenchmarkService().run(cases, max_iterations=args.max_iterations)
    if args.dataset:
        report['dataset_version'] = 'user-supplied-frozen-snapshots'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'scope': report['scope'], 'summary': report['summary'], 'report': str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
