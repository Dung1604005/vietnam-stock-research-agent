"""Verify existing V3 CSV and V4 eligible-session coverage, without GPU training."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from stock_multi_agent.services.ml_dataset_v3 import sha256_file
from stock_multi_agent.services.ml_features_v3 import window_endpoints
from stock_multi_agent.services.ml_v4 import FEATURES, FEATURE_VERSION, feature_frame, eligible_session_folds, split_calibration


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('evaluation/configs/lstm-v4-protocol.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new preflight output')
    manifest = json.loads((args.data.parent / 'dataset.manifest.json').read_text(encoding='utf-8'))
    data_hash = sha256_file(args.data)
    if data_hash != manifest['files']['dataset']['sha256']:
        raise ValueError('Source CSV checksum mismatch')
    config = json.loads(args.protocol.read_text(encoding='utf-8'))
    if config['feature_version'] != FEATURE_VERSION:
        raise ValueError('V4 protocol feature version mismatch')
    frame = feature_frame(pd.read_csv(args.data, parse_dates=['date', 'target_end_1', 'target_end_5']), reuse_verified_v3_features=True)
    endpoints = window_endpoints(frame, config['window'])
    folds = eligible_session_folds(frame, endpoints, folds=config['outer_folds'], gap_sessions=config['gap_sessions'], minimum_sessions=config['minimum_eligible_sessions'])
    split_reports = []
    for fold in folds:
        first, second = split_calibration(frame, fold.calibration)
        split_reports.append({**fold.manifest(), 'calibration_fit': {'samples': len(first), 'dates': frame.iloc[first].date.nunique()},
            'interval_fit': {'samples': len(second), 'dates': frame.iloc[second].date.nunique()}})
    report = {'schema_version': 'lstm-v4-preflight-v1', 'generated_at': datetime.now(timezone.utc).isoformat(),
        'dataset_sha256': data_hash, 'protocol_sha256': sha256_file(args.protocol),
        'rows': len(frame), 'tickers': frame.ticker.nunique(), 'features': FEATURES, 'feature_count': len(FEATURES),
        'endpoints': len(endpoints), 'splits': split_reports, 'training_status': 'NOT_TRAINED_ON_MARKET_DATA',
        'policy_eligibility': False, 'source_adjustment_counts': frame.price_adjustment.value_counts().to_dict(),
        'limitations': ['Coverage and feature preflight only; no V4 performance measurement or promotion.',
            'Reuses the frozen V3 CSV; no new market observations were collected.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'output': str(args.output.resolve()), 'rows': len(frame), 'endpoints': len(endpoints), 'splits': split_reports}, indent=2))


if __name__ == '__main__':
    main()
