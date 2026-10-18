"""Evidence-only V3 failure audit; never load models or select a V4 on test."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from stock_multi_agent.services.ml_dataset_v3 import sha256_file
from stock_multi_agent.services.ml_validation import expanding_purged_folds
from stock_multi_agent.services.ml_result_review import review_saved_v3_archive


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Audit output exists; choose a new path')
    review = review_saved_v3_archive(args.archive)
    if sha256_file(args.data) != review['dataset_sha256']:
        raise ValueError('Audit dataset differs from the run dataset')
    columns = ['ticker', 'date', 'target_end_5', 'target_return_1', 'target_return_5',
        'volume', 'active_ratio_20', 'training_endpoint']
    frame = pd.read_csv(args.data, usecols=columns, parse_dates=['date', 'target_end_5'])
    endpoints = np.flatnonzero(frame.training_endpoint.to_numpy())
    splits = []
    for fold in expanding_purged_folds(frame, endpoints):
        for name in ('train', 'validation', 'calibration', 'test'):
            rows = frame.iloc[getattr(fold, name)]
            actual = rows.target_return_1.to_numpy(float)
            splits.append({'fold_id': fold.fold_id, 'split': name, 'samples': len(rows),
                'eligible_dates': rows.date.nunique(), 'median_next_return': float(np.median(actual)),
                'flat_fraction': float(np.mean(actual == 0)), 'up_fraction': float(np.mean(actual > 0)),
                'down_fraction': float(np.mean(actual < 0)), 'median_volume': float(rows.volume.median()),
                'fully_active_context_fraction': float(rows.active_ratio_20.eq(1).mean())})
    trials = []
    with zipfile.ZipFile(args.archive) as archive:
        root = archive.namelist()[0].split('/')[0]
        for fold in range(3):
            for seed in (42, 123):
                with archive.open(f'{root}/fold-{fold}-seed-{seed}/history.csv') as handle:
                    history = pd.read_csv(handle)
                result = next(row for row in review['corrected_evaluation']['folds']
                    if (row['fold_id'], row['seed'], row['horizon']) == (fold, seed, 1))
                trials.append({'fold_id': fold, 'seed': seed, 'epochs_run': len(history),
                    'best_combined_validation_epoch': int(history.val_loss.argmin() + 1),
                    'validation_loss_first': float(history.val_loss.iloc[0]),
                    'validation_loss_best': float(history.val_loss.min()),
                    'validation_loss_last': float(history.val_loss.iloc[-1]),
                    'next_session_calibration_alpha': result['calibration_alpha'],
                    'next_session_uses_lstm_signal': result['uses_lstm_signal']})
    result = {'schema_version': 'v3-failure-audit-v1', 'generated_at': datetime.now(timezone.utc).isoformat(),
        'source_archive': args.archive.name, 'archive_sha256': review['source_archive_sha256'],
        'dataset_sha256': review['dataset_sha256'], 'scope': 'POSTHOC_DEVELOPMENT_ANALYSIS_NOT_NEW_HOLDOUT',
        'split_statistics': splits, 'trial_histories': trials,
        'confirmed': ['Fold0 validation has only five eligible dates; calendar duration overstated effective validation coverage.',
            'Every original train/validation/calibration/test block has next-return median zero.',
            'Calibration selected alpha=0 in both seeds of folds0/1; predictions reduce to the constant baseline.',
            'Next-session test flat frequency is about23-25%; flat labels remain in the benchmark.',
            'Old checkpoint selection used combined normalized multi-horizon loss, not raw one-session MAE.'],
        'hypotheses_not_proven': ['Equal horizon loss and shrink-to-median objective may suppress weak one-session signal.',
            'Liquidity/flat-state features and auxiliary direction learning may help; only controlled new experiments can test this.',
            'Stronger model capacity or longer training alone is not demonstrated to solve these issues.'],
        'non_claims': ['No evidence that all unadjusted jumps are errors; no future-label filtering was performed.',
            'No causal proof from training curves alone; calibration alpha0 is not proof that raw pre-calibration model output is zero.',
            'V4 needs GPU training and independent forward evaluation; no pass is guaranteed.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'output': str(args.output.resolve()), 'split_statistics': splits}, indent=2))


if __name__ == '__main__':
    main()
