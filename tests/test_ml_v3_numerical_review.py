"""Regression for false float32 MAE victories; synthetic saved ZIP fixtures only."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from stock_multi_agent.services.ml_features_v3 import FEATURES, FEATURE_VERSION
from stock_multi_agent.services.ml_model_registry import (
    mae_strictly_better, evaluate_model_gate, research_gate_diagnostics,
)
from stock_multi_agent.services.ml_result_review import review_saved_v3_archive, render_numerical_review
from stock_multi_agent.services.ml_validation import regression_metrics, paired_date_block_bootstrap

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return json.loads((ROOT / 'evaluation/configs/lstm-v3-protocol.json').read_text(encoding='utf-8'))


def report_fixture():
    return {'protocol_id': protocol()['protocol_id'], 'pilot': False, 'scope': 'DEVELOPMENT_WALK_FORWARD',
        'dataset_sha256': 'a' * 64, 'price_adjustment': 'UNKNOWN', 'engineering_checks': {}, 'forward': {},
        'folds': [{'fold_id': fold, 'seed': seed, 'horizon': horizon, 'mae': .009, 'zero_mae': .01,
            'ridge_mae': .009, 'tree_mae': .009, 'calibration_alpha': .5, 'uses_lstm_signal': True}
            for fold in range(3) for seed in (42, 123) for horizon in (1, 5)]}


@pytest.mark.parametrize('candidate,reference,expected', [
    (.01739245808927901, .017392458394169807, False),
    (.014825900977589236, .01482590101659298, False),
    (.015411226024639827, .015460080467164516, True),
    (.015459010960003005, .015460080467164516, True),
    (.01, .01, False), (.011, .01, False), (0., .01, True),
    (float('nan'), .01, False), (.009, float('inf'), False), (True, .01, False),
])
def test_mae_comparison_has_numeric_tolerance_not_forced_improvement(candidate, reference, expected):
    assert mae_strictly_better(candidate, reference) is expected


@pytest.mark.parametrize('alpha', [0., .5])
def test_rounding_ties_cannot_count_as_winning_folds_even_with_alpha(alpha):
    report = report_fixture()
    for row in report['folds']:
        if row['horizon'] == 1 and row['fold_id'] < 2:
            row.update(mae=.01739245808927901, zero_mae=.017392458394169807,
                ridge_mae=.018, tree_mae=.018, calibration_alpha=alpha)
    details = research_gate_diagnostics(report, protocol())
    assert details['winning_folds'] == 1 and not details['passed']
    gate = evaluate_model_gate(report, protocol())
    assert gate.validation_status == 'RESEARCH_GATE_PENDING_OR_REJECTED'
    assert not gate.policy_eligibility and gate.technical_blend_weight == 0
    assert any('Engineering' in reason for reason in gate.reasons)
    assert any('adjusted' in reason for reason in gate.reasons)
    assert any('forward' in reason for reason in gate.reasons)


def test_alpha_zero_bias_improvement_is_not_a_winning_lstm_fold():
    report = report_fixture()
    for row in report['folds']:
        row['calibration_alpha'] = 0
        row['uses_lstm_signal'] = False
    assert research_gate_diagnostics(report, protocol())['winning_folds'] == 0
    assert not evaluate_model_gate(report, protocol()).policy_eligibility


def test_forward_pending_does_not_hide_other_blockers():
    gate = evaluate_model_gate(report_fixture(), protocol())
    assert gate.validation_status == 'PENDING_FORWARD_VALIDATION'
    assert len(gate.reasons) == 3


def test_zero_prediction_bootstrap_and_mae_agree_across_float_dtypes():
    actual = np.full(100, .01739245808927901, dtype=np.float32)
    zero32 = np.zeros(100, dtype=np.float32)
    zero64 = np.zeros(100, dtype=np.float64)
    dates = np.repeat(pd.bdate_range('2025-01-01', periods=50).to_numpy(), 2)
    result = paired_date_block_bootstrap(actual, zero64, zero32, dates, repetitions=50)
    assert result['estimate'] == result['lower'] == result['upper'] == 0
    assert regression_metrics(actual, zero32)['mae'] == regression_metrics(actual, zero64)['mae']


def archive_fixture(tmp_path, *, corrupt=None):
    config, report = protocol(), report_fixture()
    report['folds'] = []
    spec = {'features': FEATURES, 'feature_version': FEATURE_VERSION, 'window': 30, 'horizons': [1, 5],
        'protocol_id': config['protocol_id'], 'dataset_sha256': report['dataset_sha256'],
        'selected_fold': 2, 'seed': 42, 'selected_using': 'LATEST_FOLD_VALIDATION_LOSS_ONLY'}
    contents = {}
    for fold in range(3):
        for seed in config['seeds']:
            parts = []
            for horizon in config['horizons']:
                actual = np.full(20, .01739245808927901)
                predicted = np.zeros(20) if horizon == 1 and fold < 2 else actual * .8
                frame = pd.DataFrame({'ticker': 'DEMO', 'date': pd.bdate_range('2025-01-01', periods=20).strftime('%Y-%m-%d'),
                    'fold_id': fold, 'seed': seed, 'horizon': horizon,
                    'actual': actual, 'prediction': predicted, 'zero': 0., 'ridge': actual * .8, 'tree': actual * .8})
                row = {'fold_id': fold, 'seed': seed, 'horizon': horizon, **regression_metrics(actual, predicted),
                    'zero_mae': float(np.mean(np.abs(actual.astype(np.float32)))),
                    'ridge_mae': regression_metrics(actual, frame.ridge)['mae'],
                    'tree_mae': regression_metrics(actual, frame.tree)['mae'],
                    'calibration_alpha': 0. if horizon == 1 and fold < 2 else 1.,
                    'uses_lstm_signal': not (horizon == 1 and fold < 2)}
                report['folds'].append(row)
                parts.append(frame)
            contents[f'fold-{fold}-seed-{seed}/predictions.csv'] = pd.concat(parts).to_csv(index=False).encode()
    contents.update({'model.keras': b'Not a real model; never loaded by review',
        'spec.json': json.dumps(spec).encode(), 'evaluation.json': json.dumps(report).encode(),
        'protocol.json': json.dumps(config).encode()})
    hashes = {name: hashlib.sha256(contents[name]).hexdigest() for name in ('model.keras', 'spec.json', 'evaluation.json', 'protocol.json')}
    manifest = {'schema_version': 'lstm-artifact-v3', 'files': hashes,
        'artifact_identity': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        'predictor_identity': hashlib.sha256(json.dumps({name: hashes[name] for name in ('model.keras', 'spec.json', 'protocol.json')}, sort_keys=True).encode()).hexdigest()}
    contents['manifest.json'] = json.dumps(manifest).encode()
    if corrupt == 'pinned':
        contents['model.keras'] = b'Changed bytes'
    if corrupt == 'predictions':
        key = 'fold-0-seed-42/predictions.csv'
        contents[key] = contents[key].replace(b'DEMO', b'OTHER', 1)
    if corrupt == 'path':
        contents['../escape.py'] = b'Never execute'
    path = tmp_path / 'synthetic-results.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        for name, data in contents.items():
            archive.writestr('run/' + name, data)
    return path


def test_review_zip_recomputes_metrics_without_mutating_or_extracting(tmp_path):
    path = archive_fixture(tmp_path)
    original = path.read_bytes()
    review = review_saved_v3_archive(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    assert review['research_diagnostics_by_horizon']['1']['winning_folds'] == 1
    assert review['research_diagnostics_by_horizon']['5']['winning_folds'] == 3
    assert len(review['prediction_audit']) == 12
    assert review['corrected_evaluation']['gate'] == review['gates_by_horizon']['1']
    for row in review['prediction_audit']:
        if row['horizon'] == 1 and row['fold_id'] < 2:
            assert row['all_zero_prediction']
            assert row['corrected_mae'] == row['corrected_zero_mae']
    assert all(gate['technical_blend_weight'] == 0 for gate in review['gates_by_horizon'].values())
    assert 'NOT_PASSED' in render_numerical_review(review)


@pytest.mark.parametrize('corrupt,match', [('pinned', 'hash mismatch'), ('predictions', 'Seed endpoints'), ('path', 'Unsafe ZIP')])
def test_review_rejects_corrupt_or_inconsistent_archives(tmp_path, corrupt, match):
    with pytest.raises(ValueError, match=match):
        review_saved_v3_archive(archive_fixture(tmp_path, corrupt=corrupt))


def test_training_keeps_float32_batches_but_float64_evaluation():
    source = (ROOT / 'notebooks/train_lstm_kaggle_v3.py').read_text(encoding='utf-8')
    assert 'evaluation_targets' in source and 'dtype=np.float64' in source
    assert 'baseline_mae["zero"]' in source
