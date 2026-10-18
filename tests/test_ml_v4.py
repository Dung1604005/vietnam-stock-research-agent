"""V4 engineering checks only. Synthetic data is never investment evidence."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from stock_multi_agent.services.ml_dataset_v3 import build_dataset
from stock_multi_agent.services.ml_features_v3 import window_endpoints
from stock_multi_agent.services.ml_validation import TrainOnlyScaler
from stock_multi_agent.services.ml_v4 import (FEATURES, FEATURE_VERSION, feature_frame,
    eligible_session_folds, origin_scales, direction_labels, direction_class_weights,
    split_calibration, fit_calibration, apply_calibration, classification_metrics,
    baseline_predictions, scaler_specification)

ROOT = Path(__file__).resolve().parents[1]


def synthetic_raw(count=1600):
    rng = np.random.default_rng(42)
    price = 100 * np.exp(np.cumsum(rng.normal(0, .01, count)))
    return pd.DataFrame({'ticker': 'DEMO', 'date': pd.bdate_range('2017-01-02', periods=count),
        'open': price * .999, 'high': price * 1.01, 'low': price * .99, 'close': price,
        'volume': rng.integers(10000, 20000, count), 'source': 'synthetic-v4-test',
        'source_id': 'synthetic-v4-test:DEMO', 'fetched_at': '2026-10-18T00:00:00+00:00', 'price_adjustment': 'UNKNOWN'})


@pytest.fixture(scope='module')
def built():
    return build_dataset(synthetic_raw()).rows


def test_v4_new_features_causal_and_verified_cache_parity(built):
    one = feature_frame(built)
    cached = feature_frame(built, reuse_verified_v3_features=True)
    np.testing.assert_allclose(one[FEATURES], cached[FEATURES], equal_nan=True, rtol=1e-7)
    changed = built.copy()
    changed.loc[1300:, ['open', 'high', 'low', 'close', 'volume']] *= 2
    future = feature_frame(changed)
    np.testing.assert_allclose(one.loc[:1299, FEATURES], future.loc[:1299, FEATURES], equal_nan=True)
    assert len(FEATURES) == 25 and one.endpoint_eligible.any()
    np.testing.assert_array_equal(window_endpoints(one), window_endpoints(built))


def test_v4_folds_enforce_actual_eligible_dates_and_purge(built):
    frame = feature_frame(built)
    folds = eligible_session_folds(frame, window_endpoints(frame))
    used = set()
    for fold in folds:
        splits = [frame.iloc[getattr(fold, name)] for name in ('train', 'validation', 'calibration', 'test')]
        for left, right in zip(splits, splits[1:]):
            assert left.target_end_5.max() < right.date.min()
        assert all(fold.boundaries['eligible_session_counts'][name] >= minimum
            for name, minimum in {'train': 120, 'validation': 30, 'calibration': 30, 'test': 40}.items())
        assert not used.intersection(fold.test)
        used.update(fold.test)
        first, second = split_calibration(frame, fold.calibration)
        assert not set(first).intersection(second)
        assert frame.iloc[first].target_end_5.max() < frame.iloc[second].date.min()


def test_sparse_validation_never_silently_counts_as_months(built):
    frame = feature_frame(built)
    indices = window_endpoints(frame)
    with pytest.raises(ValueError, match='insufficient eligible sessions|Too few'):
        eligible_session_folds(frame, indices[::40])


def test_source_calendar_gap_not_compressed_to_endpoint_calendar(built):
    frame = feature_frame(built)
    folds = eligible_session_folds(frame, window_endpoints(frame)[::2],
        minimum_sessions={'train': 60, 'validation': 10, 'calibration': 10, 'test': 20})
    calendar = np.sort(frame.date.unique())
    for fold in folds:
        end = np.searchsorted(calendar, np.datetime64(fold.boundaries['train_end']))
        start = np.searchsorted(calendar, np.datetime64(fold.boundaries['validation_start']))
        assert start - end == 6


def test_origin_volatility_scaling_is_causal_finite_and_bounded(built):
    frame = feature_frame(built)
    before = origin_scales(frame)
    changed = frame.copy()
    changed.loc[1300:, 'volatility_20'] *= 10
    after = origin_scales(changed)
    np.testing.assert_array_equal(before[:1300], after[:1300])
    assert np.isfinite(before).all() and (before[:, 0] >= .005).all() and (before[:, 0] <= .10).all()
    np.testing.assert_allclose(before[:, 1], before[:, 0] * np.sqrt(5))


def test_class_weights_use_only_the_labels_passed_and_keep_flat_class():
    labels = direction_labels(np.array([-.01, 0., .01, 0.]))
    np.testing.assert_array_equal(labels, [0, 1, 2, 1])
    weights = direction_class_weights(labels)
    assert len(weights) == 3 and np.isfinite(weights).all()
    with pytest.raises(ValueError):
        direction_labels([np.nan])


def test_calibration_can_honestly_choose_zero_instead_of_forcing_signal():
    actual = np.zeros(40)
    raw = np.linspace(-.04, .04, 40)
    probabilities = np.full(40, .2)
    fitted = fit_calibration(actual, raw, probabilities)
    assert fitted['alpha'] == 0
    np.testing.assert_array_equal(apply_calibration(raw, probabilities, fitted), actual)


def test_flat_gate_fits_only_registered_calibration_arrays():
    actual = np.tile([0., .02, -.02], 30)
    raw = np.tile([.01, .02, -.02], 30)
    probability = np.tile([.9, .05, .05], 30)
    fitted = fit_calibration(actual, raw, probability)
    assert fitted['flat_threshold'] in (.5, .65, .8)
    assert fitted['fit_mae'] == pytest.approx(0)
    before = dict(fitted)
    apply_calibration(raw * 3, probability, fitted)
    assert fitted == before


@pytest.mark.parametrize('bad', [np.nan, np.inf, -1., 2.])
def test_bad_class_probabilities_are_rejected(bad):
    with pytest.raises(ValueError):
        fit_calibration(np.ones(10), np.zeros(10), np.full(10, bad))


def test_three_class_report_not_fifty_percent_assumption():
    actual = np.array([-.01, 0., .01])
    result = classification_metrics(actual, np.eye(3))
    assert result['three_class_accuracy'] == result['balanced_three_class_accuracy'] == 1
    assert 'UNCALIBRATED' in result['probability_status']


def test_v4_scaler_metadata_and_baselines_match_25_features(built):
    frame = feature_frame(built)
    folds = eligible_session_folds(frame, window_endpoints(frame))
    scaler = TrainOnlyScaler().fit(frame.iloc[folds[0].train][FEATURES].to_numpy(float))
    before = scaler_specification(scaler)
    scaler.transform(np.full((1, 25), 1e6))
    assert before == scaler_specification(scaler) and before['features'] == FEATURES
    values, details = baseline_predictions(frame, folds[0], 1, include_tree=False)
    assert details['feature_count'] == 25 and all(len(x) == len(folds[0].test) for x in values.values())


def test_notebook_is_self_contained_python_and_exact_source_snapshot():
    path = ROOT / 'notebooks/train_lstm_kaggle_v4_ready.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    embedded = {}
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            ast.parse(source)
            compile(source, f'v4-notebook-cell-{index}', 'exec')
            if source.startswith('PORTABLE_SOURCES = '):
                exec(source, embedded)
    assert embedded['TRAINING_SOURCE'] == (ROOT / 'notebooks/train_lstm_kaggle_v4.py').read_text(encoding='utf-8')
    assert json.loads(embedded['PROTOCOL_TEXT']) == json.loads((ROOT / 'evaluation/configs/lstm-v4-protocol.json').read_text(encoding='utf-8'))
    for name, source in embedded['PORTABLE_SOURCES'].items():
        if name.endswith('__init__.py'):
            continue
        assert source == (ROOT / 'src' / name).read_text(encoding='utf-8')
    spec = importlib.util.spec_from_file_location('v4_safe_import', ROOT / 'notebooks/train_lstm_kaggle_v4.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main) and callable(module.build_model)


def test_embedded_runtime_imports_in_fresh_python_without_app_startup(isolated_runtime_root):
    notebook = json.loads((ROOT / 'notebooks/train_lstm_kaggle_v4_ready.ipynb').read_text(encoding='utf-8'))
    namespace = {}
    exec(''.join(notebook['cells'][2]['source']), namespace)
    runtime = isolated_runtime_root / 'v4-notebook-portable-test.zip'
    with zipfile.ZipFile(runtime, 'x') as archive:
        for name, source in namespace['PORTABLE_SOURCES'].items():
            archive.writestr(name, source)
    snippet = "import sys; sys.path.insert(0, sys.argv[1]); from stock_multi_agent.services.ml_v4 import FEATURES; from stock_multi_agent.services.ml_model_registry import GATE_IMPLEMENTATION_VERSION; assert 'tensorflow' not in sys.modules; print(len(FEATURES), GATE_IMPLEMENTATION_VERSION)"
    result = subprocess.run([sys.executable, '-I', '-c', snippet, str(runtime)], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == '25 v3-gate-numeric-fix-1'


def test_v4_source_keeps_validation_primary_and_saves_prediction_integrity():
    source = (ROOT / 'notebooks/train_lstm_kaggle_v4.py').read_text(encoding='utf-8')
    assert "monitor = 'val_raw_return_1_mae'" in source
    assert 'initial_value_threshold=resume_score' in source
    assert 'predictions_sha256' in source
    assert 'load_model(checkpoint, safe_mode=True)' in source
