"""Read-only numerical audit of a saved V3 ZIP; never load models or extract code."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

import numpy as np
import pandas as pd

from stock_multi_agent.services.ml_dataset_v3 import sha256_file
from stock_multi_agent.services.ml_features_v3 import FEATURES, FEATURE_VERSION
from stock_multi_agent.services.ml_model_registry import (
    GATE_IMPLEMENTATION_VERSION, evaluate_model_gate, research_gate_diagnostics,
)
from stock_multi_agent.services.ml_validation import regression_metrics, paired_date_block_bootstrap


def review_saved_v3_archive(path: Path) -> dict:
    """Recompute existing predictions, with honest provenance, not a new holdout.

    The frozen manifest pins the selected weights/spec/protocol/evaluation only.
    Trial CSVs are unpinned exports: compare their metrics with the pinned report,
    and record the complete ZIP hash. This does NOT attest model execution.
    """
    path = Path(path)
    source_hash = sha256_file(path)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or len(names) > 500:
            raise ValueError("Duplicate ZIP names or too many members")
        if sum(item.file_size for item in members) > 512 * 1024 * 1024:
            raise ValueError("ZIP exceeds 512 MiB uncompressed review bound")
        for item in members:
            parts = PurePosixPath(item.filename)
            if parts.is_absolute() or '..' in parts.parts or '\\' in item.filename or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Unsafe ZIP member path/type")
        roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
        if len(roots) != 1:
            raise ValueError("ZIP must contain exactly one root run directory")
        root = roots.pop()
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failed")

        def payload(leaf: str) -> bytes:
            name = root + '/' + leaf
            if name not in names:
                raise ValueError("Required saved result missing: " + leaf)
            return archive.read(name)

        def read_json(leaf: str) -> dict:
            raw = payload(leaf)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("JSON exceeds review size bound")
            return json.loads(raw)

        manifest = read_json('manifest.json')
        files = manifest.get('files', {})
        if manifest.get('schema_version') != 'lstm-artifact-v3' or set(files) != {'model.keras', 'spec.json', 'evaluation.json', 'protocol.json'}:
            raise ValueError("Invalid pinned V3 manifest")
        for name, expected in files.items():
            if hashlib.sha256(payload(name)).hexdigest() != expected:
                raise ValueError("Pinned file hash mismatch: " + name)
        identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        predictor = hashlib.sha256(json.dumps({name: files[name] for name in ('model.keras', 'spec.json', 'protocol.json')}, sort_keys=True).encode()).hexdigest()
        if identity != manifest.get('artifact_identity') or predictor != manifest.get('predictor_identity'):
            raise ValueError("Artifact/predictor identity mismatch")
        spec, original, protocol = [read_json(name) for name in ('spec.json', 'evaluation.json', 'protocol.json')]
        if spec.get('features') != FEATURES or spec.get('feature_version') != FEATURE_VERSION or spec.get('window') != 30 or spec.get('horizons') != [1, 5]:
            raise ValueError("Saved feature/window/horizon contract mismatch")
        if any(value.get('protocol_id') != protocol.get('protocol_id') for value in (spec, original)):
            raise ValueError("Saved protocol identity mismatch")
        if protocol.get('window') != 30 or protocol.get('horizons') != [1, 5]:
            raise ValueError("Saved protocol window/horizon mismatch")
        if original.get('dataset_sha256') != spec.get('dataset_sha256'):
            raise ValueError("Spec/evaluation dataset identity mismatch")
        for forward in original.get('forward', {}).values():
            if forward.get('predictor_identity') != predictor:
                raise ValueError("Forward evidence belongs to a different predictor")

        corrected = deepcopy(original)
        corrected['folds'] = []
        expected_keys = {(fold, seed, horizon) for fold in range(protocol['outer_folds']) for seed in protocol['seeds'] for horizon in protocol['horizons']}
        rows_by_key = {}
        for row in original['folds']:
            key = (row['fold_id'], row['seed'], row['horizon'])
            if key in rows_by_key:
                raise ValueError("Duplicate fold/seed/horizon report")
            rows_by_key[key] = row
        if set(rows_by_key) != expected_keys:
            raise ValueError("Full review requires every registered fold/seed/horizon")
        prediction_audit, alignment = [], {}
        for fold in range(protocol['outer_folds']):
            for seed in protocol['seeds']:
                name = f'fold-{fold}-seed-{seed}/predictions.csv'
                with archive.open(root + '/' + name) as handle:
                    frame = pd.read_csv(handle, float_precision='round_trip')
                required = {'ticker', 'date', 'horizon', 'fold_id', 'seed', 'actual', 'prediction', 'zero', 'ridge'}
                if required.difference(frame.columns) or frame.empty:
                    raise ValueError("Prediction CSV missing aligned metrics columns: " + name)
                if not frame.fold_id.eq(fold).all() or not frame.seed.eq(seed).all() or set(frame.horizon) != set(protocol['horizons']):
                    raise ValueError("Prediction fold/seed/horizon mismatch")
                for horizon in protocol['horizons']:
                    part = frame[frame.horizon.eq(horizon)].sort_values(['ticker', 'date']).reset_index(drop=True)
                    dates = pd.to_datetime(part.date, utc=True, errors='raise')
                    if dates.isna().any() or part.ticker.isna().any() or part.duplicated(['ticker', 'date']).any():
                        raise ValueError("Missing/duplicate prediction endpoint")
                    numeric = ['actual', 'prediction', 'zero', 'ridge'] + (['tree'] if 'tree' in part else [])
                    if not np.isfinite(part[numeric].to_numpy(dtype=np.float64)).all() or not part.zero.eq(0).all():
                        raise ValueError("Nonfinite prediction values or nonzero zero-return baseline")
                    saved = rows_by_key[(fold, seed, horizon)]
                    if len(part) != saved.get('samples'):
                        raise ValueError("Prediction count differs from frozen evaluation")
                    # Each seed must see exactly the same endpoints, labels and baselines.
                    signature = part[['ticker', 'date', 'actual', 'zero', 'ridge'] + (['tree'] if 'tree' in part else [])]
                    align_key = (fold, horizon)
                    if align_key in alignment and not signature.equals(alignment[align_key]):
                        raise ValueError("Seed endpoints/labels/baselines differ")
                    alignment[align_key] = signature
                    actual, predicted = [part[column].to_numpy(dtype=np.float64) for column in ('actual', 'prediction')]
                    metrics = regression_metrics(actual, predicted)
                    # Training labels are not in this ZIP. Never infer training
                    # majority direction from test labels or pretend to recheck it.
                    metrics.pop('nonflat_train_majority_accuracy')
                    if not np.isclose(metrics['mae'], saved.get('mae', np.nan), rtol=1e-6, atol=1e-9):
                        raise ValueError("Saved predictions do not reproduce pinned candidate MAE")
                    baseline_mae = {column: regression_metrics(actual, part[column].to_numpy(dtype=np.float64))['mae'] for column in ['zero', 'ridge'] + (['tree'] if 'tree' in part else [])}
                    if any(not np.isclose(value, saved.get(column + '_mae', np.nan), rtol=1e-6, atol=1e-9) for column, value in baseline_mae.items()):
                        raise ValueError("Saved predictions do not reproduce pinned baseline MAE")
                    corrected_row = {**saved, **metrics, 'metric_precision': 'float64',
                        'zero_mae': baseline_mae['zero'], 'ridge_mae': baseline_mae['ridge'], 'tree_mae': baseline_mae.get('tree'),
                        'mae_improvement_ci': paired_date_block_bootstrap(actual, predicted, np.zeros_like(actual), dates.to_numpy(),
                            **{key: protocol['bootstrap'][key] for key in ('block_sessions', 'repetitions')})}
                    corrected['folds'].append(corrected_row)
                    prediction_audit.append({'fold_id': fold, 'seed': seed, 'horizon': horizon,
                        'csv_sha256': hashlib.sha256(payload(name)).hexdigest(),
                        'constant_prediction': bool(np.ptp(predicted) == 0), 'all_zero_prediction': bool(np.all(predicted == 0)),
                        'saved_mae': saved['mae'], 'saved_zero_mae': saved['zero_mae'],
                        'corrected_mae': metrics['mae'], 'corrected_zero_mae': baseline_mae['zero'],
                        'mae_relative_improvement': 1 - metrics['mae'] / baseline_mae['zero'] if baseline_mae['zero'] else None})

    if sha256_file(path) != source_hash:
        raise ValueError("Source archive changed while being reviewed")
    gates = {str(horizon): evaluate_model_gate(corrected, protocol, horizon=horizon).as_dict() for horizon in protocol['horizons']}
    # Do not leave the stale original headline gate inside the corrected report.
    corrected['gate'] = gates['1']
    corrected['gates_by_horizon'] = gates
    corrected['gate_implementation_version'] = GATE_IMPLEMENTATION_VERSION
    return {'schema_version': 'lstm-v3-numerical-review-v1', 'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'gate_implementation_version': GATE_IMPLEMENTATION_VERSION,
        'source_archive': path.name, 'source_archive_sha256': source_hash,
        'source_release_identity': identity, 'predictor_identity': predictor, 'dataset_sha256': spec['dataset_sha256'],
        'pinned_file_hashes_verified': True, 'scope': 'DEVELOPMENT_WALK_FORWARD_RECOMPUTED_NOT_NEW_HOLDOUT',
        'original_gate': original.get('gate'), 'gates_by_horizon': gates,
        'research_diagnostics_by_horizon': {str(h): research_gate_diagnostics(corrected, protocol, horizon=h) for h in protocol['horizons']},
        'selected_model': {'fold_id': spec['selected_fold'], 'seed': spec['seed'], 'selected_using': spec.get('selected_using')},
        'prediction_audit': prediction_audit, 'corrected_evaluation': corrected,
        'limitations': ['Original ZIP, weights, manifest and frozen protocol were not modified.',
            'Prediction CSVs are not pinned by the original artifact manifest; this review hashes the received ZIP and checks consistency with pinned metrics, not model execution attestation.',
            'Float64 recomputation uses exported labels/predictions, not retraining or recovery of pre-export precision.',
            'Existing train-majority direction metrics are retained, not reconstructed from test labels.',
            'Historical viewed results are development evidence, not a new independent forward holdout.',
            'Numerical comparison tolerance fixes rounding equivalence; it is not a tuned economic improvement threshold.',
            'A separate review does not promote the artifact or change its Technical weight.']}


def render_numerical_review(review: dict) -> str:
    lines = ['# LSTM V3 — numerical re-evaluation', '',
        f"Archive: `{review['source_archive']}`", f"Gate implementation: `{review['gate_implementation_version']}`", '',
        'No retraining, artifact mutation or promotion was performed.', '',
        '| Horizon | Winning folds | Development gate | Validation status | Technical weight |',
        '| --- | --- | --- | --- | --- |']
    for horizon, gate in review['gates_by_horizon'].items():
        diagnostics = review['research_diagnostics_by_horizon'][horizon]
        lines.append(f"| {horizon} | {diagnostics['winning_folds']}/{len(diagnostics['folds'])} | {'PASS' if diagnostics['passed'] else 'NOT_PASSED'} | {gate['validation_status']} | {gate['technical_blend_weight']} |")
    selected = review['selected_model']
    lines += ['', f"## Selected model (fold {selected['fold_id']}, seed {selected['seed']})", '',
        'Selected using validation only; this table does not choose another seed using test.', '',
        '| Horizon | MAE (raw return) | Zero MAE | Relative MAE reduction | Direction accuracy incl. flat |',
        '| --- | --- | --- | --- | --- |']
    for row in review['corrected_evaluation']['folds']:
        if row['fold_id'] == selected['fold_id'] and row['seed'] == selected['seed']:
            improvement = 100 * (1 - row['mae'] / row['zero_mae']) if row['zero_mae'] else None
            lines.append(f"| {row['horizon']} | {row['mae']:.10f} | {row['zero_mae']:.10f} | {improvement:.4f}% | {100 * row['directional_accuracy']:.2f}% |" if improvement is not None else f"| {row['horizon']} | {row['mae']} | 0 | undefined | {row['directional_accuracy']} |")
    lines += ['', '## Blockers', '']
    for reason in dict.fromkeys(reason for gate in review['gates_by_horizon'].values() for reason in gate['reasons']):
        lines.append('- ' + reason)
    lines += ['', '## Limitations', ''] + ['- ' + reason for reason in review['limitations']]
    return '\n'.join(lines) + '\n'
