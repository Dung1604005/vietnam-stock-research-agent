"""V4 one-session-priority experiment. Import is safe: no TensorFlow/training.

The ready .ipynb embeds this script, the pinned protocol and shared runtime.
Reuse the existing V3 CSV+manifest; no extra dataset or ZIP upload is required.
Historical results are DEVELOPMENT, not a newly independent holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def build_model(tf, window, feature_count, config, *, variant='priority_multitask'):
    inputs = tf.keras.Input((window, feature_count), name='market_sequence')
    history = tf.keras.layers.LSTM(32, recurrent_dropout=0., kernel_regularizer=tf.keras.regularizers.l2(config['l2']))(inputs)
    last = tf.keras.layers.Flatten()(tf.keras.layers.Cropping1D((window - 1, 0))(inputs))
    last = tf.keras.layers.Dense(16, activation='relu')(last)
    combined = tf.keras.layers.Concatenate()([history, last])
    combined = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(config['l2']))(combined)
    combined = tf.keras.layers.Dropout(config['dropout'])(combined)
    outputs = {'return_1': tf.keras.layers.Dense(1, name='return_1')(combined),
        'return_5': tf.keras.layers.Dense(1, name='return_5')(combined),
        'direction_1': tf.keras.layers.Dense(3, activation='softmax', name='direction_1')(combined)}
    model = tf.keras.Model(inputs, outputs, name='stock_lstm_v4_one_session_priority')
    weights = dict(config['loss_weights'])
    if variant == 'regression_only':
        weights['direction_1'] = 0.
    model.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate'], clipnorm=1.),
        loss={'return_1': tf.keras.losses.Huber(delta=config['huber_delta']),
            'return_5': tf.keras.losses.Huber(delta=config['huber_delta']),
            'direction_1': tf.keras.losses.SparseCategoricalCrossentropy()}, loss_weights=weights,
        metrics={'return_1': ['mae'], 'return_5': ['mae'], 'direction_1': ['accuracy']})
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--variant', choices=['priority_multitask', 'regression_only'], default='priority_multitask')
    args = parser.parse_args(argv)
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from stock_multi_agent.services.ml_dataset_v3 import sha256_file
    from stock_multi_agent.services.ml_features_v3 import window_endpoints, window_batch, FEATURE_VERSION as V3_FEATURE_VERSION, FEATURES as V3_FEATURES
    from stock_multi_agent.services.ml_validation import TrainOnlyScaler, regression_metrics, paired_date_block_bootstrap
    from stock_multi_agent.services.ml_model_registry import GATE_IMPLEMENTATION_VERSION, evaluate_model_gate, research_gate_diagnostics
    from stock_multi_agent.services.ml_v4 import (FEATURES, FEATURE_VERSION, feature_frame, eligible_session_folds,
        origin_scales, direction_labels, direction_class_weights, split_calibration, fit_calibration,
        apply_calibration, classification_metrics, baseline_predictions, scaler_specification)

    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if protocol.get('feature_version') != FEATURE_VERSION or protocol.get('horizons') != [1, 5] or protocol.get('window') != 30:
        raise ValueError('V4 protocol/feature mismatch')
    configuration = protocol['training']
    manifest_path = args.data.parent / 'dataset.manifest.json'
    metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
    dataset_hash = sha256_file(args.data)
    if metadata.get('schema_version') != 'lstm-dataset-v3' or metadata.get('feature_version') != V3_FEATURE_VERSION or metadata.get('features') != V3_FEATURES or metadata['files']['dataset']['filename'] != args.data.name or dataset_hash != metadata['files']['dataset']['sha256']:
        raise ValueError('Attach the unchanged V3 CSV with its matching original manifest')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Use a new output folder, or resume the original config explicitly')
    if args.resume and not (args.output / 'run-config.json').is_file():
        raise ValueError('Resume requires original run-config.json and checkpoints')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = {'dataset_sha256': dataset_hash, 'protocol_sha256': sha256_file(args.protocol),
        'trainer_sha256': sha256_file(Path(__file__)), 'feature_version': FEATURE_VERSION,
        'gate_implementation_version': GATE_IMPLEMENTATION_VERSION,
        'runtime_sha256': os.getenv('STOCK_AGENT_V4_RUNTIME_SHA256', 'LOCAL_SOURCE'),
        'pilot': args.pilot, 'variant': args.variant, 'tensorflow': tf.__version__, 'numpy': np.__version__}
    lock_path = args.output / 'run-config.json'
    if lock_path.exists() and json.loads(lock_path.read_text(encoding='utf-8')) != lock:
        raise ValueError('Dataset/code/runtime/protocol/config changed: start a new experiment')
    lock_path.write_text(json.dumps(lock, indent=2), encoding='utf-8')
    if (args.output / 'manifest.json').exists():
        raise FileExistsError('Run already finished; do not retune/rewrite the completed result')
    print('GPU:', tf.config.list_physical_devices('GPU'), '| V4 feature build; no training yet.', flush=True)
    frame = feature_frame(pd.read_csv(args.data, parse_dates=['date', 'target_end_1', 'target_end_5']), reuse_verified_v3_features=True)
    endpoints = window_endpoints(frame, protocol['window'])
    folds = eligible_session_folds(frame, endpoints, folds=protocol['outer_folds'], gap_sessions=protocol['gap_sessions'], minimum_sessions=protocol['minimum_eligible_sessions'])
    targets = frame[['target_return_1', 'target_return_5']].to_numpy(dtype=np.float64)
    scales = origin_scales(frame, floor=configuration['volatility_floor'], ceiling=configuration['volatility_ceiling'])
    normalized = np.clip(targets / scales, -configuration['training_target_clip'], configuration['training_target_clip']).astype(np.float32)
    # Warm-up/unlabelled rows cannot be batched; only known finite endpoints are used.
    classes = np.ones(len(frame), dtype=np.int32)
    classes[endpoints] = direction_labels(targets[endpoints, 0])
    report = {'schema_version': 'lstm-evaluation-v4', 'protocol_id': protocol['protocol_id'],
        'dataset_sha256': dataset_hash, 'scope': 'DEVELOPMENT_AFTER_V3_REVIEW_NOT_INDEPENDENT_HOLDOUT',
        'pilot': bool(args.pilot or metadata.get('pilot_row_limit')), 'variant': args.variant,
        'metric_precision': 'float64', 'gate_implementation_version': GATE_IMPLEMENTATION_VERSION,
        'price_adjustment': 'ADJUSTED' if frame.price_adjustment.eq('ADJUSTED').all() else 'UNKNOWN',
        'engineering_checks': {}, 'forward': {}, 'folds': [], 'splits': [fold.manifest() for fold in folds],
        'feature_count': len(FEATURES), 'endpoint_count': len(endpoints),
        'limitations': protocol['limitations'] + ['No automatic V4 integration; software tests do not constitute reviewed model engineering evidence.', 'Classifier probabilities are uncalibrated, not trade confidence.']}
    print('Features:', len(FEATURES), '| endpoints:', len(endpoints), flush=True)
    for split in report['splits']:
        print('SPLIT:', json.dumps(split), flush=True)

    def predict(model, values, indices):
        pieces = {name: [] for name in ('return_1', 'return_5', 'direction_1')}
        for offset in range(0, len(indices), configuration['batch_size']):
            batch = indices[offset:offset + configuration['batch_size']]
            result = model.predict_on_batch(window_batch(values, batch, protocol['window']))
            if not isinstance(result, dict):
                raise ValueError('Expected named V4 outputs, not positional guesses')
            for name in pieces:
                pieces[name].append(np.asarray(result[name], dtype=np.float64))
        result = {name: np.concatenate(value) for name, value in pieces.items()}
        result['returns'] = np.column_stack([result['return_1'][:, 0], result['return_5'][:, 0]]) * scales[indices]
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError('Nonfinite model outputs')
        return result

    class Batches(tf.keras.utils.Sequence):
        def __init__(self, values, indices, class_weights, *, shuffle=False, seed=42):
            super().__init__()
            self.values, self.indices, self.class_weights = values, np.array(indices), class_weights
            self.shuffle, self.rng = shuffle, np.random.default_rng(seed)
            self.on_epoch_end()
        def __len__(self):
            return int(np.ceil(len(self.indices) / configuration['batch_size']))
        def __getitem__(self, number):
            indices = self.indices[number * configuration['batch_size']:(number + 1) * configuration['batch_size']]
            ones = np.ones(len(indices), dtype=np.float32)
            labels = {'return_1': normalized[indices, [0]][:, None], 'return_5': normalized[indices, [1]][:, None], 'direction_1': classes[indices]}
            weights = {'return_1': ones, 'return_5': ones, 'direction_1': self.class_weights[classes[indices]]}
            return window_batch(self.values, indices, protocol['window']), labels, weights
        def on_epoch_end(self):
            if self.shuffle:
                self.rng.shuffle(self.indices)

    class RawValidationMAE(tf.keras.callbacks.Callback):
        def __init__(self, values, indices):
            super().__init__()
            self.values, self.indices = values, indices
        def on_epoch_end(self, epoch, logs=None):
            result = predict(self.model, self.values, self.indices)
            score = regression_metrics(targets[self.indices, 0], result['returns'][:, 0])['mae']
            if logs is not None:
                logs['val_raw_return_1_mae'] = score
            print(f' | raw next-session validation MAE: {score:.8f}', flush=True)

    started, selected, best_selection = time.perf_counter(), None, float('inf')
    seeds = [protocol['seeds'][0]] if args.pilot else protocol['seeds']
    for fold in folds:
        scaler = TrainOnlyScaler().fit(frame.iloc[fold.train][FEATURES].to_numpy(float))
        values = scaler.transform(frame[FEATURES].to_numpy(float))
        fit_indices, interval_indices = split_calibration(frame, fold.calibration)
        baselines = {h: baseline_predictions(frame, fold, h) for h in protocol['horizons']}
        class_weights = direction_class_weights(classes[fold.train])
        for seed in seeds:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            trial = args.output / f'fold-{fold.fold_id}-seed-{seed}'
            trial.mkdir(exist_ok=True)
            checkpoint, result_path = trial / 'best.keras', trial / 'results.json'
            if args.resume and result_path.exists():
                saved = json.loads(result_path.read_text(encoding='utf-8'))
                if saved['checkpoint_sha256'] != sha256_file(checkpoint) or saved['predictions_sha256'] != sha256_file(trial / 'predictions.csv'):
                    raise ValueError('Completed trial checkpoint/predictions changed')
                report['folds'].extend(saved['results'])
                if fold.fold_id == folds[-1].fold_id and saved['selection_mae'] < best_selection:
                    best_selection, selected = saved['selection_mae'], saved['spec']
                    import shutil
                    shutil.copy2(checkpoint, args.output / 'model.keras')
                continue
            model = tf.keras.models.load_model(checkpoint, safe_mode=True) if args.resume and checkpoint.exists() else build_model(tf, protocol['window'], len(FEATURES), configuration, variant=args.variant)
            resume_score = None
            if args.resume and checkpoint.exists():
                resume_score = regression_metrics(targets[fold.validation, 0], predict(model, values, fold.validation)['returns'][:, 0])['mae']
            train_indices = fold.train
            if args.pilot and len(train_indices) > 50000:
                train_indices = np.random.default_rng(seed).choice(train_indices, 50000, replace=False)
            monitor = 'val_raw_return_1_mae'
            callbacks = [RawValidationMAE(values, fold.validation),
                tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor=monitor, mode='min', save_best_only=True, initial_value_threshold=resume_score),
                tf.keras.callbacks.EarlyStopping(monitor=monitor, mode='min', baseline=resume_score, patience=configuration['early_stopping_patience'], restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode='min', patience=3, factor=.5, min_lr=1e-5)]
            print('Training fold/seed:', fold.fold_id, seed, flush=True)
            history = model.fit(Batches(values, train_indices, class_weights, shuffle=True, seed=seed),
                validation_data=Batches(values, fold.validation, np.ones(3, dtype=np.float32)),
                epochs=3 if args.pilot else configuration['epochs'], callbacks=callbacks, verbose=2)
            pd.DataFrame(history.history).to_csv(trial / 'history.csv', index=False)
            model = tf.keras.models.load_model(checkpoint, safe_mode=True)
            calibration_raw = predict(model, values, fit_indices)
            interval_raw = predict(model, values, interval_indices)
            # All choices above were frozen before looking at this trial's outer test.
            test_raw = predict(model, values, fold.test)
            trial_rows, csv_rows, calibration = [], [], {}
            for column, horizon in enumerate(protocol['horizons']):
                fitted = fit_calibration(targets[fit_indices, column], calibration_raw['returns'][:, column],
                    calibration_raw['direction_1'][:, 1], allow_flat_gate=horizon == 1 and args.variant != 'regression_only')
                interval_prediction = apply_calibration(interval_raw['returns'][:, column], interval_raw['direction_1'][:, 1], fitted)
                radius = float(np.quantile(np.abs(targets[interval_indices, column] - interval_prediction), .9, method='higher'))
                calibration[str(horizon)] = {**fitted, 'absolute_residual_quantile_90': radius,
                    'interval_fit_scope': 'PURGED_DISJOINT_CALIBRATION_INTERVAL_BLOCK',
                    'calibration_fit_samples': len(fit_indices), 'interval_samples': len(interval_indices)}
                prediction = apply_calibration(test_raw['returns'][:, column], test_raw['direction_1'][:, 1], fitted)
                actual = targets[fold.test, column]
                references, baseline_details = baselines[horizon]
                train_signs = np.sign(targets[fold.train, column])
                majority = 1 if np.count_nonzero(train_signs > 0) >= np.count_nonzero(train_signs < 0) else -1
                metrics = regression_metrics(actual, prediction, train_direction=majority)
                reference_metrics = {name: regression_metrics(actual, value, train_direction=majority) for name, value in references.items()}
                row = {'fold_id': fold.fold_id, 'seed': seed, 'horizon': horizon, **metrics,
                    'zero_mae': reference_metrics['zero']['mae'], 'ridge_mae': reference_metrics['ridge']['mae'],
                    'tree_mae': reference_metrics.get('tree', {}).get('mae'), 'tree_status': baseline_details['tree_status'],
                    'all_baselines': reference_metrics, 'calibration_alpha': fitted['alpha'],
                    'flat_threshold': fitted['flat_threshold'], 'uses_lstm_signal': fitted['alpha'] > 0 and bool(np.ptp(test_raw['returns'][:, column]) > 1e-12),
                    'mae_improvement_ci': paired_date_block_bootstrap(actual, prediction, references['zero'], frame.iloc[fold.test].date.to_numpy(),
                        **{name: protocol['bootstrap'][name] for name in ('block_sessions', 'repetitions')}),
                    'interval_coverage': float(np.mean(np.abs(actual - prediction) <= radius)), 'interval_average_width_return': 2 * radius,
                    'interval_status': 'EMPIRICAL_DISJOINT_CALIBRATION_NO_FUTURE_COVERAGE_GUARANTEE',
                    'split': fold.manifest(), 'subgroups': {}}
                for label, mask in {'active100': frame.iloc[fold.test].active_ratio_20.eq(1).to_numpy(),
                    'low_activity': frame.iloc[fold.test].active_ratio_20.lt(1).to_numpy(), 'flat_target': np.abs(actual) <= 1e-12,
                    'nonflat_target': np.abs(actual) > 1e-12}.items():
                    row['subgroups'][label] = regression_metrics(actual[mask], prediction[mask], train_direction=majority) if mask.any() else None
                if horizon == 1:
                    row['direction_head'] = classification_metrics(actual, test_raw['direction_1'])
                trial_rows.append(row)
                part = frame.iloc[fold.test][['ticker', 'date', 'target_end_1', 'target_end_5', 'active_ratio_20']].copy()
                part['actual'], part['prediction'], part['raw_prediction'] = actual, prediction, test_raw['returns'][:, column]
                part['horizon'], part['fold_id'], part['seed'] = horizon, fold.fold_id, seed
                part['flat_probability'] = test_raw['direction_1'][:, 1]
                part['majority_direction'] = majority
                for name, value in references.items():
                    part[name] = value
                csv_rows.append(part)
            pd.concat(csv_rows).to_csv(trial / 'predictions.csv', index=False)
            selection_mae = float(min([*history.history[monitor], *([] if resume_score is None else [resume_score])]))
            spec = {'schema_version': 'lstm-spec-v4', 'model_version': 'lstm-one-session-priority-v4',
                'protocol_id': protocol['protocol_id'], 'feature_version': FEATURE_VERSION, 'features': FEATURES,
                'window': protocol['window'], 'horizons': protocol['horizons'], 'scaler': scaler_specification(scaler),
                'origin_scale': {'feature': 'volatility_20', 'floor': configuration['volatility_floor'], 'ceiling': configuration['volatility_ceiling'], 'horizon_multiplier': [1., float(np.sqrt(5.))]},
                'output_heads': ['return_1', 'return_5', 'direction_1'], 'class_order': ['DOWN', 'FLAT', 'UP'],
                'calibration': calibration, 'seed': seed, 'selected_fold': fold.fold_id,
                'selected_using': 'LATEST_FOLD_RAW_ONE_SESSION_VALIDATION_MAE_ONLY', 'dataset_sha256': dataset_hash,
                'max_gap_business_days': metadata['max_gap_business_days'],
                'price_adjustment': report['price_adjustment'], 'price_unit': str(frame.price_unit.iloc[0]) if frame.price_unit.nunique() == 1 else 'MIXED_UNKNOWN'}
            result_path.write_text(json.dumps({'results': trial_rows, 'selection_mae': selection_mae, 'spec': spec,
                'checkpoint_sha256': sha256_file(checkpoint), 'predictions_sha256': sha256_file(trial / 'predictions.csv')}, indent=2, allow_nan=False), encoding='utf-8')
            report['folds'].extend(trial_rows)
            if fold.fold_id == folds[-1].fold_id and selection_mae < best_selection:
                best_selection, selected = selection_mae, spec
                import shutil
                shutil.copy2(checkpoint, args.output / 'model.keras')
    report['elapsed_seconds'] = time.perf_counter() - started
    report['gates_by_horizon'] = {str(h): evaluate_model_gate(report, protocol, horizon=h).as_dict() for h in protocol['horizons']}
    report['research_diagnostics'] = {str(h): research_gate_diagnostics(report, protocol, horizon=h) for h in protocol['horizons']}
    # V4 is deliberately NOT an app-serving artifact. Preserve research gate
    # results but do not equate successful serialization with integration.
    for gate in report['gates_by_horizon'].values():
        gate.update(policy_eligibility=False, technical_blend_weight=0.)
    for name, value in [('spec.json', selected), ('evaluation.json', report), ('protocol.json', protocol)]:
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    import shutil
    shutil.copy2(Path(__file__), args.output / 'training-source.py')
    runtime_path = os.getenv('STOCK_AGENT_V4_RUNTIME_PATH', '')
    if runtime_path:
        if sha256_file(Path(runtime_path)) != lock['runtime_sha256']:
            raise ValueError('Runtime bundle changed during training')
        shutil.copy2(runtime_path, args.output / 'runtime-source.zip')
    files = {str(path.relative_to(args.output)).replace('\\', '/'): sha256_file(path) for path in args.output.rglob('*') if path.is_file()}
    artifact = {'schema_version': 'lstm-artifact-v4', 'model_version': 'lstm-one-session-priority-v4',
        'files': files, 'artifact_identity': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        'predictor_identity': hashlib.sha256(json.dumps({name: files[name] for name in ('model.keras', 'spec.json', 'protocol.json')}, sort_keys=True).encode()).hexdigest(),
        'default_policy': 'EXPERIMENTAL_SHADOW_ZERO_WEIGHT_NO_AUTO_INTEGRATION',
        'prediction_csvs_pinned': True, 'code_config': lock}
    (args.output / 'manifest.json').write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print(json.dumps({'gates_by_horizon': report['gates_by_horizon'], 'output': str(args.output), 'elapsed_seconds': report['elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
