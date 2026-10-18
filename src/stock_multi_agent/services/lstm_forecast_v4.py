"""V4 inference from a hash-verified local archive. Never executes bundled code.

The UI's explicit experimental blend is separate from research eligibility.
No training, downloads, or modification of the original V2/V4 files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
from threading import Lock
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from stock_multi_agent.models.analysis import MarketData
from stock_multi_agent.models.forecast import ForecastStatus, PriceForecast
from stock_multi_agent.services.ml_features_v3 import window_endpoints
from stock_multi_agent.services.ml_v4 import FEATURES, FEATURE_VERSION, feature_frame, origin_scales, apply_calibration


class LstmV4ArtifactError(RuntimeError):
    pass


class LstmV4ForecastModel:
    def __init__(self, archive: Path, *, experimental_blend: bool = False, keras_model: Any = None):
        self.archive = Path(archive).resolve()
        self.experimental_blend = experimental_blend
        self._model = keras_model
        self._lock = Lock()
        try:
            with zipfile.ZipFile(self.archive) as bundle:
                names = bundle.namelist()
                if len(names) != len(set(names)):
                    raise ValueError('Duplicate ZIP entries')
                manifests = [name for name in names if PurePosixPath(name).name == 'manifest.json']
                if len(manifests) != 1:
                    raise ValueError('Expected exactly one V4 manifest')
                self.prefix = manifests[0][:-len('manifest.json')]
                self.manifest = json.loads(bundle.read(manifests[0]))
                if self.manifest.get('schema_version') != 'lstm-artifact-v4':
                    raise ValueError('Unsupported V4 manifest schema')
                files = self.manifest['files']
                if not {'model.keras', 'spec.json', 'protocol.json', 'evaluation.json'} <= set(files):
                    raise ValueError('Missing pinned V4 files')
                for name, expected in files.items():
                    path = PurePosixPath(name)
                    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
                        raise ValueError('Unsafe manifest path')
                    info = bundle.getinfo(self.prefix + name)
                    if info.file_size > 512 * 1024 * 1024:
                        raise ValueError('Oversized artifact member')
                    digest = hashlib.sha256()
                    with bundle.open(info) as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b''):
                            digest.update(block)
                    if digest.hexdigest() != expected:
                        raise ValueError(f'Artifact checksum mismatch: {name}')
                identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
                predictor = {name: files[name] for name in ('model.keras', 'spec.json', 'protocol.json')}
                if self.manifest.get('artifact_identity') != identity or self.manifest.get('predictor_identity') != hashlib.sha256(json.dumps(predictor, sort_keys=True).encode()).hexdigest():
                    raise ValueError('V4 identity mismatch')
                self.spec = json.loads(bundle.read(self.prefix + 'spec.json'))
                self.report = json.loads(bundle.read(self.prefix + 'evaluation.json'))
                self.protocol = json.loads(bundle.read(self.prefix + 'protocol.json'))
                self._model_bytes = bundle.read(self.prefix + 'model.keras')
            if self.spec.get('schema_version') != 'lstm-spec-v4' or self.spec.get('features') != FEATURES or self.spec.get('feature_version') != FEATURE_VERSION:
                raise ValueError('Unsupported V4 feature contract')
            if self.spec['protocol_id'] != self.protocol['protocol_id'] or self.report['protocol_id'] != self.protocol['protocol_id']:
                raise ValueError('V4 protocol mismatch')
            self.version = self.spec['model_version']
            self.window = int(self.spec['window'])
            if not 1 <= self.window <= 512 or self.spec.get('max_gap_business_days', 5) != 5:
                raise ValueError('Unsupported window/gap contract')
            if self.spec.get('output_heads') != ['return_1', 'return_5', 'direction_1'] or self.spec.get('class_order') != ['DOWN', 'FLAT', 'UP']:
                raise ValueError('Unsupported output contract')
            scaler = self.spec['scaler']
            if scaler.get('fit_scope') != 'TRAIN_ONLY' or scaler.get('features') != FEATURES:
                raise ValueError('Invalid scaler scope/order')
            self.mean, self.std, self.lower, self.upper = [np.asarray(scaler[key], dtype=np.float64) for key in ('mean', 'std', 'lower', 'upper')]
            if any(a.shape != (len(FEATURES),) or not np.isfinite(a).all() for a in (self.mean, self.std, self.lower, self.upper)) or (self.std <= 0).any() or (self.lower > self.upper).any():
                raise ValueError('Invalid scaler values')
            scale = self.spec['origin_scale']
            if scale.get('feature') != 'volatility_20' or not 0 < scale['floor'] <= scale['ceiling'] or not np.isfinite([scale['floor'], scale['ceiling']]).all():
                raise ValueError('Invalid origin scale')
            self.calibration = self.spec['calibration']['1']
            apply_calibration(np.array([0.]), np.array([0.]), self.calibration)
            self.radius = float(self.calibration['absolute_residual_quantile_90'])
            if not np.isfinite(self.radius) or self.radius < 0:
                raise ValueError('Invalid residual interval')
        except Exception as exc:
            raise LstmV4ArtifactError(f'V4 artifact rejected: {exc}') from exc

    @staticmethod
    def default_archive() -> Path:
        directory = Path(__file__).resolve().parents[3] / '.runtime' / 'ml' / 'v4'
        matches = sorted(directory.glob('lstm-v4-full-*-results.zip'))
        if len(matches) != 1:
            raise LstmV4ArtifactError('Cần đúng một ZIP V4 full trong .runtime/ml/v4 hoặc cấu hình STOCK_AGENT_LSTM_V4_ARCHIVE')
        return matches[0]

    def _load_model(self):
        import tensorflow as tf
        # Extract ONLY the verified model, never Python/runtime source. Temporary
        # generated binary stays on the same drive as the archive (D locally).
        with tempfile.TemporaryDirectory(prefix='v4-inference-', dir=self.archive.parent) as directory:
            path = Path(directory) / 'model.keras'
            path.write_bytes(self._model_bytes)
            return tf.keras.models.load_model(path, compile=False, safe_mode=True)

    def predict_next_session(self, market_data: MarketData) -> PriceForecast:
        rows = [{'ticker': market_data.ticker, **bar.model_dump()} for bar in market_data.bars]
        frame = feature_frame(pd.DataFrame(rows))
        endpoints = window_endpoints(frame, self.window, require_targets=False)
        if not len(endpoints) or endpoints[-1] != len(frame) - 1:
            raise LstmV4ArtifactError('Latest bar lacks a contiguous eligible V4 window')
        values = frame.iloc[-self.window:][FEATURES].to_numpy(dtype=np.float64)
        values = ((np.clip(values, self.lower, self.upper) - self.mean) / self.std).astype(np.float32)
        with self._lock:
            if self._model is None:
                self._model = self._load_model()
            output = self._model.predict_on_batch(values[None, :, :])
        if not isinstance(output, dict) or set(output) != {'return_1', 'return_5', 'direction_1'}:
            raise LstmV4ArtifactError('Expected named V4 output heads')
        result = {name: np.asarray(value, dtype=np.float64) for name, value in output.items()}
        if any(not np.isfinite(a).all() for a in result.values()) or result['return_1'].shape != (1, 1) or result['return_5'].shape != (1, 1) or result['direction_1'].shape != (1, 3):
            raise LstmV4ArtifactError('Invalid V4 output shape/value')
        probability = result['direction_1'][0]
        if (probability < 0).any() or (probability > 1).any() or not np.isclose(probability.sum(), 1., atol=1e-5):
            raise LstmV4ArtifactError('Invalid V4 probabilities')
        scale = self.spec['origin_scale']
        raw_return = result['return_1'][:, 0] * origin_scales(frame, floor=scale['floor'], ceiling=scale['ceiling'])[-1, 0]
        expected = float(apply_calibration(raw_return, probability[1:2], self.calibration)[0])
        if not np.isfinite(expected) or expected <= -1:
            raise LstmV4ArtifactError('Invalid V4 calibrated return')
        price = market_data.bars[-1].close
        estimate = price * (1 + expected)
        return PriceForecast(ticker=market_data.ticker, as_of=market_data.bars[-1].date,
            current_price=price, forecast_price=estimate,
            lower_bound=max(1e-6, estimate - price * self.radius), upper_bound=estimate + price * self.radius,
            status=ForecastStatus.TRAINED_MODEL, method='V4 causal features + origin-volatility scaling + flat-aware calibration',
            model_version=self.version, confidence=.1, artifact_status='VERIFIED_FILES',
            validation_status='PENDING_FORWARD_VALIDATION', policy_eligibility=False,
            experimental_blend_enabled=self.experimental_blend,
            technical_blend_weight=.2 if self.experimental_blend else 0.,
            warnings=['V4 thử nghiệm: chưa qua forward validation; giá điều chỉnh/đơn vị nguồn chưa được xác minh.',
                'Trọng số 20% là lựa chọn demo của người dùng, không phải chứng nhận model đủ điều kiện đầu tư.' if self.experimental_blend else 'V4 shadow: không tham gia điểm Technical.',
                'Khoảng residual calibration không bảo đảm độ bao phủ tương lai.'])
