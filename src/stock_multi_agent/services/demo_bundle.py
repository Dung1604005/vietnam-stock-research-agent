"""Explicit local JSON replay bundles; no URL fetch, pickle, archive or extraction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from stock_multi_agent.models.demo_bundle import DemoBundle
from stock_multi_agent.services.audit_export import sanitize_payload


MAX_BUNDLE_BYTES = 64 * 1024 * 1024


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


class DemoBundleService:
    @staticmethod
    def capture(datasets, destination, *, rights_note: str,
                recorded_forecasts=None, redistribution_allowed=False, allow_test_fixture=False) -> Path:
        values = datasets.values() if isinstance(datasets, dict) else datasets
        payloads = {}
        for data in values:
            if data.ticker in payloads:
                raise ValueError('Duplicate replay ticker; capture one unambiguous dataset per symbol')
            if not allow_test_fixture and ('fixture' in data.source.lower() or data.source.lower().startswith('mock')):
                raise ValueError('Synthetic fixture data cannot be captured as a real replay bundle')
            payload = data.model_dump(mode='json')
            # Preserve original source identity/time/hash but never export provider request credentials.
            for source in payload['source_refs']:
                source['request_parameters'] = {}
                source['snapshot_id'] = None
            payloads[data.ticker] = sanitize_payload(payload)
        bundle = DemoBundle(datasets=payloads, rights_note=rights_note,
            recorded_forecasts=sanitize_payload(recorded_forecasts or {}),
            redistribution_allowed=redistribution_allowed)
        payload = bundle.model_dump(mode='json')
        checksum = hashlib.sha256(_canonical(payload)).hexdigest()
        envelope = {'bundle_sha256': checksum, 'payload': payload}
        encoded = _canonical(envelope)
        if len(encoded) > MAX_BUNDLE_BYTES:
            raise ValueError('Replay bundle exceeds size limit; capture a smaller universe')
        target = Path(destination)
        if target.suffix.lower() != '.json':
            raise ValueError('Replay destination must be a local .json file')
        if target.exists():
            raise FileExistsError('Replay destination already exists; choose a new filename')
        if __import__('os').name == 'nt' and target.resolve().drive.upper() != 'D:':
            raise ValueError('Local replay bundles must be stored on drive D')
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids overwriting a previously captured bundle.
        with target.open('xb') as stream:
            stream.write(encoded)
        return target

    @staticmethod
    def load(path) -> DemoBundle:
        target = Path(path)
        if target.suffix.lower() != '.json' or target.stat().st_size > MAX_BUNDLE_BYTES:
            raise ValueError('Expected a bounded local replay JSON file')
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('Duplicate JSON key in replay bundle')
                result[key] = value
            return result
        raw = json.loads(target.read_text(encoding='utf-8'), object_pairs_hook=unique_keys)
        if set(raw) != {'bundle_sha256', 'payload'}:
            raise ValueError('Invalid replay envelope')
        calculated = hashlib.sha256(_canonical(raw['payload'])).hexdigest()
        if raw['bundle_sha256'] != calculated:
            raise ValueError('Replay checksum mismatch: bundle rejected')
        return DemoBundle.model_validate(raw['payload'])


capture_demo_bundle = DemoBundleService.capture
load_demo_bundle = DemoBundleService.load
