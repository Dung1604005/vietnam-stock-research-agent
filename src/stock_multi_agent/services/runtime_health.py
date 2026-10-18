"""Cheap local readiness checks. Constructing or rendering this report never does I/O over network."""
from __future__ import annotations

import os
import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def runtime_health(runtime_root=None, *, provider=None, gateway=None, artifact_root=None) -> dict:
    """runtime_root means the .runtime directory itself, not the project root."""
    root = Path(runtime_root or os.getenv('STOCK_AGENT_RUNTIME_DIR', Path(__file__).resolve().parents[3] / '.runtime'))
    checks = []
    checks.append({'component': 'runtime', 'status': 'READY' if root.exists() else 'NOT_CREATED',
                   'detail': 'D-drive storage' if root.drive.upper() == 'D:' else 'Local runtime storage'})
    if os.name == 'nt' and root.resolve().drive.upper() != 'D:':
        checks[-1].update(status='WARNING', detail='Runtime is not on required drive D')
    probe_path = root if root.exists() else root.parent
    try:
        checks.append({'component': 'disk', 'status': 'READY' if shutil.disk_usage(probe_path).free > 512 * 1024 * 1024 else 'LOW_SPACE',
                       'free_bytes': shutil.disk_usage(probe_path).free})
    except OSError:
        checks.append({'component': 'disk', 'status': 'UNKNOWN'})
    database = root / 'research' / 'research.sqlite3'
    if database.exists():
        try:
            with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=.1)) as db:
                tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                counts = {}
                for name in ('research_documents', 'research_chunks', 'research_chunk_embeddings'):
                    if name in tables:
                        counts[name] = db.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]
                expected = {'research_documents', 'research_chunks'}
                checks.append({'component': 'research_store',
                    'status': 'READY' if expected.issubset(tables) else 'SCHEMA_MISMATCH', 'counts': counts})
        except sqlite3.Error:
            checks.append({'component': 'research_store', 'status': 'UNAVAILABLE'})
    else:
        checks.append({'component': 'research_store', 'status': 'NOT_CREATED'})
    checks.append({'component': 'market_provider', 'status': 'UNKNOWN' if provider is None or not getattr(provider, 'last_success_at', None) else 'OBSERVED_SUCCESS',
        'detail': 'Configuration/last observation only; no live quote requested'})
    llm = {'component': 'llm', 'status': 'CONFIGURED_UNPROBED' if os.getenv('STOCK_AGENT_LLM_PROVIDER', '').lower() not in {'', 'none', 'disabled'} else 'DISABLED',
        'detail': 'No generation or remote health request performed'}
    if gateway is not None and callable(getattr(gateway, 'health', None)):
        # Existing gateway.health reads only its lock-protected local observations.
        try:
            observed = gateway.health()
            llm.update(status=observed.get('state', 'UNKNOWN'),
                consecutive_failures=observed.get('consecutive_failures', 0),
                model=str(observed.get('model', 'UNKNOWN'))[:120])
        except Exception:
            # A missing optional service must never crash local readiness/UI.
            llm.update(status='UNKNOWN', detail='Local gateway observations unavailable; no remote probe attempted')
    checks.append(llm)
    artifact = Path(artifact_root) if artifact_root is not None else Path(__file__).resolve().parents[3] / 'notebooks' / 'trainResult'
    manifest = artifact / 'manifest.json'
    if manifest.exists() and manifest.stat().st_size < 128000:
        try:
            specification = json.loads(manifest.read_text(encoding='utf-8'))
            declared = specification.get('files', {})
            valid_names = isinstance(declared, dict) and bool(declared) and all(isinstance(name, str) and Path(name).name == name
                and '/' not in name and '\\' not in name and name not in {'.', '..'} for name in declared)
            files_present = valid_names and all((artifact / name).is_file() for name in declared)
            checks.append({'component': 'lstm_v2', 'status': 'PRESENT_UNLOADED' if files_present else 'INCOMPLETE_ARTIFACT',
                'detail': 'Manifest/files inspected only; model file hashes and TensorFlow loading are not checked by health',
                'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'declared_files_present': bool(files_present), 'model_hashes_verified': False,
                'validation_status': specification.get('validation_status', 'UNKNOWN')})
        except (OSError, ValueError, AttributeError, TypeError):
            checks.append({'component': 'lstm_v2', 'status': 'INVALID_MANIFEST'})
    else:
        checks.append({'component': 'lstm_v2', 'status': 'NOT_PRESENT'})
    checks.append({'component': 'neo4j', 'status': 'OPTIONAL_UNPROBED', 'detail': 'Not required for local deterministic demo'})
    return {'schema_version': 'runtime-health-v1', 'checked_at': datetime.now(timezone.utc).isoformat(),
            'external_calls': 0, 'checks': checks,
            'limitations': ['Readiness is local configuration/observation, not a live service guarantee.']}


class RuntimeHealthService:
    def __init__(self, runtime_root=None):
        self.runtime_root = runtime_root

    def snapshot(self, **kwargs):
        return runtime_health(self.runtime_root, **kwargs)
