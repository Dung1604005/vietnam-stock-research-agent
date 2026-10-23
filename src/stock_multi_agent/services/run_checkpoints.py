"""Private, local SQLite checkpoints for synchronous analysis runs.

Uses LangGraph's checkpoint protocol (including pending writes), so resuming
does not replay successful siblings. No pickle or arbitrary module loading.
"""
from __future__ import annotations

import os
import sqlite3
import hashlib
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from threading import Lock

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple, WRITES_IDX_MAP, get_checkpoint_metadata
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import AnyUrl, BaseModel


_run_locks: dict[tuple[str, str], Lock] = {}
_run_locks_guard = Lock()


class _StateSerializer(JsonPlusSerializer):
    """Save project models as data; Pydantic rebuilds them at node boundaries.

    Nested HttpUrl values are converted by mode='json'. No project class or
    arbitrary module needs to be imported or executed while reading storage.
    """

    def __init__(self):
        super().__init__(allowed_msgpack_modules=(), allowed_json_modules=(), pickle_fallback=False)

    @classmethod
    def _plain(cls, value):
        if isinstance(value, BaseModel):
            return value.model_dump(mode='json')
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, AnyUrl):
            return str(value)
        if isinstance(value, dict):
            return {cls._plain(key): cls._plain(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._plain(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._plain(item) for item in value)
        if isinstance(value, (set, frozenset)):
            return type(value)(cls._plain(item) for item in value)
        return value

    def dumps_typed(self, obj):
        return super().dumps_typed(self._plain(obj))


def runtime_database(name: str) -> Path:
    root = Path(os.getenv('STOCK_AGENT_RUNTIME_DIR', Path(__file__).resolve().parents[3] / '.runtime'))
    if os.name == 'nt' and root.resolve().drive.upper() != 'D:':
        raise ValueError('Project run storage must be on drive D')
    return root / 'runs' / name


class SQLiteRunCheckpointer(BaseCheckpointSaver):
    """Connection-per-operation supports LangGraph's checkpoint worker thread."""

    def __init__(self, path: Path | None = None):
        super().__init__(serde=_StateSerializer())
        self.path = Path(path) if path is not None else runtime_database('analysis.sqlite3')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS run_checkpoints (
                    thread_id TEXT, ns TEXT, checkpoint_id TEXT, parent_id TEXT,
                    checkpoint_type TEXT, checkpoint BLOB, metadata_type TEXT, metadata BLOB,
                    PRIMARY KEY(thread_id, ns, checkpoint_id));
                CREATE TABLE IF NOT EXISTS run_writes (
                    thread_id TEXT, ns TEXT, checkpoint_id TEXT, task_id TEXT, idx INTEGER,
                    channel TEXT, type TEXT, value BLOB, task_path TEXT,
                    PRIMARY KEY(thread_id, ns, checkpoint_id, task_id, idx));
            ''')

    def _connect(self):
        return sqlite3.connect(self.path, timeout=15)

    @contextmanager
    def run_guard(self, run_id: str):
        """Reject concurrent execution of one run, including another process."""
        key = (str(self.path.resolve()), run_id)
        with _run_locks_guard:
            local = _run_locks.setdefault(key, Lock())
        if not local.acquire(blocking=False):
            raise RuntimeError('Analysis run is already executing')
        try:
            directory = self.path.with_name(self.path.name + '.locks')
            directory.mkdir(parents=True, exist_ok=True)
            lock_path = directory / (hashlib.sha256(run_id.encode()).hexdigest() + '.lock')
            with lock_path.open('a+b') as handle:
                if handle.tell() == 0:
                    handle.write(b'\0')
                    handle.flush()
                handle.seek(0)
                try:
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise RuntimeError('Analysis run is already executing') from exc
                try:
                    yield
                finally:
                    handle.seek(0)
                    if os.name == 'nt':
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            local.release()

    @staticmethod
    def _config(thread_id, ns, checkpoint_id):
        return {'configurable': {'thread_id': thread_id, 'checkpoint_ns': ns, 'checkpoint_id': checkpoint_id}}

    def get_tuple(self, config):
        c = config['configurable']
        thread_id, ns = c['thread_id'], c.get('checkpoint_ns', '')
        query = 'SELECT checkpoint_id, parent_id, checkpoint_type, checkpoint, metadata_type, metadata FROM run_checkpoints WHERE thread_id=? AND ns=?'
        params = [thread_id, ns]
        if c.get('checkpoint_id'):
            query += ' AND checkpoint_id=?'
            params.append(c['checkpoint_id'])
        with self._connect() as db:
            row = db.execute(query + ' ORDER BY checkpoint_id DESC LIMIT 1', params).fetchone()
            if row is None:
                return None
            checkpoint_id, parent_id, kind, payload, meta_kind, meta = row
            writes = db.execute('SELECT task_id, channel, type, value FROM run_writes WHERE thread_id=? AND ns=? AND checkpoint_id=? ORDER BY task_id, idx',
                                (thread_id, ns, checkpoint_id)).fetchall()
        return CheckpointTuple(config=self._config(thread_id, ns, checkpoint_id),
            checkpoint=self.serde.loads_typed((kind, payload)),
            metadata=self.serde.loads_typed((meta_kind, meta)),
            parent_config=self._config(thread_id, ns, parent_id) if parent_id else None,
            pending_writes=[(task, channel, self.serde.loads_typed((typ, value)))
                            for task, channel, typ, value in writes])

    def list(self, config, *, filter=None, before=None, limit=None):
        query = 'SELECT thread_id, ns, checkpoint_id FROM run_checkpoints WHERE 1=1'
        params = []
        if config:
            c = config['configurable']
            query += ' AND thread_id=?'
            params.append(c['thread_id'])
            for field, column in (('checkpoint_ns', 'ns'), ('checkpoint_id', 'checkpoint_id')):
                if field in c:
                    query += f' AND {column}=?'
                    params.append(c[field])
        if before and before['configurable'].get('checkpoint_id'):
            query += ' AND checkpoint_id<?'
            params.append(before['configurable']['checkpoint_id'])
        with self._connect() as db:
            rows = db.execute(query + ' ORDER BY checkpoint_id DESC', params).fetchall()
        count = 0
        for thread, ns, checkpoint in rows:
            if limit is not None and count >= limit:
                break
            item = self.get_tuple(self._config(thread, ns, checkpoint))
            if item and (not filter or all(item.metadata.get(k) == v for k, v in filter.items())):
                count += 1
                yield item

    def put(self, config, checkpoint, metadata, new_versions):
        c = config['configurable']
        thread_id, ns = c['thread_id'], c.get('checkpoint_ns', '')
        kind, payload = self.serde.dumps_typed(checkpoint)
        meta_kind, meta = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        with self._connect() as db:
            db.execute('INSERT OR REPLACE INTO run_checkpoints VALUES (?,?,?,?,?,?,?,?)',
                       (thread_id, ns, checkpoint['id'], c.get('checkpoint_id'), kind, payload, meta_kind, meta))
        return self._config(thread_id, ns, checkpoint['id'])

    def put_writes(self, config, writes, task_id, task_path=''):
        c = config['configurable']
        with self._connect() as db:
            for index, (channel, value) in enumerate(writes):
                idx = WRITES_IDX_MAP.get(channel, index)
                kind, payload = self.serde.dumps_typed(value)
                statement = 'INSERT OR REPLACE' if idx < 0 else 'INSERT OR IGNORE'
                db.execute(f'{statement} INTO run_writes VALUES (?,?,?,?,?,?,?,?,?)',
                    (c['thread_id'], c.get('checkpoint_ns', ''), c['checkpoint_id'], task_id,
                     idx, channel, kind, payload, task_path))

    def delete_thread(self, thread_id):
        with self._connect() as db:
            for table in ('run_checkpoints', 'run_writes'):
                db.execute(f'DELETE FROM {table} WHERE thread_id=?', (thread_id,))
