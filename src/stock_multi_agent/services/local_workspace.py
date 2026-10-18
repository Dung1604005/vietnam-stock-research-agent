"""Small, local-only watchlist and notes store (never orders or credentials)."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import string
import tempfile
import threading
import time
from typing import Iterator
from urllib.parse import urlsplit

from stock_multi_agent.services.interaction_actions import validate_broker_template


class WorkspaceError(RuntimeError):
    """Actionable local workspace persistence/validation error."""


_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_PREFERENCE_KEYS = {"broker_name", "broker_url_template"}
_MAX_NOTE = 10_000


def normalize_ticker(ticker: str) -> str:
    if not isinstance(ticker, str):
        raise WorkspaceError("Mã chứng khoán phải là chuỗi ký tự.")
    normalized = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9]{1,9}", normalized):
        raise WorkspaceError("Mã chứng khoán phải có 2–10 chữ cái/chữ số, bắt đầu bằng chữ cái.")
    return normalized


def _validate_preferences(preferences: dict) -> dict[str, str]:
    if not isinstance(preferences, dict) or set(preferences) - _PREFERENCE_KEYS:
        raise WorkspaceError("Chỉ lưu broker_name và broker_url_template; không lưu thông tin đăng nhập.")
    for key, value in preferences.items():
        if not isinstance(value, str):
            raise WorkspaceError(f"{key} phải là chuỗi ký tự.")
    name = preferences.get("broker_name", "")
    template = preferences.get("broker_url_template", "")
    if len(name) > 80 or len(template) > 2048:
        raise WorkspaceError("Tên công ty chứng khoán hoặc URL quá dài.")
    if template:
        try:
            fields = list(string.Formatter().parse(template))
            if any(field not in (None, "ticker") or spec or conversion for _, field, spec, conversion in fields):
                raise ValueError("Only {ticker} is allowed")
            parsed = urlsplit(template.format(ticker="FPT"))
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("HTTPS URL without credentials required")
            # Apply the same navigation-only policy even when callers bypass UI.
            # This also revalidates persisted data before it reaches the browser.
            validate_broker_template(template)
        except (ValueError, KeyError, IndexError) as exc:
            raise WorkspaceError("URL công ty chứng khoán phải là HTTPS, không chứa thông tin đăng nhập; chỉ hỗ trợ {ticker}.") from exc
    return dict(preferences)


class LocalWorkspace:
    """Atomic JSON storage, serialized across threads and cooperating processes.

    Notes are plain text. Render them with text widgets, never unsafe HTML.
    Reads validate existing JSON and never replace a corrupt/incompatible file.
    """

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            default = Path(__file__).resolve().parents[3] / ".runtime"
            root = Path(os.getenv("STOCK_AGENT_RUNTIME_DIR", str(default)))
            path = root / "interaction" / "workspace.json"
        self.path = Path(path).resolve()
        with _LOCKS_GUARD:
            self._thread_lock = _LOCKS.setdefault(str(self.path).casefold(), threading.RLock())

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # Keep the sidecar: unlinking a live lock file can split lock ownership.
                with self.path.with_suffix(self.path.suffix + ".lock").open("a+b") as handle:
                    handle.seek(0, os.SEEK_END)
                    if not handle.tell():
                        handle.write(b"0")
                        handle.flush()
                    acquired = False
                    deadline = time.monotonic() + 5
                    while not acquired:
                        try:
                            handle.seek(0)
                            if os.name == "nt":
                                import msvcrt
                                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                            else:
                                import fcntl
                                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            acquired = True
                        except OSError:
                            if time.monotonic() >= deadline:
                                raise WorkspaceError("Workspace đang được sử dụng; hãy thử lại sau vài giây.")
                            time.sleep(0.025)
                    try:
                        yield
                    finally:
                        handle.seek(0)
                        if os.name == "nt":
                            import msvcrt
                            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError as exc:
                raise WorkspaceError(f"Không thể đọc/ghi workspace tại {self.path}. Kiểm tra quyền truy cập và dung lượng ổ đĩa.") from exc

    def _read(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "tickers": [], "notes": {}, "preferences": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or set(data) != {"version", "tickers", "notes", "preferences"} or type(data["version"]) is not int or data["version"] != 1:
                raise ValueError("Unsupported workspace schema")
            tickers, notes = data["tickers"], data["notes"]
            if not isinstance(tickers, list) or len(tickers) > 1000 or any(normalize_ticker(ticker) != ticker for ticker in tickers) or len(set(tickers)) != len(tickers):
                raise ValueError("Invalid watchlist")
            if not isinstance(notes, dict) or len(notes) > 1000:
                raise ValueError("Invalid notes")
            for ticker, note in notes.items():
                if normalize_ticker(ticker) != ticker or not isinstance(note, str) or len(note) > _MAX_NOTE:
                    raise ValueError("Invalid note")
            _validate_preferences(data["preferences"])
            return data
        except (ValueError, TypeError, WorkspaceError) as exc:
            raise WorkspaceError(f"Workspace JSON lỗi hoặc không tương thích: {self.path}. File được giữ nguyên; hãy sao lưu và sửa/đổi tên file trước khi thử lại.") from exc

    def _write(self, data: dict) -> None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, prefix=".workspace-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def list_tickers(self) -> list[str]:
        with self._locked():
            return list(self._read()["tickers"])

    def add_ticker(self, ticker: str) -> None:
        ticker = normalize_ticker(ticker)
        with self._locked():
            data = self._read()
            if ticker not in data["tickers"]:
                if len(data["tickers"]) >= 1000:
                    raise WorkspaceError("Danh sách theo dõi đã đạt giới hạn 1.000 mã.")
                data["tickers"].append(ticker)
                self._write(data)

    def remove_ticker(self, ticker: str) -> None:
        ticker = normalize_ticker(ticker)
        with self._locked():
            data = self._read()
            if ticker in data["tickers"]:
                data["tickers"].remove(ticker)
                # Removing a favorite does not destroy its personal note.
                self._write(data)

    def get_note(self, ticker: str) -> str:
        ticker = normalize_ticker(ticker)
        with self._locked():
            return self._read()["notes"].get(ticker, "")

    def save_note(self, ticker: str, note: str) -> None:
        ticker = normalize_ticker(ticker)
        if not isinstance(note, str) or len(note) > _MAX_NOTE:
            raise WorkspaceError("Ghi chú phải là văn bản, tối đa 10.000 ký tự.")
        with self._locked():
            data = self._read()
            if note:
                if ticker not in data["notes"] and len(data["notes"]) >= 1000:
                    raise WorkspaceError("Đã đạt giới hạn ghi chú cho 1.000 mã.")
                data["notes"][ticker] = note
            else:
                data["notes"].pop(ticker, None)
            self._write(data)

    def get_preferences(self) -> dict[str, str]:
        with self._locked():
            return dict(self._read()["preferences"])

    def save_preferences(self, preferences: dict) -> None:
        validated = _validate_preferences(preferences)
        with self._locked():
            data = self._read()
            data["preferences"] = validated
            self._write(data)
