from __future__ import annotations

import os
from pathlib import Path


def load_workspace_env(workspace: Path | None = None) -> Path | None:
    """Load simple KEY=VALUE settings without overriding process secrets."""
    root = workspace or Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if not env_path.is_file():
        return None

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return env_path
