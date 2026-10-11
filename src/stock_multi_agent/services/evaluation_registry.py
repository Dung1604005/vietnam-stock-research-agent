from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EvaluationRegistry:
    """Append-only local reports with version/config/input lineage, stored on drive D."""

    def __init__(self, root: Path | None = None) -> None:
        default_root = Path(__file__).resolve().parents[3] / ".runtime" / "evaluations"
        self.root = (root or default_root).resolve()
        if os.name == "nt" and self.root.drive.upper() != "D:":
            raise ValueError(f"Evaluation registry must stay on drive D:, got {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, report: dict[str, Any], input_versions: dict[str, str], config: dict[str, Any]) -> Path:
        generated_at = datetime.now(timezone.utc)
        canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        envelope = {
            "schema_version": "evaluation-registry-v1",
            "kind": kind,
            "generated_at": generated_at.isoformat(),
            "input_versions": input_versions,
            "config": config,
            "code_revision": os.getenv("GIT_COMMIT", "unknown"),
            "report_sha256": hashlib.sha256(canonical).hexdigest(),
            "report": report,
        }
        stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
        destination = self.root / f"{kind}-{stamp}-{envelope['report_sha256'][:12]}.json"
        destination.write_text(json.dumps(envelope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return destination
