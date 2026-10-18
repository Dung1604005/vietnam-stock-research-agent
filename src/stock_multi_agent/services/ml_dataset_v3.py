"""Auditable V3 dataset builder. Original data and V2 are never modified."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from stock_multi_agent.services.ml_features_v3 import FEATURE_VERSION, FEATURES, add_targets, build_feature_frame, window_endpoints


@dataclass
class DatasetBuildResult:
    rows: pd.DataFrame
    quarantine: pd.DataFrame
    manifest: dict


RUNTIME_MODULES = ("ml_features_v3.py", "ml_dataset_v3.py", "ml_validation.py", "ml_baselines.py", "ml_model_registry.py")


def _write_runtime_bundle(bundle: Path) -> None:
    service_root = Path(__file__).resolve().parent
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("stock_multi_agent/__init__.py", '"""Portable V3 training runtime."""\n')
        archive.writestr("stock_multi_agent/services/__init__.py", '"""Portable causal ML modules."""\n')
        for filename in RUNTIME_MODULES:
            archive.write(service_root / filename, f"stock_multi_agent/services/{filename}")


def refresh_runtime_bundle(output_dir: Path) -> dict:
    """Explicit engineering refresh, preserving CSVs and the prior runtime ZIP.

    No training or performance result is rewritten. A content-addressed new
    runtime bundle and history entry make the preparation change inspectable.
    """
    manifest_path = output_dir / "dataset.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_name = manifest["files"]["dataset"]["filename"]
    if Path(dataset_name).name != dataset_name or sha256_file(output_dir / dataset_name) != manifest["files"]["dataset"]["sha256"]:
        raise ValueError("Cannot refresh a changed dataset or unsafe dataset filename")
    service_root = Path(__file__).resolve().parent
    code_hash = hashlib.sha256("".join(sha256_file(service_root / name) for name in RUNTIME_MODULES).encode()).hexdigest()
    bundle = output_dir / f"ml_v3_runtime.{code_hash[:12]}.zip"
    if manifest["runtime_bundle"].get("filename") == bundle.name:
        if sha256_file(bundle) != manifest["runtime_bundle"]["sha256"]:
            raise ValueError("Current runtime bundle was modified")
        return manifest
    _write_runtime_bundle(bundle)
    manifest.setdefault("runtime_bundle_history", []).append(dict(manifest["runtime_bundle"]))
    manifest["runtime_bundle"] = {"filename": bundle.name, "sha256": sha256_file(bundle), "code_sha256": code_hash}
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_dataset(raw: pd.DataFrame, *, window: int = 30, max_gap_business_days: int = 5) -> DatasetBuildResult:
    required = {"ticker", "date", "open", "high", "low", "close", "volume", "source", "source_id", "fetched_at", "price_adjustment"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing source data columns: {sorted(missing)}")
    frame = raw.copy()
    frame["input_row"] = np.arange(len(frame))
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["available_at"] = pd.to_datetime(frame["fetched_at"], errors="coerce", utc=True)
    frame["price_unit"] = frame.get("price_unit", "UNKNOWN")
    frame["price_unit"] = frame["price_unit"].fillna("UNKNOWN")
    frame["price_adjustment"] = frame["price_adjustment"].fillna("UNKNOWN").astype(str).str.upper()
    reason = pd.Series("", index=frame.index)
    prices = frame[["open", "high", "low", "close"]]
    bad = ~np.isfinite(frame[["open", "high", "low", "close", "volume"]]).all(axis=1)
    bad |= prices.le(0).any(axis=1) | frame["volume"].lt(0)
    bad |= frame["high"].lt(prices[["open", "low", "close"]].max(axis=1))
    bad |= frame["low"].gt(prices[["open", "high", "close"]].min(axis=1))
    reason.loc[bad] = "INVALID_OHLCV"
    bad_metadata = frame["date"].isna() | frame["available_at"].isna() | ~frame["ticker"].str.fullmatch(r"[A-Z][A-Z0-9]{1,9}")
    for column in ("source", "source_id"):
        bad_metadata |= frame[column].isna() | frame[column].astype(str).str.strip().eq("")
    bad_metadata |= frame["date"].gt(frame["available_at"].dt.tz_localize(None).dt.normalize())
    reason.loc[bad_metadata] = "INVALID_DATE_OR_PROVENANCE"
    # Identical duplicates are explicitly reconciled; conflicting bars all quarantine.
    identity = ["ticker", "date"]
    exact = frame.duplicated(identity + ["open", "high", "low", "close", "volume", "source_id", "price_adjustment", "price_unit"], keep="first")
    reason.loc[exact & reason.eq("")] = "EXACT_DUPLICATE"
    candidates = frame[reason.eq("")]
    conflict_ids = candidates.index[candidates.duplicated(identity, keep=False)]
    reason.loc[conflict_ids] = "CONFLICTING_DUPLICATE"
    frame["quarantine_reason"] = reason
    quarantine = frame[reason.ne("")].copy()
    clean = frame[reason.eq("")].copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    if clean.empty:
        raise ValueError("No valid source rows remain")
    # A removed bad bar must break the next segment even if the calendar gap is short.
    bad_dates = quarantine[~quarantine["quarantine_reason"].eq("EXACT_DUPLICATE")].dropna(subset=["date"])
    breaks = clean["ticker"].ne(clean["ticker"].shift())
    previous = clean.groupby("ticker")["date"].shift(1)
    good_previous = previous.notna()
    gaps = np.zeros(len(clean), dtype=np.int64)
    gaps[good_previous] = np.busday_count(previous[good_previous].to_numpy(dtype="datetime64[D]"), clean.loc[good_previous, "date"].to_numpy(dtype="datetime64[D]"))
    breaks |= gaps > max_gap_business_days
    for _, indices in clean.groupby("ticker").groups.items():
        ticker = clean.loc[indices[0], "ticker"]
        removed = bad_dates.loc[bad_dates["ticker"].eq(ticker), "date"].to_numpy(dtype="datetime64[ns]")
        if len(removed):
            dates = clean.loc[indices, "date"].to_numpy(dtype="datetime64[ns]")
            removed = np.sort(removed)
            positions = np.searchsorted(dates, removed, side="right")
            positions = positions[positions < len(indices)]
            breaks.loc[np.asarray(indices)[positions]] = True
    # Mixing provider scale/adjustment regimes inside a window is not allowed.
    for column in ("price_unit", "price_adjustment", "source"):
        breaks |= clean[column].ne(clean.groupby("ticker")[column].shift())
    clean["segment_id"] = breaks.cumsum().astype(int)
    clean = add_targets(build_feature_frame(clean, max_gap_business_days=max_gap_business_days))
    endpoints = window_endpoints(clean, window)
    clean["training_endpoint"] = False
    clean.loc[endpoints, "training_endpoint"] = True
    manifest = {
        "schema_version": "lstm-dataset-v3", "feature_version": FEATURE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(), "window": window, "features": FEATURES,
        "input_rows": len(raw), "retained_rows": len(clean), "quarantined_rows": len(quarantine),
        "reconciled_counts": len(clean) + len(quarantine) == len(raw),
        "quarantine_reasons": quarantine["quarantine_reason"].value_counts().to_dict(),
        "training_endpoints": len(endpoints), "tickers": int(clean.ticker.nunique()),
        "segments": int(clean.segment_id.nunique()), "max_gap_business_days": max_gap_business_days,
        "label_policy": "Observed bars within a segment; no future-return magnitude filtering",
        "known_at_policy": "available_at is actual fetched_at; historical forecast_origin is simulated close, not point-in-time availability",
        "calendar_policy": "Observed source bars; conservative weekday gap segmentation, not certified exchange calendar",
        "adjustment_counts": clean.price_adjustment.value_counts().to_dict(),
        "price_unit_counts": clean.price_unit.value_counts().to_dict(),
        "universe_policy": "Source current-universe snapshot; survivorship bias not resolved",
        "validation_status": "DEVELOPMENT_ONLY", "policy_eligibility": "SHADOW_ONLY",
    }
    return DatasetBuildResult(clean, quarantine, manifest)


def write_dataset(result: DatasetBuildResult, output_dir: Path, *, source_path: Path | None = None) -> dict:
    # Refuse a partially populated directory too: a failed earlier export must
    # not be mixed with fresh CSVs or a stale shared-code ZIP.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("V3 output directory is not empty; choose a new --output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"dataset": output_dir / "vci_ohlcv_lstm_v3.csv", "quarantine": output_dir / "quarantine.csv", "manifest": output_dir / "dataset.manifest.json"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("V3 output exists; choose a new --output directory to preserve the experiment")
    result.rows.to_csv(paths["dataset"], index=False)
    result.quarantine.to_csv(paths["quarantine"], index=False)
    manifest = dict(result.manifest)
    manifest["files"] = {key: {"filename": path.name, "sha256": sha256_file(path)} for key, path in paths.items() if key != "manifest"}
    if source_path:
        manifest["source_file"] = {"filename": source_path.name, "sha256": sha256_file(source_path)}
    # Portable shared code, without importing the full app/dependencies on Kaggle.
    # The upload is private/user-controlled; raw reports and credentials are absent.
    bundle = output_dir / "ml_v3_runtime.zip"
    _write_runtime_bundle(bundle)
    manifest["runtime_bundle"] = {"filename": bundle.name, "sha256": sha256_file(bundle)}
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
