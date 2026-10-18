"""Fail-closed approved-adapter registry. This module never loads GPU libraries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from stock_multi_agent.models.finetuning import AdapterManifest
from stock_multi_agent.services.finetuning_dataset import content_hash


def _confined_file(root: Path, name: str) -> Path:
    unresolved = root / name
    candidate = unresolved.resolve()
    if not candidate.is_relative_to(root.resolve()) or candidate == root.resolve() or not candidate.is_file() or unresolved.is_symlink():
        raise ValueError('Adapter manifest file path must stay inside the approved artifact directory')
    return candidate


def load_approved_adapter(manifest_path, *, expected_base_model, expected_base_revision,
    expected_tokenizer_revision, expected_schema_hash, expected_prompt_hash,
    expected_chat_template_hash=None, expected_task=None) -> AdapterManifest:
    path = Path(manifest_path)
    manifest = AdapterManifest.model_validate_json(path.read_text(encoding='utf-8'))
    expected = {'base_model': expected_base_model, 'base_revision': expected_base_revision,
        'tokenizer_revision': expected_tokenizer_revision, 'output_schema_hash': expected_schema_hash,
        'prompt_hash': expected_prompt_hash}
    if expected_chat_template_hash is not None:
        expected['chat_template_hash'] = expected_chat_template_hash
    if expected_task is not None:
        expected['task'] = expected_task
    if any(not value or getattr(manifest, name) != value for name, value in expected.items()):
        raise ValueError('Adapter base/tokenizer/prompt/schema/task compatibility mismatch')
    if (manifest.approval_status != 'APPROVED' or manifest.evaluation_status != 'PASSED'
        or manifest.dataset_status != 'REVIEWED_TRAINING_SET' or not manifest.license_reviewed
        or not manifest.reviewer.strip() or manifest.reviewed_at is None
        or manifest.reviewed_at.tzinfo is None or manifest.reviewed_at.utcoffset() is None):
        raise ValueError('Adapter requires reviewed rights, human approval and passing independent evaluation')
    root = path.parent.resolve()
    if not {'adapter_config.json', 'adapter_model.safetensors'}.issubset(manifest.adapter_files):
        raise ValueError('Adapter requires safe weights and a hashed PEFT config')
    for name, expected_hash in manifest.adapter_files.items():
        if hashlib.sha256(_confined_file(root, name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Adapter file checksum mismatch')
    peft_config = json.loads(_confined_file(root, 'adapter_config.json').read_text(encoding='utf-8'))
    if peft_config.get('base_model_name_or_path') != manifest.base_model:
        raise ValueError('PEFT config base model mismatch')
    report_path = _confined_file(root, manifest.evaluation_report)
    if hashlib.sha256(report_path.read_bytes()).hexdigest() != manifest.evaluation_sha256:
        raise ValueError('Evaluation report checksum mismatch')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    candidate = report.get('identities', {}).get('candidate', {})
    if (report.get('schema_version') != 'llm-ab-report-v1' or report.get('status') != 'PASSED'
        or not report.get('raw_model_metrics_only') or report.get('dataset_digest') != manifest.dataset_digest
        or candidate.get('adapter_id') != manifest.adapter_id or candidate.get('base_revision') != manifest.base_revision
        or candidate.get('schema_hash') != manifest.output_schema_hash or candidate.get('prompt_hash') != manifest.prompt_hash):
        raise ValueError('Evaluation report does not approve this artifact and contract')
    if any(candidate.get(field) != getattr(manifest, field) for field in
        ['base_model', 'tokenizer_revision', 'quantization', 'chat_template_hash']):
        raise ValueError('Evaluated serving identity does not match adapter manifest')
    if candidate.get('adapter_files_digest') != content_hash(manifest.adapter_files):
        raise ValueError('Evaluation report was produced for different adapter weights/config')
    if manifest.task == 'grounded_synthesis' and not report.get('blind_human_review_passed'):
        raise ValueError('Synthesis adapter requires a blind human review attestation')
    return manifest
