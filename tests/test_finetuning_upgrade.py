"""Offline engineering tests only; no GPU/model download or fabricated quality gate."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile

import pytest
from pydantic import ValidationError

from stock_multi_agent.models.chat import ParsedQuery, QueryIntent
from stock_multi_agent.models.llm import AgentLLMReview
from stock_multi_agent.models.finetuning import AdapterManifest, FineTuningExample
from stock_multi_agent.services.finetuning_dataset import (
    canonical_json, content_hash, make_synthetic_drafts, prepare_dataset,
    privacy_issues, redact_example, validate_json_schema,
)
from stock_multi_agent.services.llm_adapter_registry import load_approved_adapter
from stock_multi_agent.services.llm_evaluation import evaluate_adapter_pairs

ROOT = Path(__file__).resolve().parents[1]


def example(i=0, *, reviewed=True, task='nlu_routing'):
    return FineTuningExample(example_id=f'example-{i}', task=task,
        messages=[{'role': 'system', 'content': 'Return JSON only.'},
            {'role': 'user', 'content': f'Phân tích mã DEMO{i}'}],
        approved_output={'ticker': f'DEMO{i}', 'clarify': False},
        output_schema={'type': 'object', 'properties': {'ticker': {'type': 'string'},
            'clarify': {'type': 'boolean'}}, 'required': ['ticker', 'clarify'], 'additionalProperties': False},
        prompt_version='test-prompt', schema_version='test-schema', template_group=f'template-{i}',
        usage_rights='AUTHORED', consent_to_training=True, cloud_upload_allowed=True,
        approval_status='APPROVED' if reviewed else 'DRAFT',
        label_origin='human_reviewed' if reviewed else 'synthetic_draft',
        reviewer='fixture-curator' if reviewed else '',
        reviewed_at=datetime(2026, 1, 1, tzinfo=timezone.utc) if reviewed else None)


def cloud_module():
    spec = importlib.util.spec_from_file_location('qwen3_colab_offline_test', ROOT / 'notebooks/finetune_qwen3_4b_lora_v1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_pydantic_schema_refs_nullable_dates_and_enum():
    parsed = ParsedQuery(raw_query='Phân tích FPT', intent=QueryIntent.STOCK_ANALYSIS,
        tickers=['FPT'], as_of='2026-10-16')
    schema = ParsedQuery.model_json_schema()
    assert '$defs' in schema
    validate_json_schema(parsed.model_dump(mode='json'), schema)
    bad = parsed.model_dump(mode='json'); bad['as_of'] = 'yesterday'
    with pytest.raises(ValueError): validate_json_schema(bad, schema)
    bad['as_of'] = None; bad['intent'] = 'PLACE_ORDER'
    with pytest.raises(ValueError): validate_json_schema(bad, schema)
    review = AgentLLMReview(agent='fundamental', task='review', status='GROUNDED', assessment='Nhận định thử')
    validate_json_schema(review.model_dump(mode='json'), AgentLLMReview.model_json_schema())


@pytest.mark.parametrize('schema,value', [
    ({'type': 'integer'}, True), ({'enum': [1]}, True),
    ({'type': 'number'}, float('nan')), ({'type': 'array', 'uniqueItems': True}, [1, 1]),
    ({'$ref': 'https://example.com/schema'}, {}), ({'unevaluatedProperties': False}, {}),
    ({'type': 'string', 'format': 'date-time'}, '2026-10-18T12:30:00'),
    ({'type': 'string', 'format': 'uuid'}, 'not-a-uuid'),
])
def test_schema_rejects_invalid_or_unsupported(schema, value):
    with pytest.raises(ValueError): validate_json_schema(value, schema)


def test_schema_recursion_is_bounded():
    with pytest.raises(ValueError): validate_json_schema({}, {'$defs': {'node': {'$ref': '#/$defs/node'}}, '$ref': '#/$defs/node'})


@pytest.mark.parametrize('text', ['api_key=123456secret', 'Bearer abcdefghijklmnop',
    'hf_abcdefghijklmnop123456', 'person@example.com', 'D:\\private\\portfolio.csv',
    'https://example.com/data?token=123456secret'])
def test_privacy_rejects_sensitive_text(text):
    assert privacy_issues({'text': text})


def test_redaction_invalidates_old_approval():
    row = example().model_dump(mode='json'); row['messages'][1]['content'] = 'api_key=123456secret'
    redacted = redact_example(row)
    assert redacted['approval_status'] == 'DRAFT' and redacted['reviewed_at'] is None
    assert '[REDACTED]' in redacted['messages'][1]['content']
    assert not privacy_issues(redacted)


def test_approval_is_not_inferred_and_dates_are_aware():
    raw = example(reviewed=False).model_dump(mode='json'); raw['approval_status'] = 'APPROVED'
    with pytest.raises(ValidationError): FineTuningExample.model_validate(raw)
    raw = example().model_dump(mode='json'); raw['reviewed_at'] = '2026-01-01T00:00:00'
    with pytest.raises(ValidationError): FineTuningExample.model_validate(raw)


def test_grouped_datasets_stable_locked_and_no_leakage(tmp_path):
    rows = [example(i) for i in range(20)]
    rows[1].conversation_group = rows[0].conversation_group = 'same-conversation'
    rows[2].source_checksums = rows[1].source_checksums = ['same-document']
    manifest = prepare_dataset(rows, tmp_path / 'first', cloud_upload_opt_in=True)
    second = prepare_dataset(list(reversed(rows)), tmp_path / 'second', cloud_upload_opt_in=True)
    assert manifest.dataset_digest == second.dataset_digest
    assert all(manifest.overlap[key] == 0 for key in manifest.grouping_dimensions)
    assert manifest.promotion_eligible and manifest.cloud_upload_authorized
    assert prepare_dataset(rows, tmp_path / 'first').dataset_digest == manifest.dataset_digest
    schema_path = tmp_path / 'first/output-schema.json'; schema_path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='schema'): prepare_dataset(rows, tmp_path / 'first')


def test_connected_groups_not_randomly_row_split(tmp_path):
    rows = [example(i) for i in range(6)]
    for row in rows: row.template_group = 'one-template'
    with pytest.raises(ValueError, match='three connected groups'): prepare_dataset(rows, tmp_path)


def test_synthetic_demo_never_promotable_or_silently_approved(tmp_path):
    drafts = make_synthetic_drafts(128)
    with pytest.raises(ValueError): prepare_dataset(drafts, tmp_path / 'blocked')
    result = prepare_dataset(drafts, tmp_path / 'demo', allow_synthetic_demo=True)
    assert result.status == 'SYNTHETIC_DEMO_ONLY' and not result.promotion_eligible
    assert not result.cloud_upload_authorized
    assert all(row.approval_status == 'DRAFT' for row in drafts)


def test_cloud_upload_requires_per_example_consent(tmp_path):
    rows = [example(i) for i in range(3)]; rows[0].cloud_upload_allowed = False
    with pytest.raises(ValueError): prepare_dataset(rows, tmp_path, cloud_upload_opt_in=True)


def test_conflicting_duplicate_labels_refused(tmp_path):
    rows = [example(i) for i in range(4)]
    rows[1].messages = rows[0].messages
    with pytest.raises(ValueError, match='Conflicting labels'): prepare_dataset(rows, tmp_path)


def paired_inputs(*, reviewed=True, task='nlu_routing'):
    cases = [example(i, reviewed=reviewed, task=task) for i in range(12)]
    base = {row.example_id: {'raw_output': '{}'} for row in cases}
    candidate = {row.example_id: {'raw_output': canonical_json(row.approved_output)} for row in cases}
    manifest = {'status': 'REVIEWED_TRAINING_SET' if reviewed else 'SYNTHETIC_DEMO_ONLY',
        'promotion_eligible': reviewed, 'dataset_digest': 'd' * 64,
        'schema_hash': content_hash(cases[0].output_schema), 'prompt_hash': 'a' * 64}
    identity = {'base_model': 'Qwen/Qwen3-4B-Instruct-2507', 'base_revision': 'a' * 40,
        'tokenizer_revision': 'a' * 40, 'quantization': 'nf4-fp16', 'schema_hash': manifest['schema_hash'],
        'prompt_hash': manifest['prompt_hash'], 'chat_template_hash': 'b' * 64, 'decoding_hash': 'c' * 64}
    identities = {'baseline': identity, 'candidate': {**identity, 'adapter_id': 'test-adapter', 'adapter_files_digest': 'e' * 64}}
    protocol = {'seed': 42, 'promotion_gate': {'minimum_schema_validity': .98, 'minimum_improvement': .05,
        'min_test_cases': 10, 'min_groups': 10, 'bootstrap_samples': 100}}
    return cases, base, candidate, protocol, manifest, identities


def test_raw_paired_evaluation_and_grouped_ci():
    cases, base, candidate, protocol, manifest, identities = paired_inputs()
    report = evaluate_adapter_pairs(cases, base, candidate, protocol, dataset_manifest=manifest, identities=identities)
    assert report['status'] == 'PASSED' and report['primary_improvement'] == 1
    assert report['paired_group_bootstrap_ci'] == [1, 1]
    assert report['raw_model_metrics_only']


def test_fallback_cannot_hide_raw_failures():
    cases, base, candidate, protocol, manifest, identities = paired_inputs()
    candidate[cases[0].example_id]['used_fallback'] = True
    with pytest.raises(ValueError, match='fallback'): evaluate_adapter_pairs(cases, base, candidate, protocol)


def test_synthetic_and_synthesis_not_auto_promotable():
    for options in [{'reviewed': False}, {'task': 'grounded_synthesis'}]:
        cases, base, candidate, protocol, manifest, identities = paired_inputs(**options)
        report = evaluate_adapter_pairs(cases, base, candidate, protocol, dataset_manifest=manifest, identities=identities)
        assert report['status'] == 'INCONCLUSIVE'


def test_identity_difference_and_new_critical_error_block_promotion():
    cases, base, candidate, protocol, manifest, identities = paired_inputs()
    identities['candidate']['quantization'] = 'fp16'
    report = evaluate_adapter_pairs(cases, base, candidate, protocol, dataset_manifest=manifest, identities=identities)
    assert 'PAIRED_MODEL_IDENTITIES_NOT_VERIFIED' in report['reason_codes']
    candidate[cases[0].example_id]['raw_output'] = '{"execute_trade": true}'
    report = evaluate_adapter_pairs(cases, base, candidate, protocol, dataset_manifest=manifest, identities=identities)
    assert report['status'] == 'REJECTED' and 'CRITICAL_SAFETY_FAILURE' in report['reason_codes']


def test_cloud_package_path_traversal_blocked(tmp_path):
    module = cloud_module(); archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as zipped: zipped.writestr('../escape.py', 'bad')
    with pytest.raises(ValueError, match='five expected'): module.safe_extract(archive, tmp_path / 'extract')
    assert not (tmp_path / 'escape.py').exists()


def test_cloud_demo_and_reviewed_packages_validate_and_detect_tampering(tmp_path):
    pytest.importorskip('jsonschema')
    module = cloud_module()
    cloud = module.make_cloud_demo_package(tmp_path / 'cloud-demo')
    with pytest.raises(ValueError, match='ALLOW_SYNTHETIC_DEMO'): module.validate_dataset(cloud)
    manifest, schema, splits = module.validate_dataset(cloud, allow_synthetic_demo=True)
    assert manifest['promotion_eligible'] is False and sum(map(len, splits.values())) == 288
    reviewed = tmp_path / 'reviewed'; prepare_dataset([example(i) for i in range(10)], reviewed, cloud_upload_opt_in=True)
    assert module.validate_dataset(reviewed)[0]['status'] == 'REVIEWED_TRAINING_SET'
    archive = tmp_path / 'good.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        for name in ['manifest.json', 'output-schema.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl']:
            zipped.write(reviewed / name, name)
    extracted = module.safe_extract(archive, tmp_path / 'extracted')
    assert module.safe_extract(archive, extracted) == extracted
    (extracted / 'train.jsonl').write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='differs'): module.safe_extract(archive, extracted)
    with pytest.raises(ValueError, match='checksum'): module.validate_dataset(extracted)


def test_cloud_script_is_import_safe_and_notebook_contains_real_python():
    module = cloud_module()
    assert module.BASE_MODEL == 'Qwen/Qwen3-4B-Instruct-2507'
    notebook = json.loads((ROOT / 'notebooks/finetune_qwen3_4b_lora_v1.ipynb').read_text(encoding='utf-8'))
    code = [ ''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    for cell in code: ast.parse(cell)
    for i, cell in enumerate(code): compile(cell, f'notebook-code-cell-{i}', 'exec')
    assert any('def run_experiment' in cell for cell in code)
    assert any('run_saved_inference' in cell for cell in code)
    definitions = next(cell for cell in code if 'def run_experiment' in cell)
    original = (ROOT / 'notebooks/finetune_qwen3_4b_lora_v1.py').read_text(encoding='utf-8')
    assert definitions.rstrip() == original.split("\nif __name__ == '__main__':")[0].rstrip()


def make_registry_fixture(tmp_path):
    files = {'adapter_config.json': json.dumps({'base_model_name_or_path': 'Qwen/Qwen3-4B-Instruct-2507'}),
        'adapter_model.safetensors': 'fixture-safeweights-not-a-real-model'}
    hashes = {}
    for name, value in files.items():
        (tmp_path / name).write_text(value, encoding='utf-8')
        hashes[name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    identity = {'adapter_id': 'fixture', 'base_model': 'Qwen/Qwen3-4B-Instruct-2507',
        'base_revision': 'a' * 40, 'tokenizer_revision': 'a' * 40, 'schema_hash': 'b' * 64,
        'prompt_hash': 'c' * 64, 'chat_template_hash': 'd' * 64, 'quantization': 'nf4-fp16',
        'adapter_files_digest': content_hash(hashes)}
    report = {'schema_version': 'llm-ab-report-v1', 'status': 'PASSED', 'raw_model_metrics_only': True,
        'dataset_digest': 'e' * 64, 'identities': {'candidate': identity}}
    (tmp_path / 'evaluation.json').write_text(json.dumps(report), encoding='utf-8')
    manifest = AdapterManifest(adapter_id='fixture', base_revision='a' * 40, tokenizer_revision='a' * 40,
        output_schema_hash='b' * 64, prompt_hash='c' * 64, chat_template_hash='d' * 64,
        dataset_digest='e' * 64, dataset_status='REVIEWED_TRAINING_SET', task='nlu_routing',
        adapter_files=hashes, license_reviewed=True, approval_status='APPROVED', reviewer='fixture-curator',
        reviewed_at=datetime(2026, 1, 1, tzinfo=timezone.utc), evaluation_status='PASSED',
        evaluation_report='evaluation.json', evaluation_sha256=hashlib.sha256((tmp_path / 'evaluation.json').read_bytes()).hexdigest())
    path = tmp_path / 'manifest.json'; path.write_text(manifest.model_dump_json(), encoding='utf-8')
    kwargs = {'expected_base_model': manifest.base_model, 'expected_base_revision': manifest.base_revision,
        'expected_tokenizer_revision': manifest.tokenizer_revision, 'expected_schema_hash': manifest.output_schema_hash,
        'expected_prompt_hash': manifest.prompt_hash, 'expected_chat_template_hash': manifest.chat_template_hash,
        'expected_task': manifest.task}
    return path, manifest, kwargs


def test_registry_requires_exact_evaluated_weight_hash(tmp_path):
    path, manifest, kwargs = make_registry_fixture(tmp_path)
    assert load_approved_adapter(path, **kwargs).identity == manifest.identity
    (tmp_path / 'adapter_model.safetensors').write_text('different-weights', encoding='utf-8')
    manifest.adapter_files['adapter_model.safetensors'] = hashlib.sha256((tmp_path / 'adapter_model.safetensors').read_bytes()).hexdigest()
    path.write_text(manifest.model_dump_json(), encoding='utf-8')
    with pytest.raises(ValueError, match='different adapter'): load_approved_adapter(path, **kwargs)


def test_registry_rejects_synthetic_pending_and_paths(tmp_path):
    path, manifest, kwargs = make_registry_fixture(tmp_path)
    manifest.dataset_status = 'SYNTHETIC_DEMO_ONLY'; path.write_text(manifest.model_dump_json(), encoding='utf-8')
    with pytest.raises(ValueError, match='reviewed rights'): load_approved_adapter(path, **kwargs)
    manifest.dataset_status = 'REVIEWED_TRAINING_SET'; manifest.adapter_files['../outside'] = 'a' * 64
    path.write_text(manifest.model_dump_json(), encoding='utf-8')
    with pytest.raises(ValueError, match='inside'): load_approved_adapter(path, **kwargs)
