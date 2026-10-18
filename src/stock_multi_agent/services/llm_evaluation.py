"""Paired raw-output evaluation. Fallback success must never hide model failures."""
from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from typing import Any

from stock_multi_agent.models.finetuning import FineTuningExample
from stock_multi_agent.services.finetuning_dataset import content_hash, privacy_issues, validate_json_schema

ANCHOR_FIELDS = {'ticker', 'tickers', 'issuer', 'issuer_id', 'period', 'period_end', 'unit', 'source_id', 'source_ids', 'cited_source_ids', 'value'}


def _raw_prediction(record):
    if isinstance(record, str):
        raw, fallback = record, False
    else:
        raw, fallback = record.get('raw_output', ''), record.get('used_fallback', False)
    if fallback:
        raise ValueError('Raw A/B files must not contain deterministic fallback outputs')
    try:
        return json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite JSON')))
    except (ValueError, TypeError):
        return None


def _anchors(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = prefix + '.' + key
            if key in ANCHOR_FIELDS:
                result[path] = item
            result.update(_anchors(item, path))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            result.update(_anchors(item, prefix + f'[{i}]'))
    return result


def _case_score(case, record):
    raw = _raw_prediction(record)
    expected = case['approved_output']
    valid = raw is not None
    if valid:
        try:
            validate_json_schema(raw, case['output_schema'])
        except ValueError:
            valid = False
    exact = valid and content_hash(raw) == content_hash(expected)
    expected_anchors = _anchors(expected)
    actual_anchors = _anchors(raw) if raw is not None else {}
    # Compare complete path sets too: extra unsupported anchors are failures.
    anchors_ok = valid and actual_anchors == expected_anchors
    expected_abstain = bool(expected.get('clarify') or expected.get('abstain'))
    actual_abstain = bool(raw.get('clarify') or raw.get('abstain')) if isinstance(raw, dict) else False
    abstention_ok = valid and expected_abstain == actual_abstain
    critical = bool(privacy_issues(raw)) if raw is not None else False
    if isinstance(raw, dict):
        # A research routing/extraction output must never invent trade execution permissions.
        critical = critical or any(raw.get(key) for key in ['execute_trade', 'place_order', 'broker_order'])
    return {'schema_valid': int(valid), 'exact': int(exact), 'anchors_correct': int(anchors_ok),
            'abstention_correct': int(abstention_ok), 'critical': int(critical)}


def _group(case):
    return case.get('template_group') or case.get('document_family') or case.get('conversation_group') or case['example_id']


def _connected_eval_groups(cases):
    """Use the same connected components as dataset split, not only first group field."""
    parent = list(range(len(cases)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, case in enumerate(cases):
        keys = [(name, case.get(name)) for name in ['template_group', 'document_family', 'conversation_group'] if case.get(name)]
        keys.extend(('checksum', value) for value in case.get('source_checksums', []))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    return [str(find(i)) for i in range(len(cases))]


def evaluate_adapter_pairs(cases, baseline, candidate, protocol, *, dataset_manifest=None, identities=None) -> dict[str, Any]:
    """No network. Inputs are frozen cases and separately captured raw generations."""
    cases = [FineTuningExample.model_validate(case).model_dump(mode='json') for case in cases]
    if not cases or len({case['example_id'] for case in cases}) != len(cases):
        raise ValueError('Nonempty unique case IDs required')
    expected_ids = {case['example_id'] for case in cases}
    if set(baseline) != expected_ids or set(candidate) != expected_ids:
        raise ValueError('A/B must contain exactly the same complete frozen test case IDs')
    if any(case.get('approval_status') != 'APPROVED' or case.get('label_origin') != 'human_reviewed' for case in cases):
        reviewed = False
    else:
        reviewed = True
    scores = {name: [_case_score(case, predictions[case['example_id']]) for case in cases] for name, predictions in [('baseline', baseline), ('candidate', candidate)]}
    aggregates = {name: {key: statistics.mean(row[key] for row in rows) for key in rows[0]} for name, rows in scores.items()}
    differences = [cand['exact'] - base['exact'] for base, cand in zip(scores['baseline'], scores['candidate'])]
    group_ids = _connected_eval_groups(cases)
    by_group = defaultdict(list)
    for group, difference in zip(group_ids, differences):
        by_group[group].append(difference)
    gate = protocol['promotion_gate']
    samples = max(100, min(int(gate.get('bootstrap_samples', 2000)), 10000))
    rng = random.Random(protocol.get('seed', 42))
    groups = list(by_group.values())
    bootstrap = []
    for _ in range(samples):
        selected = [rng.choice(groups) for _ in groups]
        bootstrap.append(sum(sum(items) for items in selected) / sum(len(items) for items in selected))
    bootstrap.sort()
    alpha = (1 - gate.get('confidence', .95)) / 2
    low, high = bootstrap[int(samples * alpha)], bootstrap[min(samples - 1, int(samples * (1 - alpha))) ]
    improvement = statistics.mean(differences)
    reasons = []
    failure = False
    if aggregates['candidate']['schema_valid'] < gate['minimum_schema_validity']:
        reasons.append('RAW_SCHEMA_VALIDITY_BELOW_GATE'); failure = True
    for key in ['anchors_correct', 'abstention_correct']:
        if aggregates['candidate'][key] < aggregates['baseline'][key] - gate.get('safety_regression_tolerance', 0):
            reasons.append(key.upper() + '_REGRESSION'); failure = True
    new_critical = sum(c['critical'] > b['critical'] for b, c in zip(scores['baseline'], scores['candidate']))
    if new_critical or aggregates['candidate']['critical']:
        reasons.append('CRITICAL_SAFETY_FAILURE'); failure = True
    if improvement < gate['minimum_improvement']:
        reasons.append('PRIMARY_IMPROVEMENT_BELOW_GATE')
    if low <= 0:
        reasons.append('GROUPED_CI_NOT_POSITIVE')
    if len(cases) < gate['min_test_cases'] or len(groups) < gate['min_groups']:
        reasons.append('INSUFFICIENT_INDEPENDENT_CASES_OR_GROUPS')
    if not reviewed or not dataset_manifest or dataset_manifest.get('status') != 'REVIEWED_TRAINING_SET':
        reasons.append('HUMAN_REVIEW_OR_REVIEWED_DATASET_MISSING')
    if dataset_manifest and not dataset_manifest.get('promotion_eligible'):
        reasons.append('DATASET_NOT_PROMOTION_ELIGIBLE')
    if any(case['task'] == 'grounded_synthesis' for case in cases):
        reasons.append('BLIND_HUMAN_SYNTHESIS_REVIEW_PENDING')
    identity_ok = bool(identities)
    if identities:
        base, cand = identities.get('baseline', {}), identities.get('candidate', {})
        for field in ['base_model', 'base_revision', 'tokenizer_revision', 'quantization', 'decoding_hash', 'schema_hash', 'prompt_hash', 'chat_template_hash']:
            if not base.get(field) or base.get(field) != cand.get(field):
                identity_ok = False
        if not cand.get('adapter_id') or not cand.get('adapter_files_digest'):
            identity_ok = False
        if dataset_manifest and (base.get('schema_hash') != dataset_manifest.get('schema_hash')
            or base.get('prompt_hash') != dataset_manifest.get('prompt_hash')):
            identity_ok = False
    if not identity_ok:
        reasons.append('PAIRED_MODEL_IDENTITIES_NOT_VERIFIED')
    status = 'REJECTED' if failure else 'INCONCLUSIVE' if reasons else 'PASSED'
    return {'schema_version': 'llm-ab-report-v1', 'status': status, 'reason_codes': reasons,
        'test_cases': len(cases), 'independent_groups': len(groups), 'baseline': aggregates['baseline'],
        'candidate': aggregates['candidate'], 'primary_improvement': improvement,
        'paired_group_bootstrap_ci': [low, high], 'new_critical_errors': new_critical,
        'protocol_digest': content_hash(protocol), 'case_digest': content_hash(cases),
        'dataset_digest': (dataset_manifest or {}).get('dataset_digest'),
        'identities': identities or {}, 'raw_model_metrics_only': True,
        'limitations': ['Mechanical exact-match and anchor checks do not establish financial truth.',
            'Synthesis promotion additionally requires independent blind human review.',
            'Previously inspected test labels are development data, not a new independent holdout.']}
