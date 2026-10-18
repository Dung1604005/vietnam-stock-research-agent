"""Human review packs for frozen real snapshots; never invent accuracy labels."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone


RUBRIC = ('issuer_correct', 'period_correct', 'unit_correct', 'claims_supported',
          'risk_limitations_clear', 'decision_explained')


def export_review_pack(database_path, *, limit=20):
    path = database_path.resolve()
    # URI read-only: do not create or mutate an application database for evaluation.
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='analysis_snapshots_v2'").fetchone()
        rows = db.execute('SELECT payload FROM analysis_snapshots_v2 ORDER BY captured_at DESC LIMIT ?',
                          (max(1, min(limit, 100)),)).fetchall() if exists else []
    cases = []
    for (payload,) in rows:
        record = json.loads(payload)
        sections = record.get('sections', {})
        data = sections.get('snapshot', {}).get('data', {})
        if str(data.get('source', '')).casefold() in {'test-fixture', 'mock', 'synthetic'}:
            continue
        # Exclude raw request_parameters and credentials from shareable evaluation packs.
        provenance = [{key: source.get(key) for key in
            ('source_id', 'provider', 'data_kind', 'source_url', 'as_of', 'fetched_at', 'content_sha256')}
            for source in data.get('source_refs', [])]
        frozen = {'snapshot_id': record['snapshot_id'], 'ticker': record['ticker'],
            'horizon': record['horizon'], 'captured_at': record['captured_at'], 'as_of': record['as_of'],
            'source_provenance': provenance,
            'analysis': {key: sections.get(key, {}).get('data') for key in
                ('technical', 'fundamental', 'risk', 'decision', 'forecast', 'financial_evidence', 'research', 'llm_reviews', 'evidence_follow_up')}}
        digest = hashlib.sha256(json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cases.append({'case_id': record['snapshot_id'], 'frozen_sha256': digest, 'frozen': frozen,
            'reviewer': None, 'reviewed_at': None, 'review_status': 'PENDING',
            'judgments': {key: None for key in RUBRIC}, 'notes': ''})
    return {'schema_version': 'real-snapshot-review-v1', 'generated_at': datetime.now(timezone.utc).isoformat(),
        'provenance': 'Saved application snapshots, not a synthetic correctness benchmark',
        'independence': 'Reviewer independence is not verified by this tool. Do not tune on this pack and call it holdout.',
        'cases': cases}


def summarize_review_pack(pack):
    valid, pending = [], 0
    for case in pack.get('cases', []):
        digest = hashlib.sha256(json.dumps(case['frozen'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if digest != case['frozen_sha256']:
            raise ValueError('Frozen snapshot changed after export')
        reviewed = (case.get('review_status') == 'REVIEWED' and bool(case.get('reviewer'))
            and bool(case.get('reviewed_at')) and all(type(case.get('judgments', {}).get(key)) is bool for key in RUBRIC))
        if reviewed:
            valid.append(case)
        else:
            pending += 1
    return {'reviewed_cases': len(valid), 'pending_cases': pending,
        'status': 'REVIEWED_SAMPLE_ONLY' if valid else 'PENDING_HUMAN_REVIEW',
        'rubric_pass_rates': {key: sum(case['judgments'][key] for case in valid) / len(valid) if valid else None for key in RUBRIC},
        'investment_accuracy_or_profitability': 'NOT_MEASURED', 'reviewer_independence': 'NOT_VERIFIED'}
