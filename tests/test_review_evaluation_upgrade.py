import copy
import json
import sqlite3

import pytest

from stock_multi_agent.services.review_evaluation import export_review_pack, summarize_review_pack, RUBRIC


def test_real_review_pack_is_pending_and_does_not_leak_request_parameters(tmp_path):
    path = tmp_path / 'research.sqlite3'
    record = {'snapshot_id': 'example', 'ticker': 'FPT', 'horizon': '3 months', 'captured_at': '2026-10-17',
        'as_of': '2026-10-16', 'sections': {'snapshot': {'data': {'source': 'actual-provider',
            'source_refs': [{'source_id': 'price', 'request_parameters': {'api_key': 'DO_NOT_EXPORT'}}]}}}}
    fixture = copy.deepcopy(record)
    fixture['sections']['snapshot']['data']['source'] = 'test-fixture'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE analysis_snapshots_v2(captured_at TEXT, payload TEXT)')
        for row in (record, fixture): db.execute('INSERT INTO analysis_snapshots_v2 VALUES (?,?)', (row['captured_at'], json.dumps(row)))
    pack = export_review_pack(path)
    assert len(pack['cases']) == 1 and 'DO_NOT_EXPORT' not in json.dumps(pack)
    result = summarize_review_pack(pack)
    assert result['status'] == 'PENDING_HUMAN_REVIEW' and result['rubric_pass_rates']['claims_supported'] is None
    case = pack['cases'][0]
    case.update(reviewer='Reviewer', reviewed_at='2026-10-17T10:00:00+07:00', review_status='REVIEWED', judgments={key: True for key in RUBRIC})
    assert summarize_review_pack(pack)['reviewed_cases'] == 1
    case['frozen']['ticker'] = 'HPG'
    with pytest.raises(ValueError, match='changed'): summarize_review_pack(pack)
