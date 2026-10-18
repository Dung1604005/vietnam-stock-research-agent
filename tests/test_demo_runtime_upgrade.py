"""Synthetic engineering cases, not financial-model quality evidence."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import json
import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from stock_multi_agent.models.chat import AssistantResponse
from stock_multi_agent.models.execution_profile import ExecutionMode, ExecutionProfile
from stock_multi_agent.models.research_contracts import RequestBudget
from stock_multi_agent.providers.market_provider import MarketDataError
from stock_multi_agent.providers.offline_provider import OfflineMarketDataProvider
from stock_multi_agent.services.audit_export import build_audit, export_audit_json, export_html, export_markdown
from stock_multi_agent.services.demo_bundle import DemoBundleService
from stock_multi_agent.services.execution_profiles import (
    ExternalAccessDenied, assert_external_allowed, constrain_budget,
    current_profile, execution_scope, resolve_profile,
)
from stock_multi_agent.services.progress_events import ProgressSink, progress_scope
from stock_multi_agent.services.request_telemetry import observed, request_scope
from stock_multi_agent.services.runtime_health import runtime_health


def test_profiles_are_immutable_scoped_and_standard_preserves_budget():
    budget = RequestBudget(max_llm_calls=6, deadline_seconds=240)
    assert constrain_budget(budget).max_llm_calls == 6
    with execution_scope('QUICK'):
        assert current_profile().mode == ExecutionMode.QUICK
        assert constrain_budget(budget).max_llm_calls == 0
        assert constrain_budget(budget).deadline_seconds == 30
        with pytest.raises(ValidationError):
            current_profile().external_allowed = False
        with execution_scope('OFFLINE'):
            with pytest.raises(ExternalAccessDenied):
                assert_external_allowed('web search')
        assert current_profile().mode == ExecutionMode.QUICK
    assert current_profile().mode == ExecutionMode.STANDARD


def test_profile_concurrency_does_not_leak():
    def check(mode):
        with execution_scope(mode):
            return current_profile().mode
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(check, ['QUICK', 'OFFLINE']))
    assert results == [ExecutionMode.QUICK, ExecutionMode.OFFLINE]
    assert current_profile().mode == ExecutionMode.STANDARD


def test_observed_progress_is_real_ordered_and_failure_not_success():
    @observed('test.actual')
    def actual(fail=False):
        if fail:
            raise ValueError('not exposed to UI')
        return 42
    sink = ProgressSink()
    assert sink.snapshot() == []
    with progress_scope(sink), request_scope() as metrics:
        assert actual() == 42
        with pytest.raises(ValueError):
            actual(True)
    events = sink.snapshot()
    assert [item['status'] for item in events] == ['STARTED', 'SUCCEEDED', 'STARTED', 'FAILED']
    assert [item['attempt'] for item in events] == [1, 1, 2, 2]
    assert len({item['event_id'] for item in events}) == 4
    assert {item['request_id'] for item in events} == {metrics.request_id}
    assert sink.drain() == events
    assert sink.drain() == []
    assert metrics.failures == 1


def test_progress_threads_use_explicit_context_and_sink_is_thread_safe():
    @observed('parallel')
    def task():
        return True
    with progress_scope() as sink, ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(copy_context().run, task) for _ in range(20)]
        assert all(item.result() for item in futures)
    events = sink.snapshot()
    assert len(events) == 40
    assert len({item['sequence'] for item in events}) == 40
    assert len({item['event_id'] for item in events}) == 40


def test_observed_budget_failure_emits_failed_event():
    @observed('llm', 'llm')
    def task():
        pytest.fail('must not run')
    with progress_scope() as sink, request_scope(RequestBudget(max_llm_calls=0)):
        with pytest.raises(RuntimeError):
            task()
    assert [event['status'] for event in sink.snapshot()] == ['STARTED', 'FAILED']


def test_partial_agent_report_emits_partial_not_completed():
    @observed('agent.partial')
    def task():
        return {'agent_reports': [{'status': 'PARTIAL'}]}
    with progress_scope() as sink:
        task()
    assert [event['status'] for event in sink.snapshot()] == ['STARTED', 'PARTIAL']


def test_audit_sanitizes_credentials_links_paths_and_does_not_dump_metadata(market_data):
    source = market_data.source_refs[0].model_copy(update={
        'request_parameters': {'Authorization': 'SENTINEL_HEADER'},
        'source_url': 'https://name:SENTINEL_PASSWORD@example.com/report.pdf?api_key=SENTINEL_QUERY#secret',
    })
    response = AssistantResponse(message='<script>alert(1)</script> api_key=SENTINEL_INLINE user@example.com',
        source_details=[source], metadata={
            'raw_query': 'SENTINEL_PRIVATE_CHAT',
            'api_key': 'SENTINEL_KEY',
            'request_metrics': {'llm_calls': 1, 'request_parameters': {'token': 'SENTINEL_NESTED'}},
            'llm_reviews': [{'assessment_vi': 'Evidence review', 'raw_output': 'SENTINEL_RAW',
                'source_url': 'https://example.com/x?token=SENTINEL_TOKEN', 'local_path': 'SENTINEL_PATH'}],
        })
    exported = export_audit_json(response)
    assert 'SENTINEL' not in exported
    assert 'user@example.com' not in exported
    assert 'https://example.com/report.pdf' in exported
    assert '<script>' not in export_html(response)
    assert '<script>' not in export_markdown(response)
    assert build_audit(response)['schema_version'] == 'research-audit-v1'


def test_audit_keeps_version_identity_but_not_prompt_or_private_portfolio():
    response = AssistantResponse(message='Safe', metadata={
        'model_identity': {'model': 'Qwen/example', 'prompt_version': 'v1',
            'prompt_sha256': 'a' * 64, 'tokenizer_revision': 'pinned', 'prompt': 'SENTINEL_PROMPT'},
    })
    audit = build_audit(response)
    assert audit['audit']['model_identity']['prompt_version'] == 'v1'
    assert audit['audit']['model_identity']['tokenizer_revision'] == 'pinned'
    assert 'SENTINEL' not in json.dumps(audit)
    response.metadata['portfolio_risk'] = {'weights': [{'ticker': 'FPT', 'weight': .5}]}
    audit = build_audit(response)
    assert 'portfolio_risk' not in audit['audit']
    assert 'portfolio_risk' in build_audit(response, include_private_portfolio=True)['audit']


def test_audit_does_not_leak_user_query_or_private_portfolio_via_message(market_data):
    response = AssistantResponse(message='FPT PRIVATE_HOLDING weight 50%', tickers=['FPT'],
        source_details=market_data.source_refs,
        metadata={'evidence_bundle': {'query': 'SENTINEL_PRIVATE_QUERY', 'citations': []}})
    assert 'SENTINEL' not in export_audit_json(response)
    response.metadata['portfolio_risk'] = {'weights': [{'ticker': 'FPT', 'weight': .5}]}
    exported = export_audit_json(response)
    assert 'PRIVATE_HOLDING' not in exported and 'FPT' not in exported
    assert 'PRIVATE_HOLDING' in export_audit_json(response, include_private_portfolio=True)


def test_bundle_replays_every_bar_timestamp_provenance_and_missing_is_error(market_data, isolated_runtime_root):
    destination = isolated_runtime_root / 'demo' / f'{uuid4().hex}.json'
    DemoBundleService.capture([market_data], destination, rights_note='Synthetic fixture for engineering tests only.', allow_test_fixture=True)
    bundle = DemoBundleService.load(destination)
    provider = OfflineMarketDataProvider(bundle)
    with execution_scope('OFFLINE'):
        replay = provider.get_market_data('fpt')
        with pytest.raises(MarketDataError, match='no network fallback'):
            provider.get_market_data('TCB')
    assert replay.bars == market_data.bars
    assert replay.fetched_at == market_data.fetched_at
    assert replay.as_of == market_data.as_of
    assert replay.source == market_data.source
    assert replay.source_refs[0].source_id == market_data.source_refs[0].source_id
    assert 'SNAPSHOT_REPLAY' in replay.warnings[-1]
    replay.bars.clear()
    assert len(provider.get_market_data('FPT').bars) == len(market_data.bars)


def test_bundle_rejects_corruption_and_never_overwrites(market_data, isolated_runtime_root):
    destination = isolated_runtime_root / 'demo' / f'{uuid4().hex}.json'
    kwargs = {'rights_note': 'Synthetic test only, not real prices.', 'allow_test_fixture': True}
    DemoBundleService.capture([market_data], destination, **kwargs)
    with pytest.raises(FileExistsError):
        DemoBundleService.capture([market_data], destination, **kwargs)
    raw = json.loads(destination.read_text(encoding='utf-8'))
    raw['payload']['datasets']['FPT']['bars'][-1]['close'] += 1
    destination.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        DemoBundleService.load(destination)


def test_bundle_rejects_fixture_mislabeled_as_real_and_no_source(market_data, isolated_runtime_root):
    destination = isolated_runtime_root / 'demo' / f'{uuid4().hex}.json'
    with pytest.raises(ValueError, match='Synthetic'):
        DemoBundleService.capture([market_data], destination, rights_note='Private personal local research only.')
    missing = market_data.model_copy(update={'source_refs': []})
    with pytest.raises(ValueError, match='source provenance'):
        DemoBundleService.capture([missing], destination, rights_note='Engineering fixture test only.', allow_test_fixture=True)


def test_health_twenty_reruns_do_not_touch_network(monkeypatch, isolated_runtime_root):
    import socket
    def fail(*args, **kwargs):
        pytest.fail('Health must not attempt outbound access')
    monkeypatch.setattr(socket, 'create_connection', fail)
    for _ in range(20):
        result = runtime_health(isolated_runtime_root)
        assert result['external_calls'] == 0
        assert next(item for item in result['checks'] if item['component'] == 'market_provider')['status'] == 'UNKNOWN'


def test_workbench_export_uses_shared_sanitizer():
    from stock_multi_agent.services.research_workbench import ResearchWorkbench
    response = AssistantResponse(message='Public research', metadata={'authorization': 'SENTINEL_PRIVATE'})
    assert 'SENTINEL' not in ResearchWorkbench.export(response, 'json')
    assert json.loads(ResearchWorkbench.export(response, 'json'))['schema_version'] == 'research-audit-v1'
    with pytest.raises(ValueError):
        ResearchWorkbench.export(response, 'binary')


def test_invalid_offline_profile_cannot_enable_network_or_llm():
    with pytest.raises(ValidationError, match='OFFLINE'):
        ExecutionProfile(mode='OFFLINE', external_allowed=True, max_llm_calls=0)
    with pytest.raises(ValidationError, match='OFFLINE'):
        ExecutionProfile(mode='OFFLINE', external_allowed=False, max_llm_calls=1)
    # model_copy is not a validator; scope still rejects bypassed invalid policies.
    invalid = resolve_profile('OFFLINE').model_copy(update={'external_allowed': True})
    with pytest.raises(ValueError, match='OFFLINE'):
        with execution_scope(invalid):
            pytest.fail('Invalid policy must not enter scope')


def test_bundle_keeps_unit_metadata_and_rejects_ambiguous_tickers(market_data, isolated_runtime_root):
    source = market_data.model_copy(update={'price_unit': 'THOUSAND_VND'})
    target = isolated_runtime_root / 'demo' / f'{uuid4().hex}.json'
    kwargs = dict(rights_note='Synthetic fixture for tests only.', allow_test_fixture=True)
    with pytest.raises(ValueError, match='Duplicate replay ticker'):
        DemoBundleService.capture([source, source], target, **kwargs)
    DemoBundleService.capture([source], target, **kwargs)
    assert DemoBundleService.load(target).datasets['FPT'].price_unit == 'THOUSAND_VND'
    assert market_data.price_unit == 'UNKNOWN'


def test_bundle_rejects_missing_price_provenance_asof(market_data, isolated_runtime_root):
    missing = market_data.model_copy(deep=True)
    missing.source_refs[0].as_of = None
    with pytest.raises(ValidationError, match='OHLCV provenance'):
        DemoBundleService.capture([missing], isolated_runtime_root / 'bad-source.json',
            rights_note='Synthetic fixture for tests only.', allow_test_fixture=True)


def test_bundle_rejects_duplicate_json_keys(isolated_runtime_root):
    target = isolated_runtime_root / f'duplicate-{uuid4().hex}.json'
    target.write_text('{"payload":{},"payload":{},"bundle_sha256":"invalid"}', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        DemoBundleService.load(target)


def test_health_counts_actual_research_tables_and_detects_wrong_schema(isolated_runtime_root):
    root = isolated_runtime_root / f'health-{uuid4().hex}'
    database = root / 'research' / 'research.sqlite3'
    database.parent.mkdir(parents=True)
    db = sqlite3.connect(database)
    try:
        db.executescript('CREATE TABLE documents (id INTEGER); CREATE TABLE chunks (id INTEGER);')
        db.commit()
        report = runtime_health(root)
        check = next(item for item in report['checks'] if item['component'] == 'research_store')
        assert check['status'] == 'SCHEMA_MISMATCH'
        db.executescript('CREATE TABLE research_documents (id INTEGER); CREATE TABLE research_chunks (id INTEGER); INSERT INTO research_documents VALUES (1); INSERT INTO research_chunks VALUES (1),(2);')
        db.commit()
        report = runtime_health(root)
        check = next(item for item in report['checks'] if item['component'] == 'research_store')
        assert check['status'] == 'READY'
        assert check['counts'] == {'research_documents': 1, 'research_chunks': 2}
    finally:
        db.close()


def test_health_manifest_present_is_not_model_validation(isolated_runtime_root):
    artifact = isolated_runtime_root / f'artifact-{uuid4().hex}'
    artifact.mkdir()
    manifest = artifact / 'manifest.json'
    manifest.write_text(json.dumps({'files': {'missing.keras': 'a' * 64}}), encoding='utf-8')
    report = runtime_health(isolated_runtime_root, artifact_root=artifact)
    check = next(item for item in report['checks'] if item['component'] == 'lstm_v2')
    assert check['status'] == 'INCOMPLETE_ARTIFACT'
    assert not check['model_hashes_verified']
    (artifact / 'missing.keras').write_bytes(b'not a TensorFlow artifact')
    report = runtime_health(isolated_runtime_root, artifact_root=artifact)
    check = next(item for item in report['checks'] if item['component'] == 'lstm_v2')
    assert check['status'] == 'PRESENT_UNLOADED'
    assert not check['model_hashes_verified']


def test_health_optional_gateway_observation_failure_does_not_crash(isolated_runtime_root):
    class BrokenGateway:
        def health(self):
            raise OSError('SENTINEL_PRIVATE_ERROR')
    report = runtime_health(isolated_runtime_root, gateway=BrokenGateway())
    assert next(item for item in report['checks'] if item['component'] == 'llm')['status'] == 'UNKNOWN'
    assert 'SENTINEL' not in json.dumps(report)
