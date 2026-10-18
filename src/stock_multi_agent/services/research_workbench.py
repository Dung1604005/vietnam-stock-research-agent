"""Small deterministic research tools with explicit lineage and SQLite persistence."""
from __future__ import annotations

import hashlib
import html
import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stock_multi_agent.models.chat import AssistantResponse, ResponseTable
from stock_multi_agent.models.data import PortfolioPosition


class ResearchWorkbench:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS analysis_snapshots_v2(
                    id TEXT PRIMARY KEY,ticker TEXT NOT NULL,horizon TEXT NOT NULL,captured_at TEXT NOT NULL,payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS snapshot_ticker ON analysis_snapshots_v2(ticker,horizon,captured_at);
                CREATE TABLE IF NOT EXISTS research_theses_v2(
                    id TEXT PRIMARY KEY,ticker TEXT NOT NULL,created_at TEXT NOT NULL,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS thesis_updates_v2(
                    id TEXT PRIMARY KEY,thesis_id TEXT NOT NULL,captured_at TEXT NOT NULL,payload TEXT NOT NULL);
            ''')

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def capture(self, state):
        if not state.market_data:
            raise ValueError('Cannot snapshot missing data')
        data = state.market_data
        from stock_multi_agent.services.graph_v2 import InvestmentGraph
        graph_status = 'SYNCED'
        try:
            graph = InvestmentGraph(self.db_path)
            graph.ingest_market(data)
            graph.ingest_forecast(data, state.price_forecast)
        except (OSError, ValueError, sqlite3.Error) as exc:
            graph_status = f'UNAVAILABLE:{type(exc).__name__}'
        captured = datetime.now(timezone.utc).isoformat()
        sections = {}
        for name, tool, value in [
            ('snapshot', 'DataAgent/MarketDataTool', data),
            ('technical', 'TechnicalAgent/Python', state.technical_result),
            ('fundamental', 'FundamentalAgent/Python', state.fundamental_result),
            ('risk', 'RiskAgent/Python', state.risk_result),
            ('decision', 'ManagerAgent/deterministic_policy', state.final_decision),
            ('forecast', 'PriceForecastService', state.price_forecast),
        ]:
            if value is None:
                continue
            payload = value.model_dump(mode='json')
            if name == 'snapshot':
                payload.pop('bars', None)
                payload['last_bar'] = data.bars[-1].model_dump(mode='json')
            sections[name] = {'tool': tool, 'source_ids': [source.source_id for source in data.source_refs], 'data': payload}
        sections['research'] = {'tool': 'ResearchAgent/hybrid_retrieval',
            'source_ids': [citation.chunk_id for citation in state.research_evidence],
            'data': [citation.model_dump(mode='json') for citation in state.research_evidence],
            'limitation': state.research_freshness_note}
        from stock_multi_agent.services.financial_facts import build_financial_evidence
        sections['financial_evidence'] = {'tool': 'canonical-financial-v1',
            'data': build_financial_evidence(data)}
        sections['llm_reviews'] = {'tool': 'configured_llm/claim_audit',
            'data': [item.model_dump(mode='json') for item in state.llm_reviews]}
        sections['evidence_follow_up'] = {'tool': 'ManagerAgent/allowlisted_follow_up',
            'data': state.evidence_follow_up_outcomes}
        record = {'ticker': state.ticker, 'horizon': state.investment_horizon, 'captured_at': captured,
            'as_of': data.as_of.isoformat(), 'sections': sections, 'graph_sync_status': graph_status}
        identifier = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:32]
        record['snapshot_id'] = identifier
        with self._connect() as db:
            db.execute('INSERT INTO analysis_snapshots_v2 VALUES (?,?,?,?,?)',
                (identifier, state.ticker, state.investment_horizon, captured, json.dumps(record, ensure_ascii=False)))
        for thesis in self.list_theses(state.ticker):
            self.track_thesis(state.ticker, thesis['thesis'], state.research_evidence,
                catalysts=thesis.get('catalysts'), risks=thesis.get('risks'))
        return record

    def explain_change(self, ticker: str, horizon='3 months'):
        with self._connect() as db:
            records = [json.loads(row[0]) for row in db.execute(
                'SELECT payload FROM analysis_snapshots_v2 WHERE ticker=? AND horizon=? ORDER BY captured_at DESC LIMIT 2', (ticker, horizon))]
        if len(records) < 2:
            return AssistantResponse(message='Cần ít nhất hai lần phân tích đã lưu cùng mã và cùng horizon để giải thích thay đổi.', tickers=[ticker])
        new, old = records
        rows = []
        for section, fields in {'technical': ['score', 'signal', 'policy_version'], 'fundamental': ['score', 'signal', 'policy_version'],
                'risk': ['risk_score', 'veto', 'data_uncertainty_score', 'policy_version'], 'decision': ['recommendation', 'policy_version']}.items():
            before = old['sections'].get(section, {}).get('data', {})
            after = new['sections'].get(section, {}).get('data', {})
            for key in fields:
                if before.get(key) != after.get(key):
                    rows.append({'Mục': f'{section}.{key}', 'Trước': before.get(key), 'Sau': after.get(key)})
        return AssistantResponse(message=(f'### Thay đổi của {ticker}\n\nSnapshot {old["captured_at"]} (dữ liệu {old["as_of"]}) → '
            f'{new["captured_at"]} (dữ liệu {new["as_of"]}).\n\n'
            + ('Các khác biệt dưới đây là quan sát, không tự chứng minh nguyên nhân.' if rows else 'Không có thay đổi score, policy hoặc kết luận.')),
            tickers=[ticker], tables=[ResponseTable(title='Đối chiếu snapshot', columns=['Mục', 'Trước', 'Sau'], rows=rows)],
            metadata={'snapshot_comparison': [old, new], 'tool': 'ResearchWorkbench.explain_change'})

    @staticmethod
    def report_changes(data):
        from stock_multi_agent.services.financial_facts import normalize_observation, comparability_reason
        groups = {}
        for item in data.financial_statement_observations:
            if item.period_end is None or item.period_end > data.as_of or (item.published_at and item.published_at > data.fetched_at):
                continue
            fact = normalize_observation(item)
            groups.setdefault((item.statement, item.metric, item.frequency, fact.unit, fact.currency), []).append(fact)
        rows = []
        for (statement, metric, frequency, unit, currency), items in sorted(groups.items(), key=lambda item: str(item[0])):
            # Conflicting source values in the same period must not be picked arbitrarily.
            periods = {}
            for item in items:
                periods.setdefault(item.period_end, []).append(item)
            ends = sorted(periods)
            if len(ends) < 2:
                continue
            left, right = periods[ends[-2]], periods[ends[-1]]
            if any(len({(value.normalized_value, value.reporting_basis, value.consolidation, value.assurance, value.scale)
                        for value in group}) > 1 for group in (left, right)):
                continue
            before, after = left[0], right[0]
            reason = comparability_reason(before, after)
            difference = after.normalized_value - before.normalized_value if reason is None else None
            rows.append({'Báo cáo': statement, 'Chỉ tiêu': metric, 'Tần suất': frequency, 'Đơn vị': unit,
                'Kỳ trước': str(before.period_end), 'Kỳ sau': str(after.period_end),
                'Trước': before.normalized_value if before.normalized_value is not None else before.value,
                'Sau': after.normalized_value if after.normalized_value is not None else after.value, 'Chênh lệch': difference,
                'Thay đổi %': 100 * difference / before.normalized_value if difference is not None and before.normalized_value > 0 else None,
                'Cơ sở': after.reporting_basis, 'Hợp nhất': after.consolidation,
                'Trạng thái': 'COMPARABLE' if reason is None else 'NOT_COMPARABLE',
                'Giới hạn': reason or 'Chênh lệch hai kỳ quan sát, không tự kết luận là tăng trưởng YoY hay nguyên nhân.',
                'Nguồn': [before.source_id, after.source_id]})
        return rows

    @staticmethod
    def event_impact(data):
        rows = []
        for event in sorted(data.issuer_events, key=lambda item: item.event_date, reverse=True)[:10]:
            rows.append({'Sự kiện': event.title, 'Ngày sự kiện': str(event.event_date),
                'Ngày công bố': str(event.published_date) if event.published_date else 'Chưa xác định',
                'Tác động có điều kiện': 'Chỉ tác động đến định giá nếu làm thay đổi dòng tiền kỳ vọng hoặc chi phí vốn; chưa định lượng từ tiêu đề.',
                'Mức chắc chắn': 'Sự kiện theo nguồn; tác động chưa được kiểm chứng', 'Nguồn': event.source_id})
        return rows

    @staticmethod
    def scenarios(data, shocks=(-0.05, 0, 0.05)):
        close = data.bars[-1].close
        if not all(math.isfinite(shock) and -1 < shock <= 1 for shock in shocks):
            raise ValueError('Scenario shock must be finite and between -100% and +100%')
        return [{'Giả định %': shock * 100, 'Giá theo giả định': close * (1 + shock),
            'Dữ liệu đến': str(data.as_of), 'Đơn vị': getattr(data, 'price_unit', 'UNKNOWN'),
            'Nguồn': [source.source_id for source in data.source_refs],
            'Công thức': 'close * (1 + shock); không phải dự báo'} for shock in shocks]

    @staticmethod
    def peer_percentiles(states):
        """Percentiles only on observed same-metric/unit/period/currency/sector peers."""
        from stock_multi_agent.services.financial_facts import normalize_observation
        groups = {}
        for state in states:
            data = state.market_data
            if data is None or data.instrument is None or not data.instrument.sector:
                continue
            for item in data.financial_observations:
                fact = normalize_observation(item)
                if (fact.period_end and fact.period_end <= data.as_of and fact.normalized_value is not None
                        and fact.reporting_basis != 'UNKNOWN' and fact.consolidation != 'UNKNOWN'
                        and (item.published_at is None or item.published_at <= data.fetched_at)):
                    key = (fact.metric, fact.unit, fact.period_end, fact.currency, data.instrument.sector,
                        fact.reporting_basis, fact.consolidation, fact.assurance, fact.period_start)
                    groups.setdefault(key, []).append((state.ticker, fact.normalized_value, fact.source_id))
        rows = []
        for (metric, unit, period, currency, sector, basis, consolidation, assurance, period_start), items in groups.items():
            values_by_ticker = {}
            for ticker, value, source in items:
                values_by_ticker.setdefault(ticker, []).append((value, source))
            if any(len({value for value, _ in group}) != 1 for group in values_by_ticker.values()) or len(values_by_ticker) < 2:
                continue
            unique = [(ticker, group[0][0], [source for _, source in group]) for ticker, group in values_by_ticker.items()]
            values = np.array([value for _, value, _ in unique])
            for ticker, value, sources in unique:
                percentile = 100 * (np.sum(values < value) + .5 * np.sum(values == value)) / len(values)
                rows.append({'Mã': ticker, 'Chỉ tiêu': metric, 'Giá trị': value, 'Đơn vị': unit, 'Tiền tệ': currency,
                    'Ngành': sector, 'Kỳ': str(period), 'Percentile trong tập so sánh': float(percentile),
                    'Cơ sở': basis, 'Hợp nhất': consolidation, 'Kiểm toán': assurance,
                    'Số peer': len(values), 'Nguồn': sources})
        return rows

    @staticmethod
    def portfolio(positions: list[PortfolioPosition], datasets):
        if not positions or len({p.ticker for p in positions}) != len(positions):
            raise ValueError('Portfolio must contain unique tickers and positive weights')
        total = sum(position.weight for position in positions)
        if total > 1.000001:
            raise ValueError('Portfolio weights exceed 100%; weights are never normalized silently')
        result = {'tool': 'ResearchWorkbench.portfolio', 'weights': [p.model_dump() for p in positions],
            'cash_weight': max(0, 1 - total), 'hhi': sum(p.weight ** 2 for p in positions),
            'largest_position': max(p.weight for p in positions), 'quantitative_status': 'UNAVAILABLE',
            'source_ids': [], 'limitations': ['Hiệp phương sai lịch sử không phải dự báo; giả định tỷ trọng không đổi/tái cân bằng mỗi phiên, tiền mặt lợi suất 0.']}
        if any(p.ticker not in datasets for p in positions):
            result['limitations'].append('Thiếu dữ liệu một hoặc nhiều mã; không tính volatility trên danh mục thiếu thành phần.')
            return result
        data = [datasets[p.ticker] for p in positions]
        currencies = {item.instrument.currency if item.instrument else None for item in data}
        if None in currencies or len(currencies) != 1 or len({item.as_of for item in data}) != 1:
            result['limitations'].append('Không đồng nhất currency/as_of; dừng phép tính tương quan/volatility.')
            return result
        prices = pd.concat([pd.Series({bar.date: bar.close for bar in item.bars}, name=item.ticker) for item in data], axis=1).sort_index()
        # Compute returns BEFORE alignment: never bridge missing sessions silently.
        returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) < 20:
            result['limitations'].append('Cần ít nhất 20 phiên lợi suất đồng thời hợp lệ.')
            return result
        weights = np.array([p.weight for p in positions])
        daily = returns.to_numpy() @ weights
        covariance = returns.cov().to_numpy()
        variance = max(0, float(weights @ covariance @ weights))
        result.update(quantitative_status='ASSESSED', aligned_sessions=len(returns),
            window_start=str(returns.index.min()), window_end=str(returns.index.max()),
            annualized_volatility=math.sqrt(252 * variance),
            historical_daily_std=float(np.std(daily, ddof=1)), correlation=returns.corr().replace({np.nan: None}).to_dict(),
            source_ids=list(dict.fromkeys(s.source_id for item in data for s in item.source_refs)))
        if any(item.price_adjustment.value != 'ADJUSTED' for item in data):
            result['limitations'].append('Giá điều chỉnh chưa được xác minh; corporate actions có thể làm sai volatility/correlation.')
        return result

    def track_thesis(self, ticker, thesis, evidence, *, catalysts=None, risks=None):
        captured = datetime.now(timezone.utc).isoformat()
        evidence = [item.model_dump(mode='json') if hasattr(item, 'model_dump') else item for item in evidence]
        record = {'ticker': ticker, 'thesis': thesis, 'created_at': captured,
            'catalysts': catalysts or [], 'risks': risks or [], 'evidence': evidence,
            'status': 'NEEDS_REVIEW' if evidence else 'INSUFFICIENT_EVIDENCE',
            'limitation': 'Có tài liệu liên quan không đồng nghĩa luận điểm được xác nhận; cần review từng claim.'}
        identifier = hashlib.sha256(f'{ticker}|{thesis}'.encode()).hexdigest()[:32]
        with self._connect() as db:
            previous_row = db.execute('SELECT payload FROM research_theses_v2 WHERE id=?', (identifier,)).fetchone()
            previous = json.loads(previous_row[0]) if previous_row else {}
            old_ids = {item.get('chunk_id', item.get('document_id', '')) for item in previous.get('evidence', [])}
            new_ids = {item.get('chunk_id', item.get('document_id', '')) for item in evidence}
            record['new_evidence_ids'] = sorted(new_ids - old_ids - {''})
            record['evidence_no_longer_current'] = sorted(old_ids - new_ids - {''})
            version = hashlib.sha256(json.dumps({key: value for key, value in record.items() if key != 'created_at'}, sort_keys=True).encode()).hexdigest()
            db.execute('INSERT OR IGNORE INTO thesis_updates_v2 VALUES (?,?,?,?)',
                (version, identifier, captured, json.dumps(record, ensure_ascii=False)))
            db.execute('INSERT OR REPLACE INTO research_theses_v2 VALUES (?,?,?,?)',
                (identifier, ticker, captured, json.dumps(record, ensure_ascii=False)))
        record['thesis_id'] = identifier
        return record

    def list_theses(self, ticker):
        with self._connect() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT payload FROM research_theses_v2 WHERE ticker=? ORDER BY created_at DESC LIMIT 20', (ticker,))]

    @staticmethod
    def export(response, format='markdown'):
        """All report formats share the same public-safe audit sanitizer."""
        from stock_multi_agent.services.audit_export import export_audit_json, export_html, export_markdown
        exporters = {'markdown': export_markdown, 'html': export_html, 'json': export_audit_json}
        if format not in exporters:
            raise ValueError('Only markdown/html/json reports are supported')
        return exporters[format](response)
