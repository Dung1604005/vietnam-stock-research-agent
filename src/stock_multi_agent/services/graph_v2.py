"""SQLite is the provenance ledger; Neo4j is an optional idempotent projection.

No co-mention is promoted to ownership, causation or risk propagation. Proposed
economic relationships require review; only metadata links are auto-extracted.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from stock_multi_agent.models.graph_v2 import GraphEntity, GraphEvidence, GraphTriple
from stock_multi_agent.services.knowledge_graph import Neo4jKnowledgeGraph


def _id(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:32]


class InvestmentGraph:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS ig_evidence(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS ig_nodes(id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS ig_aliases(alias TEXT NOT NULL, node_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL REFERENCES ig_evidence(id), PRIMARY KEY(alias,node_id,evidence_id));
                CREATE TABLE IF NOT EXISTS ig_edges(id TEXT PRIMARY KEY, subject TEXT NOT NULL REFERENCES ig_nodes(id),
                    object TEXT NOT NULL REFERENCES ig_nodes(id), relation TEXT NOT NULL,
                    evidence_id TEXT NOT NULL REFERENCES ig_evidence(id), valid_from TEXT NOT NULL,
                    valid_to TEXT, confidence REAL NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS ig_subject ON ig_edges(subject);
                CREATE INDEX IF NOT EXISTS ig_object ON ig_edges(object);
                CREATE TABLE IF NOT EXISTS ig_projection_state(document_id TEXT PRIMARY KEY,fingerprint TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS ig_node_evidence(node_id TEXT NOT NULL REFERENCES ig_nodes(id),
                    evidence_id TEXT NOT NULL REFERENCES ig_evidence(id),valid_from TEXT NOT NULL,
                    valid_to TEXT,confidence REAL NOT NULL,PRIMARY KEY(node_id,evidence_id,valid_from));
            ''')

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def put_evidence(self, evidence: GraphEvidence):
        with self._connect() as db:
            row = db.execute('SELECT payload FROM ig_evidence WHERE id=?', (evidence.evidence_id,)).fetchone()
            if row and GraphEvidence.model_validate_json(row['payload']) != evidence:
                raise ValueError('Evidence IDs are immutable; create a new version')
            db.execute('INSERT OR IGNORE INTO ig_evidence VALUES (?,?)', (evidence.evidence_id, evidence.model_dump_json()))

    def add_alias(self, alias: str, entity_id: str, evidence_id: str):
        normalized = Neo4jKnowledgeGraph.normalize(alias)
        if not normalized:
            raise ValueError('Empty alias')
        with self._connect() as db:
            if not db.execute('SELECT 1 FROM ig_nodes WHERE id=?', (entity_id,)).fetchone():
                raise ValueError('Unknown alias entity')
            db.execute('INSERT OR IGNORE INTO ig_aliases VALUES (?,?,?)', (normalized, entity_id, evidence_id))

    def resolve(self, alias: str) -> list[str]:
        """Multiple hits remain ambiguous: never merge companies/people on a name alone."""
        with self._connect() as db:
            return [row[0] for row in db.execute('SELECT DISTINCT node_id FROM ig_aliases WHERE alias=?',
                (Neo4jKnowledgeGraph.normalize(alias),))]

    def put(self, triple: GraphTriple, *, reviewed: bool = False, trusted_metadata: bool = False):
        # Revalidate even if caller used model_copy(update=...).
        triple = GraphTriple.model_validate(triple.model_dump())
        with self._connect() as db:
            row = db.execute('SELECT payload FROM ig_evidence WHERE id=?', (triple.evidence_id,)).fetchone()
            if not row:
                raise ValueError('Missing graph provenance')
            evidence = GraphEvidence.model_validate_json(row['payload'])
            if triple.valid_from < (evidence.published_at or evidence.retrieved_at):
                raise ValueError('Graph claim cannot be known before publication')
            if triple.relation == 'EXPOSED_TO' and not reviewed:
                raise ValueError('Risk exposure requires reviewed explicit evidence; metadata cannot imply risk')
            if not trusted_metadata or triple.relation == 'EXPOSED_TO':
                if not reviewed:
                    raise ValueError('LLM economic triples require review, not automatic acceptance')
                quote = Neo4jKnowledgeGraph.normalize(evidence.quote)
                for entity in (triple.subject, triple.object):
                    label = Neo4jKnowledgeGraph.normalize(entity.label)
                    if not re.search(rf'(?<!\w){re.escape(label)}(?!\w)', quote):
                        raise ValueError('Proposed entity is absent from the evidence quote')
            for entity in (triple.subject, triple.object):
                previous = db.execute('SELECT kind,label FROM ig_nodes WHERE id=?', (entity.entity_id,)).fetchone()
                if previous and (previous['kind'], previous['label']) != (entity.kind, entity.label):
                    raise ValueError('Entity identity conflict; add a sourced alias/version instead')
                db.execute('INSERT OR IGNORE INTO ig_nodes VALUES (?,?,?)', (entity.entity_id, entity.kind, entity.label))
                db.execute('INSERT OR IGNORE INTO ig_aliases VALUES (?,?,?)',
                    (Neo4jKnowledgeGraph.normalize(entity.label), entity.entity_id, triple.evidence_id))
                db.execute('INSERT OR REPLACE INTO ig_node_evidence VALUES (?,?,?,?,?)',
                    (entity.entity_id, triple.evidence_id, triple.valid_from.isoformat(),
                     triple.valid_to.isoformat() if triple.valid_to else None, triple.confidence))
            key = _id(f'{triple.subject.entity_id}|{triple.relation}|{triple.object.entity_id}|{triple.evidence_id}|{triple.valid_from.isoformat()}')
            db.execute('INSERT OR REPLACE INTO ig_edges VALUES (?,?,?,?,?,?,?,?,?)',
                (key, triple.subject.entity_id, triple.object.entity_id, triple.relation, triple.evidence_id,
                 triple.valid_from.isoformat(), triple.valid_to.isoformat() if triple.valid_to else None,
                 triple.confidence, triple.model_dump_json()))

    def record_reviewed_claim(self, subject: GraphEntity, claim_text: str, evidence_id: str, *, reviewed: bool = False):
        """Attach an explicitly reviewed source excerpt, never inferred causation.

        The exact quote and named subject must both be in the source. This
        mechanical guard supplements, but cannot replace, human claim review.
        All three edges remain attached to the original evidence and known date.
        """
        if not reviewed:
            raise ValueError('Claim extraction requires review')
        if subject.kind not in {'Company', 'Ticker', 'Event', 'Metric'}:
            raise ValueError('Unsupported claim subject')
        with self._connect() as db:
            row = db.execute('SELECT payload FROM ig_evidence WHERE id=?', (evidence_id,)).fetchone()
        if not row:
            raise ValueError('Missing graph provenance')
        evidence = GraphEvidence.model_validate_json(row['payload'])
        excerpt = Neo4jKnowledgeGraph.normalize(claim_text)
        quote = Neo4jKnowledgeGraph.normalize(evidence.quote)
        label = Neo4jKnowledgeGraph.normalize(subject.label)
        if not excerpt or excerpt not in quote or not re.search(rf'(?<!\w){re.escape(label)}(?!\w)', quote):
            raise ValueError('Claim excerpt or subject is absent from the evidence quote')
        claim = GraphEntity(entity_id=f'claim:{_id(evidence_id + claim_text)}', kind='Claim', label=claim_text)
        source = GraphEntity(entity_id=f'evidence:{evidence_id}', kind='Evidence', label=evidence_id)
        for origin, relation, target in [(claim, 'ABOUT', subject), (claim, 'SUPPORTED_BY', source)]:
            self.put(GraphTriple(subject=origin, relation=relation, object=target,
                evidence_id=evidence_id, valid_from=evidence.published_at or evidence.retrieved_at,
                confidence=1, origin='deterministic'), trusted_metadata=True)
        return claim.entity_id

    def ingest_document(self, document, chunks):
        """Sourced metadata only. A ticker tag establishes MENTIONS, not authorship."""
        fingerprint = _id(document.model_dump_json())
        with self._connect() as db:
            previous = db.execute('SELECT fingerprint FROM ig_projection_state WHERE document_id=?', (document.document_id,)).fetchone()
            if previous and previous[0] == fingerprint:
                return
        doc = GraphEntity(entity_id=f'document:{document.document_id}', kind='Document', label=document.title)
        layer = 'theory' if hasattr(document, 'topics') else 'research'
        published = document.published_at
        known_at = published or document.retrieved_at
        for chunk in chunks:
            quote = chunk['text'] if isinstance(chunk, dict) else chunk.text
            chunk_id = chunk['chunk_id'] if isinstance(chunk, dict) else chunk.chunk_id
            ev = GraphEvidence(evidence_id=chunk_id, document_id=document.document_id,
                source_url=str(document.source_url), quote=quote, published_at=published,
                retrieved_at=document.retrieved_at, layer=layer)
            self.put_evidence(ev)
            targets = [(f'ticker:{ticker}', 'Ticker', ticker) for ticker in getattr(document, 'tickers', [])]
            targets += [(f'topic:{_id(topic)}', 'Topic', topic) for topic in getattr(document, 'topics', [])]
            period = getattr(document, 'reporting_period', None)
            if period and period.status == 'DETECTED' and period.end:
                targets.append((f'period:{period.kind}:{period.end.isoformat()}', 'Period', period.label or period.end.isoformat()))
            for entity_id, kind, label in targets:
                self.put(GraphTriple(subject=doc, relation='MENTIONS',
                    object=GraphEntity(entity_id=entity_id, kind=kind, label=label), evidence_id=chunk_id,
                    valid_from=known_at, confidence=1, origin='deterministic'), trusted_metadata=True)
            supersedes = getattr(document, 'supersedes_id', None)
            if supersedes:
                with self._connect() as db:
                    previous = db.execute('SELECT label FROM ig_nodes WHERE id=?', (f'document:{supersedes}',)).fetchone()
                self.put(GraphTriple(subject=doc, relation='SUPERSEDES',
                    object=GraphEntity(entity_id=f'document:{supersedes}', kind='Document', label=previous['label'] if previous else f'Phiên bản {supersedes}'),
                    evidence_id=chunk_id, valid_from=known_at, confidence=1, origin='deterministic'), trusted_metadata=True)
        with self._connect() as db:
            db.execute('INSERT OR REPLACE INTO ig_projection_state VALUES (?,?)', (document.document_id, fingerprint))

    def ingest_market(self, market):
        """Profile + typed observations: no missing values are filled by LLM."""
        if not market.instrument or not market.source_refs:
            return
        instrument = market.instrument
        company = None
        sources = {source.source_id: source for source in market.source_refs if source.source_url}
        profile = next((s for s in sources.values() if s.data_kind.value == 'COMPANY_PROFILE'), None)
        if profile and instrument.name:
            company = GraphEntity(entity_id=f'company:{instrument.ticker}', kind='Company', label=instrument.name)
            evidence_id = f'profile:{instrument.ticker}:{_id(profile.model_dump_json())}'
            published = profile.fetched_at
            self.put_evidence(GraphEvidence(evidence_id=evidence_id, document_id=evidence_id,
                source_url=profile.source_url, quote=instrument.model_dump_json(),
                published_at=None, retrieved_at=profile.fetched_at, layer='structured'))
            for kind, label, relation in [('Ticker', instrument.ticker, 'LISTED_AS'),
                    ('Sector', instrument.sector, 'IN_SECTOR')]:
                if label:
                    self.put(GraphTriple(subject=company, relation=relation,
                        object=GraphEntity(entity_id=f'{kind.lower()}:{label}', kind=kind, label=label),
                        evidence_id=evidence_id, valid_from=published, confidence=1, origin='deterministic'), trusted_metadata=True)
            if instrument.exchange:
                self.put(GraphTriple(subject=GraphEntity(entity_id=f'ticker:{instrument.ticker}', kind='Ticker', label=instrument.ticker),
                    relation='RELATED_TO', object=GraphEntity(entity_id=f'exchange:{instrument.exchange}', kind='Exchange', label=instrument.exchange),
                    evidence_id=evidence_id, valid_from=published, confidence=1, origin='deterministic'), trusted_metadata=True)
        for observation in [*market.financial_observations, *market.financial_statement_observations]:
            source = sources.get(observation.source_id)
            if not source or not observation.period_end or source.data_kind.value not in {
                'FINANCIAL_RATIOS', 'INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW'}:
                continue
            if (market.as_of and observation.period_end > market.as_of) or (observation.published_at and observation.published_at > source.fetched_at):
                continue
            payload = observation.model_dump(mode='json')
            evidence_id = f'fact:{_id(json.dumps(payload, sort_keys=True))}:{_id(source.model_dump_json())}'
            self.put_evidence(GraphEvidence(evidence_id=evidence_id, document_id=evidence_id,
                source_url=source.source_url, quote=json.dumps(payload, ensure_ascii=False),
                published_at=observation.published_at, retrieved_at=source.fetched_at, layer='structured'))
            doc = GraphEntity(entity_id=f'document:{evidence_id}', kind='Document', label=f'{market.ticker} · {observation.metric} · {observation.period_end}')
            metric_entity = GraphEntity(entity_id=f'metric:{market.ticker}:{_id(json.dumps(payload, sort_keys=True))}', kind='Metric',
                label=f'{observation.metric} = {observation.value} {observation.unit} ({observation.period_end})')
            period_entity = GraphEntity(entity_id=f'period:fact:{observation.period_end}', kind='Period', label=str(observation.period_end))
            for relation, entity in [
                ('REPORTS_METRIC', metric_entity),
                ('MENTIONS', GraphEntity(entity_id=f'ticker:{market.ticker}', kind='Ticker', label=market.ticker)),
                ('MENTIONS', GraphEntity(entity_id=f'period:fact:{observation.period_end}', kind='Period', label=str(observation.period_end))),
            ]:
                self.put(GraphTriple(subject=doc, relation=relation, object=entity, evidence_id=evidence_id,
                    valid_from=observation.published_at or source.fetched_at, confidence=1, origin='deterministic'), trusted_metadata=True)
            self.put(GraphTriple(subject=metric_entity, relation='HAS_PERIOD', object=period_entity,
                evidence_id=evidence_id, valid_from=observation.published_at or source.fetched_at,
                confidence=1, origin='deterministic'), trusted_metadata=True)
        for event in market.issuer_events:
            source = sources.get(event.source_id)
            if not source or source.data_kind.value != 'ISSUER_EVENTS':
                continue
            if market.as_of and (event.event_date > market.as_of or (event.published_date and event.published_date > market.as_of)):
                continue
            payload = event.model_dump_json()
            evidence_id = f'event:{_id(payload)}:{_id(source.model_dump_json())}'
            self.put_evidence(GraphEvidence(evidence_id=evidence_id, document_id=evidence_id,
                source_url=source.source_url, quote=payload, published_at=None,
                retrieved_at=source.fetched_at, layer='structured'))
            event_entity = GraphEntity(entity_id=f'event:{_id(payload)}', kind='Event', label=event.title)
            self.put(GraphTriple(subject=GraphEntity(entity_id=f'ticker:{market.ticker}', kind='Ticker', label=market.ticker),
                relation='MENTIONS', object=event_entity,
                evidence_id=evidence_id, valid_from=source.fetched_at, confidence=1, origin='deterministic'), trusted_metadata=True)
            if company:
                self.put(GraphTriple(subject=company, relation='HAS_EVENT', object=event_entity,
                    evidence_id=evidence_id, valid_from=source.fetched_at, confidence=1, origin='deterministic'), trusted_metadata=True)
            self.put(GraphTriple(subject=event_entity, relation='HAS_PERIOD',
                object=GraphEntity(entity_id=f'period:event:{event.event_date}', kind='Period', label=str(event.event_date)),
                evidence_id=evidence_id, valid_from=source.fetched_at, confidence=1, origin='deterministic'), trusted_metadata=True)

    def ingest_forecast(self, market, forecast):
        sources = [source for source in market.source_refs if source.source_url and source.data_kind.value == 'OHLCV']
        if not sources or forecast is None:
            return
        source = sources[0]
        payload = forecast.model_dump_json()
        identifier = f'forecast:{_id(payload)}:{_id(source.model_dump_json())}'
        known_at = datetime.now(timezone.utc)
        with self._connect() as db:
            existing = db.execute('SELECT payload FROM ig_evidence WHERE id=?', (identifier,)).fetchone()
            if existing:
                known_at = GraphEvidence.model_validate_json(existing[0]).retrieved_at
        self.put_evidence(GraphEvidence(evidence_id=identifier, document_id=identifier, source_url=source.source_url,
            quote=payload, published_at=None, retrieved_at=known_at, layer='structured'))
        self.put(GraphTriple(subject=GraphEntity(entity_id=f'ticker:{market.ticker}', kind='Ticker', label=market.ticker),
            relation='RELATED_TO', object=GraphEntity(entity_id=identifier, kind='Metric', label=f'Dự báo mô hình {forecast.model_version}'),
            evidence_id=identifier, valid_from=known_at, confidence=max(.01, min(1, forecast.confidence)), origin='deterministic'), trusted_metadata=True)

    def query(self, question: str, *, as_of: datetime | None = None, max_hops: int = 2, limit: int = 20):
        cutoff = as_of or datetime.now(timezone.utc)
        if cutoff.utcoffset() is None:
            raise ValueError('Graph as_of must be timezone-aware')
        normalized = Neo4jKnowledgeGraph.normalize(question)
        preferred = ('SUPPORTED_BY', 'ABOUT', 'LISTED_AS')
        if any(token in normalized for token in ('nganh', 'sector', 'industry')):
            preferred = ('IN_SECTOR', 'LISTED_AS', 'RELATED_TO')
        elif any(token in normalized for token in ('su kien', 'event', 'dai hoi')):
            preferred = ('HAS_EVENT', 'HAS_PERIOD', 'MENTIONS')
        elif any(token in normalized for token in ('rui ro', 'risk')):
            preferred = ('EXPOSED_TO', 'SUPPORTED_BY', 'ABOUT')
        elif any(token in normalized for token in ('chi tieu', 'metric', 'doanh thu', 'loi nhuan')):
            preferred = ('REPORTS_METRIC', 'HAS_PERIOD', 'MENTIONS')
        with self._connect() as db:
            roots = {row['node_id'] for row in db.execute('SELECT DISTINCT alias,node_id FROM ig_aliases')
                if re.search(rf'(?<!\w){re.escape(row["alias"])}(?!\w)', normalized)}
            frontier = deque((node, 0) for node in sorted(roots)[:6])
            seen, result = set(), {}
            while frontier and len(result) < min(50, limit):
                node, depth = frontier.popleft()
                if node in seen or depth >= max(1, min(2, max_hops)):
                    continue
                seen.add(node)
                rows = db.execute('''SELECT e.*,v.payload AS evidence FROM ig_edges e
                    JOIN ig_evidence v ON v.id=e.evidence_id WHERE subject=? OR object=?
                    ORDER BY CASE relation WHEN ? THEN 0 WHEN ? THEN 1 WHEN ? THEN 2 ELSE 3 END,id''', (node, node, *preferred))
                for row in rows:
                    triple = GraphTriple.model_validate_json(row['payload'])
                    evidence = GraphEvidence.model_validate_json(row['evidence'])
                    if triple.valid_from > cutoff or (evidence.published_at or evidence.retrieved_at) > cutoff or (triple.valid_to and triple.valid_to <= cutoff):
                        continue
                    result[row['id']] = {'triple': triple.model_dump(mode='json'), 'evidence': evidence.model_dump(mode='json')}
                    if len(result) >= min(50, limit):
                        break
                    frontier.append((row['object'] if row['subject'] == node else row['subject'], depth + 1))
            return list(result.values())

    def stats(self):
        with self._connect() as db:
            return {name: db.execute(f'SELECT count(*) FROM ig_{name}').fetchone()[0]
                for name in ('nodes', 'edges', 'evidence')}

    def expand_context(self, question, **kwargs):
        from stock_multi_agent.models.knowledge import KnowledgeGraphContext, KnowledgeGraphNode, KnowledgeGraphPath
        paths = self.query(question, **kwargs)
        nodes, output, documents = {}, [], []
        for item in paths:
            triple, evidence = item['triple'], item['evidence']
            for entity in (triple['subject'], triple['object']):
                nodes[entity['entity_id']] = KnowledgeGraphNode(node_id=entity['entity_id'],
                    node_type=entity['kind'].upper(), label=entity['label'],
                    metadata={'evidence_id': evidence['evidence_id'], 'valid_from': triple['valid_from'],
                              'valid_to': triple['valid_to'], 'confidence': triple['confidence']})
            output.append(KnowledgeGraphPath(node_ids=[triple['subject']['entity_id'], triple['object']['entity_id']],
                labels=[triple['subject']['label'], triple['object']['label']], relations=[triple['relation']],
                score=triple['confidence'], confidence=triple['confidence'], valid_from=triple['valid_from'], valid_to=triple['valid_to'],
                evidence_document_id=evidence['document_id'], evidence_chunk_id=evidence['evidence_id']))
            documents.append(evidence['document_id'])
        return KnowledgeGraphContext(query=question, matched_nodes=list(nodes.values()), paths=output,
            related_document_ids=list(dict.fromkeys(documents)),
            retrieval_note='Investment Graph V2 · SQLite provenance fallback · quan hệ theo nguồn, không suy ra nhân quả.')

    def sync_neo4j(self, backend=None):
        backend = backend or Neo4jKnowledgeGraph()
        if not backend.available:
            return {'available': False, 'backend': 'sqlite', **self.stats()}
        try:
            with self._connect() as db, backend.driver.session(database=backend.database) as session:
                session.run('CREATE CONSTRAINT investment_v2_id IF NOT EXISTS FOR (n:InvestmentEntityV2) REQUIRE n.id IS UNIQUE').consume()
                for row in db.execute('SELECT e.*,v.payload AS evidence FROM ig_edges e JOIN ig_evidence v ON v.id=e.evidence_id'):
                    t = GraphTriple.model_validate_json(row['payload'])
                    ev = GraphEvidence.model_validate_json(row['evidence'])
                    session.run('''MERGE (s:InvestmentEntityV2 {id:$subject}) SET s.kind=$subject_kind,s.label=$subject_label
                        MERGE (o:InvestmentEntityV2 {id:$object}) SET o.kind=$object_kind,o.label=$object_label
                        MERGE (s)-[r:RELATES_V2 {id:$id}]->(o)
                        SET r.kind=$kind,r.evidence_id=$evidence_id,r.source_url=$source_url,r.quote=$quote,
                            r.valid_from=$valid_from,r.valid_to=$valid_to,r.confidence=$confidence,
                            r.published_at=$published_at,r.retrieved_at=$retrieved_at,r.ontology_version=$ontology,
                            s.evidence_id=$evidence_id,s.valid_from=$valid_from,s.valid_to=$valid_to,s.confidence=$confidence,
                            o.evidence_id=$evidence_id,o.valid_from=$valid_from,o.valid_to=$valid_to,o.confidence=$confidence''',
                        subject=t.subject.entity_id, subject_kind=t.subject.kind, subject_label=t.subject.label,
                        object=t.object.entity_id, object_kind=t.object.kind, object_label=t.object.label,
                        id=row['id'], kind=t.relation, evidence_id=t.evidence_id, source_url=ev.source_url,
                        quote=ev.quote, valid_from=t.valid_from.isoformat(), valid_to=t.valid_to.isoformat() if t.valid_to else None,
                        confidence=t.confidence, published_at=ev.published_at.isoformat() if ev.published_at else None, retrieved_at=ev.retrieved_at.isoformat(),
                        ontology=t.ontology_version).consume()
            return {'available': True, 'backend': 'neo4j', **self.stats()}
        except Exception as exc:
            return {'available': False, 'backend': 'sqlite', 'error': type(exc).__name__, **self.stats()}
