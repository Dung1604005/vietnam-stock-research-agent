"""Backfill the V2 provenance ledger; optionally project it into configured Neo4j."""
from __future__ import annotations

import argparse
import json

from stock_multi_agent.services.graph_v2 import InvestmentGraph
from stock_multi_agent.services.knowledge_service import KnowledgeService
from stock_multi_agent.services.research_store import ResearchStore
from stock_multi_agent.config import load_workspace_env
from stock_multi_agent.services.embeddings import LocalHashEmbeddingProvider
from stock_multi_agent.services.knowledge_store import KnowledgeStore
from stock_multi_agent.services.knowledge_graph import Neo4jKnowledgeGraph


def main():
    load_workspace_env()
    parser = argparse.ArgumentParser()
    parser.add_argument('--neo4j', action='store_true')
    parser.add_argument('--query')
    args = parser.parse_args()
    research = ResearchStore(embedding_provider=LocalHashEmbeddingProvider())
    graph = InvestmentGraph(research.database_path)
    with research._connect() as db:
        for row in db.execute('SELECT * FROM research_documents ORDER BY version,document_id'):
            chunks = [dict(chunk) for chunk in db.execute('SELECT * FROM research_chunks WHERE document_id=? ORDER BY ordinal', (row['document_id'],))]
            graph.ingest_document(research._document(row), chunks)
    knowledge = KnowledgeService(KnowledgeStore(knowledge_graph=Neo4jKnowledgeGraph(enabled=False)), bootstrap=False)
    knowledge.ensure_seeded()
    for record in knowledge.store.graph_records():
        graph.ingest_document(knowledge.store.get_document(record['document_id']), record['chunks'])
    print(json.dumps(graph.sync_neo4j() if args.neo4j else graph.stats(), ensure_ascii=False, indent=2))
    if args.query:
        print(json.dumps(graph.query(args.query), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
