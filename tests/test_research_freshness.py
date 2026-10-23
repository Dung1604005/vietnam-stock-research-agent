from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4
from types import SimpleNamespace

import pytest

from stock_multi_agent.models.research import ResearchDocumentType, ExternalEvidenceCitation
from stock_multi_agent.models.chat import ParsedQuery, QueryIntent
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.reporting_period import infer_reporting_period, period_display
from stock_multi_agent.services.research_freshness import ResearchScope, research_scope, freshness_reason, filter_citations, external_to_citation
from stock_multi_agent.services.research_store import ResearchStore
from stock_multi_agent.services.rag_service import RAGService
from stock_multi_agent.services.chat_service import ChatService
from stock_multi_agent.services.research_planner import ResearchPlanner
from stock_multi_agent.agents.research_agent import ResearchAgent
from stock_multi_agent.services.research_crawl_store import CandidateRepository
from stock_multi_agent.models.research_crawler import DocumentCandidate, CandidateStatus


NOW = date(2026, 10, 17)


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr("stock_multi_agent.services.research_freshness.today", lambda: NOW)
    path = Path(__file__).resolve().parents[1] / ".runtime" / "test-artifacts" / "freshness" / uuid4().hex / "research.sqlite3"
    return ResearchStore(path)


def ingest(store, title="FPT báo cáo tài chính quý 3 năm 2026", *, published="2026-10-10", kind=ResearchDocumentType.FINANCIAL_STATEMENT, url=None, text=None):
    return store.ingest_text(text=text or title + ". FPT doanh thu lợi nhuận công nghệ được báo cáo tăng trưởng.",
        title=title, document_type=kind, source_url=url or "https://example.invalid/" + uuid4().hex,
        publisher="Authored fixture", published_at=datetime.fromisoformat(published).replace(tzinfo=timezone.utc), tickers=["FPT"])


@pytest.mark.parametrize("title,end,kind,quarter", [
    ("BCTC FPT quý 2 năm 2026", "2026-06-30", "quarter", 2),
    ("FPT consolidated financial statements QIII.2025", "2025-09-30", "quarter", 3),
    ("FPT Q4/2025 financial statements", "2025-12-31", "quarter", 4),
    ("Báo cáo thường niên năm 2025", "2025-12-31", "annual", None),
    ("Báo cáo tài chính 6 tháng đầu năm 2026", "2026-06-30", "half_year", None),
    ("BCTC bán niên 2026", "2026-06-30", "half_year", None),
    ("Kết quả 9 tháng 2025", "2025-09-30", "nine_months", None),
    ("Báo cáo thường niên 2025 Cập nhật: 08/04/2026", "2025-12-31", "annual", None),
])
def test_heading_period_is_independent_of_publication_date(title, end, kind, quarter):
    period = infer_reporting_period(title, document_type=ResearchDocumentType.ANNUAL_REPORT if "thường niên" in title else None)
    assert period.status == "DETECTED"
    assert period.end == date.fromisoformat(end) and period.kind == kind and period.quarter == quarter


@pytest.mark.parametrize("text", ["For the year ended 31 December 2025", "Cho năm kết thúc ngày 31 tháng 12 năm 2025", "Kỳ kết thúc: 31/12/2025"])
def test_explicit_period_end_in_document_heading(text):
    assert infer_reporting_period("Financial statements", text).end == date(2025, 12, 31)


def test_unknown_and_conflicting_periods_are_not_guessed():
    assert infer_reporting_period("Báo cáo tài chính", "Ngày công bố: 17/10/2026").status == "UNKNOWN"
    assert infer_reporting_period("BCTC quý 2 năm 2026", "Kỳ kết thúc: 31/03/2026").status == "AMBIGUOUS"
    assert infer_reporting_period("BCTC niên độ 2024-2025").status == "AMBIGUOUS"
    assert infer_reporting_period("PDF report", "Công ty được thành lập năm 2006. Ngày công bố: 17/10/2026").status == "UNKNOWN"
    assert infer_reporting_period("BCTC quý 2 năm 2026", "Kỳ kết thúc: 31/03/2026", declared_end=date(2026, 6, 30)).status == "AMBIGUOUS"


def test_old_report_reposted_today_never_becomes_current(store):
    doc = ingest(store, "FPT báo cáo tài chính năm 2022", published="2026-10-17")
    assert doc.period_end == date(2022, 12, 31)
    current = store.search("FPT doanh thu lợi nhuận", ticker="FPT", scope=ResearchScope("CURRENT"))
    assert not current.citations
    assert current.historical_citations[0].document_id == doc.document_id
    assert "2022" in current.freshness_note and "Chưa có dữ liệu cập nhật" in current.freshness_note
    assert store.search("FPT doanh thu", ticker="FPT").citations


def test_newest_period_is_selected_before_retrieval_limit(store):
    for quarter in (1, 2):
        for index in range(12):
            ingest(store, f"FPT báo cáo tài chính quý {quarter} năm 2026", text=f"FPT doanh thu lợi nhuận công nghệ {quarter} chỉ tiêu {index}.")
    latest = ingest(store)
    result = store.search("FPT doanh thu lợi nhuận", ticker="FPT", limit=1, scope=ResearchScope("CURRENT"))
    assert [item.document_id for item in result.citations] == [latest.document_id]


def test_latest_period_cannot_be_replaced_by_recent_repost_of_older_year(store):
    ingest(store, "FPT báo cáo tài chính năm 2025", published="2026-10-17")
    latest = ingest(store, published="2026-10-10")
    current = store.search("FPT doanh thu", ticker="FPT", scope=ResearchScope("CURRENT"))
    assert current.citations and all(item.document_id == latest.document_id for item in current.citations)


def test_unknown_financial_period_is_withheld_but_retained(store):
    ingest(store, "FPT báo cáo tài chính", published="2026-10-17")
    current = store.search("FPT doanh thu", ticker="FPT", scope=ResearchScope("CURRENT"))
    assert not current.citations and current.historical_citations
    assert "chưa xác định" in current.freshness_note


def test_untyped_financial_numbers_cannot_bypass_period_checks(store):
    ingest(store, "FPT cập nhật doanh nghiệp", published="2026-10-17", kind=ResearchDocumentType.OTHER,
           text="FPT doanh thu năm 2022 đạt 100 tỷ đồng. Lợi nhuận công nghệ tăng trưởng.")
    result = RAGService(store, enabled=False).answer("Doanh thu FPT hiện tại?", ticker="FPT")
    assert not result.bundle.citations
    assert result.status == "NO_EVIDENCE"


def test_type_specific_age_limits_and_old_period_rule(store):
    annual = ingest(store, "FPT báo cáo thường niên 2025", published="2026-04-01", kind=ResearchDocumentType.ANNUAL_REPORT)
    news = ingest(store, "FPT doanh thu công nghệ", published="2026-08-01", kind=ResearchDocumentType.NEWS)
    citations = store.search("FPT doanh thu", ticker="FPT").citations
    selected, excluded = filter_citations(citations, ResearchScope("CURRENT"), NOW)
    assert annual.document_id in {item.document_id for item in selected}
    assert news.document_id in {item.document_id for item in excluded}


def test_explicit_year_and_quarter_queries_retrieve_history(store):
    old = ingest(store, "FPT báo cáo tài chính quý 2 năm 2022", published="2022-07-20")
    ingest(store)
    scope = research_scope("Tìm báo cáo FPT quý 2 năm 2022", ticker="FPT")
    assert scope.mode == "HISTORY" and scope.years == (2022,) and scope.quarter == 2
    result = store.search("FPT doanh thu", ticker="FPT", scope=scope)
    assert [item.document_id for item in result.citations] == [old.document_id]
    assert "Tra cứu lịch sử" in result.freshness_note


def test_compare_quarters_preserves_requested_historical_periods(store):
    one = ingest(store, "FPT báo cáo tài chính quý 1 năm 2022", published="2022-04-20")
    two = ingest(store, "FPT báo cáo tài chính quý 2 năm 2022", published="2022-07-20")
    ingest(store, "FPT báo cáo tài chính quý 3 năm 2022", published="2022-10-20")
    scope = research_scope("So sánh quý 1 và quý 2 năm 2022 của FPT", ticker="FPT")
    result = store.search("FPT doanh thu", ticker="FPT", scope=scope)
    assert {item.document_id for item in result.citations} == {one.document_id, two.document_id}


def test_latest_half_year_query_does_not_return_quarter_three(store):
    half = ingest(store, "FPT báo cáo tài chính bán niên 2026", published="2026-08-20")
    ingest(store)
    scope = research_scope("Báo cáo bán niên FPT mới nhất", ticker="FPT")
    result = store.search("FPT doanh thu", ticker="FPT", scope=scope)
    assert [item.document_id for item in result.citations] == [half.document_id]


def test_compare_quarters_across_years_does_not_cross_product_periods(store):
    requested = []
    for year in (2024, 2025):
        for quarter in (1, 4):
            item = ingest(store, f"FPT báo cáo tài chính quý {quarter} năm {year}", published=f"{year + 1}-02-01")
            if (year, quarter) in {(2024, 4), (2025, 1)}:
                requested.append(item.document_id)
    scope = research_scope("So sánh FPT quý 4/2024 và quý 1/2025", ticker="FPT")
    result = store.search("FPT doanh thu", ticker="FPT", scope=scope)
    assert {item.document_id for item in result.citations} == set(requested)


def test_reopened_store_backfills_period_without_reembedding(store):
    item = ingest(store, "FPT báo cáo tài chính quý 2 năm 2022", published="2022-07-20")
    with store._connect() as db:
        embedding_count = db.execute("SELECT count(*) FROM research_chunk_embeddings").fetchone()[0]
        db.execute("UPDATE research_documents SET reporting_period_json='{}',period_end=NULL WHERE document_id=?", (item.document_id,))
    reopened = ResearchStore(store.database_path)
    citation = reopened.search("FPT doanh thu", ticker="FPT").citations[0]
    assert citation.reporting_period.end == date(2022, 6, 30)
    with reopened._connect() as db:
        assert db.execute("SELECT count(*) FROM research_chunk_embeddings").fetchone()[0] == embedding_count


def test_reingesting_same_content_can_enrich_unknown_period_without_duplicate(store):
    text = "FPT doanh thu lợi nhuận công nghệ tăng trưởng."
    first = ingest(store, "FPT báo cáo tài chính", text=text, url="https://example.invalid/same-report")
    second = ingest(store, "FPT báo cáo tài chính quý 3 năm 2026", text=text, url="https://example.invalid/same-report")
    assert first.document_id == second.document_id
    assert second.reporting_period.end == date(2026, 9, 30)
    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM research_documents").fetchone()[0] == 1


def test_new_publication_on_same_listing_url_is_requeued_without_changing_stored_history(store):
    repo = CandidateRepository(store)
    old = repo.upsert(DocumentCandidate(source_id="issuer", source_url="https://example.invalid/report.pdf",
        title="FPT báo cáo tài chính năm 2022", published_at=datetime(2023, 3, 20, tzinfo=timezone.utc),
        date_evidence="listing date", status=CandidateStatus.INDEXED, etag="old-version"))
    fresh = repo.upsert(DocumentCandidate(source_id="issuer", source_url=old.source_url,
        title="FPT báo cáo tài chính quý 3 năm 2026", published_at=datetime(2026, 10, 10, tzinfo=timezone.utc),
        date_evidence="new listing date", reporting_period=infer_reporting_period("quý 3 năm 2026")))
    assert fresh.candidate_id == old.candidate_id and fresh.status == CandidateStatus.DISCOVERED
    assert fresh.reason == "LISTING_REVISION" and fresh.etag is None
    assert "2026" in fresh.title


def test_stale_only_rag_never_calls_llm(store):
    ingest(store, "FPT báo cáo tài chính năm 2022", published="2026-10-17")
    class Provider:
        def parse_structured(self, *args):
            raise AssertionError("Stale evidence must not reach the LLM")
    result = RAGService(store, provider=Provider()).answer("Doanh thu FPT hiện tại?", ticker="FPT")
    assert result.status == "NO_EVIDENCE" and "Chưa có dữ liệu cập nhật" in result.message


class OfflineResearch:
    calls = 0
    def research(self, question, ticker=None):
        self.calls += 1
        return SimpleNamespace(discovery_citations=[], external_citations=[], failures=[], trace=[], plan=ResearchPlanner().plan(question, ticker))


def test_chat_searches_web_after_only_old_local_evidence(store):
    ingest(store, "FPT báo cáo tài chính năm 2022", published="2026-10-17")
    web = OfflineResearch()
    service = ChatService(research_store=store, rag_service=RAGService(store, enabled=False), research_orchestrator=web)
    result = service._research(ParsedQuery(raw_query="Tìm báo cáo FPT mới nhất", tickers=["FPT"], intent=QueryIntent.RESEARCH))
    assert web.calls == 1
    assert "Chưa có dữ liệu cập nhật" in result.message and "2022" in result.message
    assert result.tables and "lịch sử" in result.tables[0].title


def test_current_local_rag_displays_period_and_avoids_unneeded_web_search(store):
    ingest(store)
    web = OfflineResearch()
    service = ChatService(research_store=store, rag_service=RAGService(store, enabled=False), research_orchestrator=web)
    result = service._research(ParsedQuery(raw_query="Tìm báo cáo FPT mới nhất", tickers=["FPT"], intent=QueryIntent.RESEARCH))
    assert web.calls == 0
    assert "Dữ liệu đến kỳ" in result.message and "Quý 3/2026" in result.message
    assert result.tables[0].rows[0]["Dữ liệu đến kỳ"].startswith("Dữ liệu đến kỳ")


def test_research_agent_does_not_pass_old_local_reports_to_specialists(store):
    ingest(store, "FPT báo cáo tài chính năm 2022", published="2026-10-17")
    web = OfflineResearch()
    agent = ResearchAgent(store, web_research=web)
    state = AgentState(ticker="FPT", investment_horizon="3 months", user_goal="Analyze FPT")
    result = agent.run(state)
    assert web.calls == 1 and not result["research_evidence"]
    assert "2022" in result["research_freshness_note"]
    agent.run(state)
    assert web.calls == 1  # Failed-update results are cached to avoid repeated slow searches.


def test_current_research_checks_web_even_when_llm_planning_is_disabled(store):
    ingest(store, "FPT báo cáo tài chính năm 2022", published="2026-10-17")
    web = OfflineResearch()
    state = AgentState(ticker="FPT", investment_horizon="3 months", user_goal="Analyze FPT", allow_llm_planning=False)
    assert not ResearchAgent(store, web_research=web).run(state)["research_evidence"]
    assert web.calls == 1  # Latest-report verification no longer depends on LLM planning.


def test_agent_checks_newer_web_reports_even_with_eligible_local_report(store):
    ingest(store)
    from stock_multi_agent.models.research import EvidenceBundle
    local = store.search('FPT báo cáo tài chính', ticker='FPT', scope=ResearchScope('CURRENT'))
    assert local.citations
    selected_store = SimpleNamespace(search=lambda *args, **kwargs: EvidenceBundle(
        query=args[0], citations=local.citations, retrieval_note='eligible local fixture'))
    web = OfflineResearch()
    result = ResearchAgent(selected_store, web_research=web).run(
        AgentState(ticker='FPT', investment_horizon='3 months', user_goal='Analyze FPT'))
    assert web.calls == 1
    assert result['research_evidence']


def test_external_old_report_with_recent_publication_is_filtered(store):
    item = ExternalEvidenceCitation(title="FPT báo cáo tài chính năm 2022", source_url="https://example.invalid/fpt",
        canonical_url="https://example.invalid/fpt", source_domain="example.invalid", published_at=datetime(2026, 10, 17, tzinfo=timezone.utc),
        retrieved_at=datetime(2026, 10, 17, tzinfo=timezone.utc), content_sha256="a" * 64, quote="FPT doanh thu tăng trưởng.",
        evidence_location="fixture", relevance_score=.9, reporting_period=infer_reporting_period("FPT báo cáo tài chính năm 2022"))
    citation = external_to_citation(item, "FPT")
    assert freshness_reason(citation, ResearchScope("CURRENT"), NOW) == "STALE_REPORTING_PERIOD"
    assert "2022" in period_display(citation)
