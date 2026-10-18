from __future__ import annotations

import json
from datetime import timedelta
from email.message import Message
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from stock_multi_agent.models.research import FetchedDocument, ResearchDocumentType, WebCitation
from stock_multi_agent.models.research_crawler import CandidateStatus, DocumentCandidate, SourceDefinition, StorageMode, utcnow
from stock_multi_agent.services.content_fetcher import ContentFetchError, SafeContentFetcher
from stock_multi_agent.services.document_metadata import metadata_problem, publication_date
from stock_multi_agent.services.research_crawl_store import CandidateRepository
from stock_multi_agent.services.research_crawl_transport import CrawlPolicyError, CrawlTransport
from stock_multi_agent.services.research_discovery import ResearchDiscovery
from stock_multi_agent.services.research_store import ResearchStore
from stock_multi_agent.services.research_sync import ResearchSyncService
from stock_multi_agent.services.source_adapters.rss import RssAdapter
from stock_multi_agent.services.source_adapters.sitemap import SitemapAdapter
from stock_multi_agent.services.source_adapters.html_listing import HtmlListingAdapter
from stock_multi_agent.services.source_adapters.json_listing import JsonListingAdapter
from stock_multi_agent.services.source_registry import SourceRegistry
from stock_multi_agent.services.embeddings import LocalHashEmbeddingProvider


ROOT = "https://example.invalid/ir"
DOC = ROOT + "/financial-report.html"


@pytest.fixture
def setup_crawler():
    root = Path(__file__).resolve().parents[1] / ".runtime" / "crawler-tests"
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root) as directory:
        directory = Path(directory)
        registry_path = directory / "sources.json"
        registry_path.write_text(json.dumps({"sources": [{"source_id": "issuer", "domain": "example.invalid",
            "publisher": "Fixture issuer", "quality_tier": "T1", "usage_note": "Authored fixture permission",
            "entrypoints": [{"url": ROOT}], "ticker_map": ["FPT"], "discovery_allowed": True,
            "fetch_allowed": True, "storage_mode": "FULL_TEXT", "allowed_path_prefixes": ["/ir"],
            "min_request_interval_seconds": 0, "max_depth": 0}]}), encoding="utf-8")
        registry = SourceRegistry(registry_path)
        store = ResearchStore(directory / "research.sqlite3", embedding_provider=LocalHashEmbeddingProvider())
        yield store, registry


def response(url, text, status=200):
    return FetchedDocument(requested_url=url, final_url=url, content_type="text/html", status_code=status,
                           body=text, retrieved_at=utcnow(), etag='"v1"', byte_size=len(text))


class FakeTransport:
    def __init__(self, *, text=None, error=None):
        self.requests = self.bytes_downloaded = 0
        self.text, self.error = text, error

    def fetch(self, url, **kwargs):
        self.requests += 1
        if url == ROOT:
            text = '<a href="/ir/financial-report.html">FPT báo cáo tài chính</a>'
        else:
            if self.error:
                raise self.error
            text = self.text or '<title>FPT báo cáo tài chính</title><meta property="article:published_time" content="2025-04-01T09:00:00+07:00"><p>' + "FPT doanh thu lợi nhuận tài chính 100 tỷ đồng. " * 12 + "</p>"
        self.bytes_downloaded += len(text)
        return response(url, text)


def make_sync(setup, **kwargs):
    store, registry = setup
    return ResearchSyncService(store, registry, transport_factory=lambda source, deadline: FakeTransport(**kwargs))


def counts(store):
    with store._connect() as db:
        return tuple(db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in
                     ("research_documents", "research_chunks", "research_chunk_embeddings"))


def test_end_to_end_discovery_ingest_retrieve_and_no_reembedding(setup_crawler):
    sync = make_sync(setup_crawler)
    report = sync.sync(force_discovery=True)
    assert report.counts["INDEXED"] == 1
    assert sync.store.search("doanh thu lợi nhuận FPT", ticker="FPT").citations
    first = counts(sync.store)
    candidate = sync.repository.list()[0]
    candidate.last_checked_at = utcnow() - timedelta(days=4)
    sync.repository.save(candidate)
    with patch.object(sync.store, "_embed_chunks", wraps=sync.store._embed_chunks) as embed:
        rerun = sync.sync(force_discovery=True)
        assert rerun.counts["UNCHANGED"] == 1
        assert embed.call_count == 0
    assert counts(sync.store) == first


def test_changed_document_keeps_revision_history(setup_crawler):
    sync = make_sync(setup_crawler)
    sync.sync(force_discovery=True)
    first = sync.repository.list()[0]
    first.last_checked_at = utcnow() - timedelta(days=4)
    sync.repository.save(first)
    text = '<title>FPT financial statements corrected</title><meta property="article:published_time" content="2025-04-02T09:00:00+07:00">' + "FPT doanh thu 150 tỷ đồng. " * 20
    make_sync(setup_crawler, text=text).sync(queue_only=True)
    with sync.store._connect() as db:
        rows = db.execute("SELECT document_id,version,supersedes_id FROM research_documents ORDER BY version").fetchall()
    assert len(rows) == 2 and rows[1][1] == 2 and rows[1][2] == rows[0][0]
    with sync.store._connect() as db:
        assert db.execute("SELECT published_at FROM research_documents WHERE version=2").fetchone()[0].startswith("2025-04-02")


def test_permission_gate_and_per_document_review_survive_restart(setup_crawler):
    sync = make_sync(setup_crawler)
    sync.registry.sources[0].storage_mode = StorageMode.LINK_ONLY
    report = sync.sync(force_discovery=True)
    assert report.counts["QUARANTINED"] == 1 and counts(sync.store)[0] == 0
    candidate = CandidateRepository(sync.store).list()[0]
    assert not candidate.excerpt and candidate.reason == "PERMISSION_REQUIRED"
    sync.repository.review(candidate.candidate_id, "approve", note="Authored fixture allowed for indexing")
    sync.sync(queue_only=True)
    assert counts(sync.store)[0] == 1
    assert sync.registry.sources[0].storage_mode == StorageMode.LINK_ONLY


@pytest.mark.parametrize("text,reason", [
    ("<title>FPT financial statements</title>" + "FPT company earnings report. " * 12, "MISSING_PUBLICATION_DATE"),
    ("<title>Access denied</title>" + "Access denied. CAPTCHA verify you are human. " * 12, "LOGIN_OR_ERROR_PAGE"),
])
def test_bad_document_is_quarantined_not_indexed(setup_crawler, text, reason):
    sync = make_sync(setup_crawler, text=text)
    sync.sync(force_discovery=True)
    assert sync.repository.list()[0].reason == reason
    assert counts(sync.store)[0] == 0


def test_locks_are_exclusive_and_owner_scoped(setup_crawler):
    store, registry = setup_crawler
    one, two = CandidateRepository(store), CandidateRepository(store)
    assert one.acquire("issuer", "a")
    assert not two.acquire("issuer", "b")
    two.release("issuer", "b")
    assert not two.acquire("issuer", "b")
    one.release("issuer", "a")
    assert two.acquire("issuer", "b")


@pytest.mark.parametrize("code,retry", [(429, True), (503, True), (403, False), (404, False)])
def test_retry_classification_and_persistent_backoff(setup_crawler, code, retry):
    sync = make_sync(setup_crawler, error=ContentFetchError("fixture", code))
    sync.sync(force_discovery=True)
    candidate = sync.repository.list()[0]
    assert candidate.status == (CandidateStatus.RETRYABLE_ERROR if retry else CandidateStatus.PERMANENT_ERROR)
    if retry:
        assert candidate.next_attempt_at > utcnow()
        assert not sync.repository.pending(sync.registry.sources[0])


def test_discovery_preserves_parent_and_avoids_foreign_paths(setup_crawler):
    sync = make_sync(setup_crawler)
    class Listing(FakeTransport):
        def fetch(self, url, **kwargs):
            return response(url, '<a href="/private/report.pdf">other path</a><a href="https://evil.invalid/report.pdf">foreign</a>'
                                 '<a href="/ir/report.pdf?utm_source=a">report</a><a href="/ir/report.pdf">report duplicate</a>')
    ResearchDiscovery().discover(sync.registry.sources[0], Listing(), sync.repository, {}, force=True)
    candidates = sync.repository.list()
    assert len(candidates) == 1
    assert str(candidates[0].source_url) == ROOT + "/report.pdf"
    assert str(candidates[0].parent_url) == ROOT


def test_source_adapters_do_not_mistake_lastmod_for_publication():
    rss = RssAdapter().discover('<rss><channel><item><title>Report</title><pubDate>Tue, 01 Apr 2025 09:00:00 +0700</pubDate><link>/ir/report</link></item></channel></rss>', ROOT)
    assert publication_date(rss[0].published, "Asia/Ho_Chi_Minh").utcoffset() == timedelta(hours=7)
    atom = RssAdapter().discover('<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>A</title><published>2025-04-01T00:00:00Z</published><link href="/ir/report.pdf"/></entry></feed>', ROOT)
    assert atom[0].url.endswith(".pdf")
    sitemap = SitemapAdapter().discover('<urlset><url><loc>https://example.invalid/ir/report</loc><lastmod>2025-05-01</lastmod></url></urlset>', ROOT)
    assert not sitemap[0].published
    listing = HtmlListingAdapter().discover('<div class="report-item"><h3>Financial statements</h3><time>01/04/2025</time><a href="/ir/report.pdf">Download</a></div>', ROOT)
    assert listing[0].published == "01/04/2025" and "Financial statements" in listing[0].title
    api = JsonListingAdapter().discover('{"items":[{"url":"/ir/report.pdf","title":"Report","published_at":"2025-04-01"}]}', ROOT)
    assert api[0].published == "2025-04-01"
    with pytest.raises(ValueError):
        SitemapAdapter().discover('<!DOCTYPE doc [<!ENTITY x "bad">]><urlset/>', ROOT)


def test_source_paths_cannot_be_escaped_or_spoofed(setup_crawler):
    source = setup_crawler[1].sources[0]
    for url in ("https://example.invalid.evil/ir/report.pdf", "https://example.invalid/ir/../private/report.pdf",
                "https://example.invalid/ir/%2e%2e/private/report.pdf", "https://user:pass@example.invalid/ir/report.pdf",
                "http://example.invalid/ir/report.pdf", "https://example.invalid/ir-other/report.pdf"):
        assert not source.accepts(url)
    assert source.accepts(DOC)


def test_per_hop_policy_stops_redirect_before_second_request(setup_crawler):
    source = setup_crawler[1].sources[0]
    transport = CrawlTransport(source, deadline=__import__("time").monotonic() + 30)
    headers = Message()
    headers["Location"] = "https://example.invalid/private/report.pdf"
    fetcher = SafeContentFetcher(resolver=lambda hostname: ["8.8.8.8"], url_guard=transport._guard)
    with patch.object(transport, "_robots") as robots, patch.object(fetcher, "_open_once", return_value=(302, headers, b"")) as opened:
        robots.return_value.can_fetch.return_value = True
        robots.return_value.crawl_delay.return_value = None
        with pytest.raises(CrawlPolicyError):
            fetcher.fetch(DOC)
        assert opened.call_count == 1


def test_robots_denial_stops_target_request(setup_crawler):
    transport = CrawlTransport(setup_crawler[1].sources[0], deadline=__import__("time").monotonic() + 30)
    with patch.object(transport, "_robots") as robots:
        robots.return_value.can_fetch.return_value = False
        with pytest.raises(CrawlPolicyError, match="ROBOTS_DISALLOWED"):
            transport._guard(DOC)
    assert transport.requests == 0


def test_chat_candidates_are_durable_and_snippets_never_become_evidence(setup_crawler):
    sync = make_sync(setup_crawler)
    citation = WebCitation(title="FPT financial statements", source_url=DOC, source_domain="example.invalid",
                          snippet="UNTRUSTED SEARCH SNIPPET", search_provider="fixture", retrieved_at=utcnow(), rank=1)
    queued = sync.enqueue_chat([citation], ticker="FPT")
    assert len(queued) == 1
    assert CandidateRepository(sync.store).get(queued[0]["candidate_id"]).excerpt == ""
    assert counts(sync.store)[0] == 0


def test_304_does_not_extract_or_reembed(setup_crawler):
    sync = make_sync(setup_crawler)
    sync.sync(force_discovery=True)
    candidate = sync.repository.list()[0]
    candidate.last_checked_at = utcnow() - timedelta(days=4)
    sync.repository.save(candidate)
    class NotModified(FakeTransport):
        def fetch(self, url, **kwargs):
            assert kwargs["etag"]
            return response(url, "", 304)
    sync.transport_factory = lambda source, deadline: NotModified()
    with patch.object(sync.store, "_embed_chunks") as embed:
        report = sync.sync(queue_only=True)
        assert report.counts["UNCHANGED"] == 1
        embed.assert_not_called()


def test_reviewer_cannot_supply_naive_publication_date(setup_crawler):
    sync = make_sync(setup_crawler)
    item = sync.repository.upsert(DocumentCandidate(source_id="issuer", source_url=DOC))
    with pytest.raises(ValueError):
        sync.repository.review(item.candidate_id, "approve", note="fixture", published_at=__import__("datetime").datetime(2025, 1, 1))


def test_issuer_cdn_is_explicit_and_navigation_is_not_a_document(setup_crawler):
    source = setup_crawler[1].sources[0]
    source.asset_domains = {"files.example.invalid": ["/issuer"]}
    assert source.accepts("https://files.example.invalid/issuer/report.pdf")
    assert not source.accepts("https://files.example.invalid/other/report.pdf")
    from stock_multi_agent.services.research_discovery import is_navigation
    assert is_navigation(source, ROOT + "?page=2")
    assert not is_navigation(source, DOC)


def test_fpt_download_and_view_buttons_share_identity():
    from stock_multi_agent.services.source_adapters.fpt import FptListingAdapter
    body = '<div class="flex"><h3>FPT Báo cáo tài chính năm 2025</h3><span>Cập nhật: 01/04/2025</span>' \
           '<a href="/api/download?url=%2Fapi%2Fmedia%2Freport.pdf" title="Tải xuống"></a>' \
           '<a href="/api/media/report.pdf" title="Xem trực tuyến"></a></div>'
    links = FptListingAdapter().discover(body, "https://fpt.com/vi/nha-dau-tu/thong-tin-cong-bo")
    assert len({item.url for item in links}) == 1
    assert all(item.published == "01/04/2025" for item in links)
    assert all("FPT Báo cáo" in item.title for item in links)


def test_budget_preserves_frontier_and_candidates(setup_crawler):
    from stock_multi_agent.services.research_crawl_transport import CrawlBudgetExceeded
    sync = make_sync(setup_crawler)
    source = sync.registry.sources[0]
    source.max_depth = 1
    class Paused(FakeTransport):
        def fetch(self, url, **kwargs):
            if url != ROOT:
                raise CrawlBudgetExceeded("fixture")
            return response(url, '<a href="/ir/financial-report.html">FPT financial statements</a>')
    sync.transport_factory = lambda source, deadline: Paused()
    report = sync.sync(force_discovery=True)
    assert report.status == "PAUSED_BUDGET"
    assert sync.repository.list()
    assert sync.repository.state(source.source_id)["frontier"][0]["url"] == DOC


def test_scanned_pdf_is_quarantined(setup_crawler):
    sync = make_sync(setup_crawler)
    with patch("stock_multi_agent.services.research_sync.HtmlContentExtractor.extract", side_effect=ValueError("Scanned PDF has no text layer; OCR required")):
        sync.sync(force_discovery=True)
    assert sync.repository.list()[0].reason == "OCR_REQUIRED"
    assert counts(sync.store)[0] == 0


def test_one_failed_candidate_does_not_stop_the_rest(setup_crawler):
    sync = make_sync(setup_crawler)
    class Mixed(FakeTransport):
        def fetch(self, url, **kwargs):
            if url == ROOT:
                return response(url, '<a href="/ir/financial-report.html">FPT report</a><a href="/ir/bad-report.html">Bad report</a>')
            if "bad-report" in url:
                raise ContentFetchError("fixture", 503)
            return super().fetch(url, **kwargs)
    sync.transport_factory = lambda source, deadline: Mixed()
    report = sync.sync(force_discovery=True)
    assert report.counts["INDEXED"] == 1
    assert report.counts["RETRYABLE_ERROR"] == 1
    assert report.status == "COMPLETED_WITH_ISSUES"


def test_source_permission_change_releases_queue_and_queue_run_does_not_delay_discovery(setup_crawler):
    sync = make_sync(setup_crawler)
    source = sync.registry.sources[0]
    source.storage_mode = StorageMode.LINK_ONLY
    sync.sync(force_discovery=True)
    old_due = sync.repository.state(source.source_id)["next_due_at"]
    source.storage_mode = StorageMode.FULL_TEXT
    assert sync.repository.pending(source)
    sync.sync(queue_only=True)
    assert counts(sync.store)[0] == 1
    assert sync.repository.state(source.source_id)["next_due_at"] == old_due
