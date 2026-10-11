from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from stock_multi_agent.models.research import ResearchDocumentType
from stock_multi_agent.services.research_store import ResearchIngestionError, ResearchStore
from stock_multi_agent.services.web_research import (
    DisabledCompanyNameResolver, DisabledIssuerDiscoveryProvider, WebResearchService, WebSearchResult,
)


TEST_ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / ".runtime" / "test-artifacts"


def test_research_store_requires_provenance_and_returns_citations() -> None:
    store = ResearchStore(TEST_ARTIFACT_ROOT / "research" / "research.sqlite3")
    document = store.ingest_text(
        text="FPT reported revenue growth and noted demand from overseas technology services.",
        title="FPT earnings release", document_type=ResearchDocumentType.EARNINGS_RELEASE,
        source_url="https://example.com/fpt-release", publisher="FPT IR",
        published_at=datetime(2026, 10, 1, tzinfo=timezone.utc), tickers=["FPT"],
    )
    bundle = store.search("revenue growth technology", ticker="FPT")
    assert bundle.citations
    citation = bundle.citations[0]
    assert citation.document_id == document.document_id
    assert str(citation.source_url).startswith("https://example.com/")
    assert citation.content_sha256 == document.content_sha256


def test_research_store_rejects_naive_publication_time() -> None:
    store = ResearchStore(TEST_ARTIFACT_ROOT / "research" / "invalid.sqlite3")
    with pytest.raises(ResearchIngestionError, match="timezone"):
        store.ingest_text(
            text="verified text", title="Document", document_type=ResearchDocumentType.OTHER,
            source_url="https://example.com/document", publisher="Example", published_at=datetime(2026, 10, 1),
        )


def test_research_store_expands_vietnamese_report_terms_for_english_document() -> None:
    store = ResearchStore(TEST_ARTIFACT_ROOT / "research" / "vi-en-alias.sqlite3")
    store.ingest_text(
        text="The annual report discusses long-term strategy.", title="Annual report",
        document_type=ResearchDocumentType.ANNUAL_REPORT, source_url="https://example.com/annual-report",
        publisher="Issuer IR", published_at=datetime(2026, 10, 1, tzinfo=timezone.utc), tickers=["ABC"],
    )
    bundle = store.search("Tìm báo cáo ABC", ticker="ABC")
    assert bundle.citations


def test_research_store_returns_at_most_one_best_chunk_per_document() -> None:
    store = ResearchStore(TEST_ARTIFACT_ROOT / "research" / "document-dedup.sqlite3")
    store.ingest_text(
        text=("annual report " * 500), title="Annual report", document_type=ResearchDocumentType.ANNUAL_REPORT,
        source_url="https://example.com/report", publisher="Issuer IR",
        published_at=datetime(2026, 10, 1, tzinfo=timezone.utc), tickers=["ABC"],
    )
    bundle = store.search("annual report", ticker="ABC", limit=5)
    assert len(bundle.citations) == 1


class FakeWebSearchProvider:
    name = "fake"

    def __init__(self, results: list[WebSearchResult]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str, limit: int) -> list[WebSearchResult]:
        self.queries.append(query)
        return self.results[:limit]


class FakeIssuerDiscoveryProvider:
    name = "hnx_issuer_page"

    def __init__(self) -> None:
        self.tickers: list[str] = []

    def search(self, ticker: str) -> list[WebSearchResult]:
        self.tickers.append(ticker)
        return [WebSearchResult(
            "HNX issuer profile and disclosures · PNG",
            "https://www.hnx.vn/vi-vn/cophieu-etfs/chi-tiet-chung-khoan-uc-PNG.html",
            "Official HNX issuer profile and disclosure page for ticker PNG. Annual report available on the source.",
        )]


class FakeCompanyNameResolver:
    def __init__(self, name: str | None) -> None:
        self.name = name
        self.tickers: list[str] = []

    def resolve(self, ticker: str) -> str | None:
        self.tickers.append(ticker)
        return self.name


def test_web_research_records_source_and_retrieval_timestamp() -> None:
    provider = FakeWebSearchProvider([
        WebSearchResult("FPT annual report", "https://fpt.com.vn/investor", "Official investor report."),
    ])
    citations = WebResearchService(
        provider, DisabledIssuerDiscoveryProvider(), DisabledCompanyNameResolver(),
    ).search("FPT annual report", ticker="FPT")
    assert len(citations) == 1
    assert citations[0].source_domain == "fpt.com.vn"
    assert citations[0].search_provider == "fake"
    assert citations[0].retrieved_at.tzinfo is not None
    assert provider.queries == ["FPT Vietnam stock FPT annual report annual report"]


def test_web_research_combines_official_issuer_lookup_with_web_sources() -> None:
    web_provider = FakeWebSearchProvider([])
    issuer_provider = FakeIssuerDiscoveryProvider()
    citations = WebResearchService(web_provider, issuer_provider).search("Tìm cho tôi báo cáo PNG", ticker="PNG")
    assert issuer_provider.tickers == ["PNG"]
    assert web_provider.queries
    assert citations[0].search_provider == "hnx_issuer_page"
    assert citations[0].source_domain == "www.hnx.vn"


def test_web_research_uses_company_name_when_official_lookup_misses() -> None:
    provider = FakeWebSearchProvider([
        WebSearchResult("PNG annual report", "https://example.com/png-report", "Phu Nhuan Trading annual report."),
    ])
    resolver = FakeCompanyNameResolver("Phu Nhuan Trading Joint Stock Company")
    citations = WebResearchService(provider, DisabledIssuerDiscoveryProvider(), resolver).search(
        "Tìm cho tôi báo cáo PNG", ticker="PNG",
    )
    assert citations
    assert resolver.tickers == ["PNG"]
    assert "Phu Nhuan Trading Joint Stock Company" in provider.queries[0]
