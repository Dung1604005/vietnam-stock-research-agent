from __future__ import annotations

import pytest
from pathlib import Path
from datetime import datetime, timezone

from stock_multi_agent.models.analysis import FinancialMetrics
from stock_multi_agent.models.chat import ConversationState, ParsedQuery, QueryIntent, ResponseType
from stock_multi_agent.models.chart import ChartType
from stock_multi_agent.models.research import ResearchDocumentType
from stock_multi_agent.providers.market_provider import MarketDataError
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.chat_service import ChatService
from stock_multi_agent.services.research_store import ResearchStore
from stock_multi_agent.services.web_research import (
    DisabledCompanyNameResolver, DisabledIssuerDiscoveryProvider, DisabledWebSearchProvider, WebResearchService, WebSearchResult,
)

from conftest import conflict_market_data


class MultiTickerProvider:
    def __init__(self, tickers: list[str], missing_financials: bool = False) -> None:
        self.data = {}
        for ticker in tickers:
            data = conflict_market_data(ticker)
            if missing_financials:
                data = data.model_copy(update={"financials": FinancialMetrics()})
            self.data[ticker] = data

    def get_market_data(self, ticker: str, include_financials: bool = True):
        del include_financials
        if ticker not in self.data:
            raise MarketDataError(f"No data for {ticker}")
        return self.data[ticker]


@pytest.fixture
def chat_service() -> ChatService:
    provider = MultiTickerProvider(["FPT", "CMG", "VNINDEX"])
    return ChatService(
        analysis_service=AnalysisService(provider=provider),
        universe={"default": ["FPT", "CMG"], "technology": ["FPT", "CMG"]},
    )


def test_required_demo_conversation_keeps_context(chat_service: ChatService) -> None:
    conversation = ConversationState()
    analysis = chat_service.handle("Phân tích FPT cho tôi", conversation)
    assert analysis.response_type is ResponseType.ANALYSIS
    assert analysis.source_details[0].source_id == "test:ohlcv:FPT"
    assert analysis.data_fetched_at is not None
    assert conversation.active_tickers == ["FPT"]

    chart = chat_service.handle("Cho tôi biểu đồ 6 tháng", conversation)
    assert chart.response_type is ResponseType.CHART
    assert chart.charts[0].request.ticker == "FPT"
    assert chart.charts[0].request.period == "6M"
    assert chart.metadata["routing_plan"] == ["DATA", "CHART"]

    technical = chat_service.handle("RSI thế nào?", conversation)
    assert technical.tickers == ["FPT"]
    assert technical.metadata["routing_plan"] == ["DATA", "TECHNICAL"]
    called_tools = [event.tool for event in technical.agent_trace if event.tool]
    assert called_tools == ["get_stock_data", "RSI"]

    comparison = chat_service.handle("So với CMG?", conversation)
    assert comparison.response_type is ResponseType.COMPARISON
    assert comparison.tickers == ["FPT", "CMG"]
    assert len(comparison.tables[0].rows) == 2


def test_market_overview_and_discovery(chat_service: ChatService) -> None:
    conversation = ConversationState()
    overview = chat_service.handle("Tình hình chứng khoán Việt Nam hiện tại?", conversation)
    assert overview.response_type is ResponseType.MARKET_OVERVIEW
    assert overview.tickers == ["VNINDEX"]
    assert "Không có dữ liệu" in overview.message

    discovery = chat_service.handle("Có cổ phiếu công nghệ nào đáng chú ý?", conversation)
    assert discovery.response_type is ResponseType.STOCK_LIST
    assert discovery.tickers[:2] == ["FPT", "CMG"]
    assert "không đại diện toàn thị trường" in discovery.tables[0].note

    chart = chat_service.handle("Cho tôi biểu đồ RSI của mã đầu tiên", conversation)
    assert chart.charts[0].request.ticker == discovery.tickers[0]
    assert chart.charts[0].request.chart_type is ChartType.RSI


def test_discovery_honors_explicit_result_count_and_reports_shortfall() -> None:
    tickers = ["FPT", "CMG", "CTR", "ELC", "VNM", "HPG", "VCB", "TCB", "MBB", "ACB", "SSI", "HCM"]
    service = ChatService(
        analysis_service=AnalysisService(provider=MultiTickerProvider(tickers)),
        universe={"default": tickers},
    )
    query = ParsedQuery(
        raw_query="Tìm 10 mã tốt nhất", intent=QueryIntent.STOCK_DISCOVERY,
        requested_result_count=10, confidence=0.9,
    )
    response = service.handle(query.raw_query, ConversationState(), parsed_query=query)
    assert response.response_type is ResponseType.STOCK_LIST
    assert len(response.tickers) == 10
    assert len(response.tables[0].rows) == 10
    assert response.metadata["requested_result_count"] == 10
    assert response.metadata["returned_result_count"] == 10

    one = query.model_copy(update={"raw_query": "Tìm mã tốt nhất", "requested_result_count": 1})
    response = service.handle(one.raw_query, ConversationState(), parsed_query=one)
    assert len(response.tickers) == 1
    assert len(response.tables[0].rows) == 1

    shortfall = query.model_copy(update={"requested_result_count": 20})
    response = service.handle(shortfall.raw_query, ConversationState(), parsed_query=shortfall)
    assert len(response.tickers) == 10
    assert any("Requested 20" in warning for warning in response.warnings)


def test_long_term_follow_up_keeps_ticker(chat_service: ChatService) -> None:
    conversation = ConversationState()
    chat_service.handle("Phân tích FPT", conversation)
    response = chat_service.handle("Thế còn trong dài hạn?", conversation)
    assert response.response_type is ResponseType.ANALYSIS
    assert response.tickers == ["FPT"]
    assert conversation.investment_horizon == "1 year"


def test_unknown_ticker_returns_natural_error(chat_service: ChatService) -> None:
    response = chat_service.handle("Phân tích XYZ", ConversationState())
    assert response.response_type is ResponseType.TEXT
    assert "chưa lấy được dữ liệu" in response.message
    assert response.warnings


def test_missing_financial_data_does_not_crash() -> None:
    provider = MultiTickerProvider(["FPT"], missing_financials=True)
    service = ChatService(analysis_service=AnalysisService(provider=provider), universe={"default": ["FPT"]})
    response = service.handle("Phân tích FPT", ConversationState())
    assert response.response_type is ResponseType.ANALYSIS
    assert response.metadata["fundamental_signal"] == "UNKNOWN"


def test_chart_service_generates_deterministic_figure(chat_service: ChatService) -> None:
    response = chat_service.handle("Vẽ MA20 và MA50 FPT 1 năm", ConversationState())
    artifact = response.charts[0]
    assert artifact.request.chart_type is ChartType.PRICE_MA
    assert len(artifact.records) == 252
    figure = chat_service.charts.render_figure(artifact)
    assert len(figure.data) == 3


def test_semantic_multi_turn_coreferences(chat_service: ChatService) -> None:
    conversation = ConversationState()
    discovered = chat_service.handle("Có mã công nghệ nào đáng chú ý?", conversation)
    assert discovered.tickers == ["FPT", "CMG"]

    first = chat_service.handle("Con đầu tiên sao?", conversation)
    assert first.tickers == ["FPT"]

    indicator = chat_service.handle("Cho RSI", conversation)
    assert indicator.tickers == ["FPT"]

    compared = chat_service.handle("Thế so với con thứ 2?", conversation)
    assert compared.response_type is ResponseType.COMPARISON
    assert compared.tickers == ["FPT", "CMG"]

    charts = chat_service.handle("Vẽ cả hai", conversation)
    assert charts.response_type is ResponseType.CHART
    assert [item.request.ticker for item in charts.charts] == ["FPT", "CMG"]


def test_clarification_does_not_call_market_data(chat_service: ChatService) -> None:
    response = chat_service.handle("Cho tôi biểu đồ", ConversationState())
    assert response.response_type is ResponseType.TEXT
    assert "mã nào" in response.message
    assert response.metadata["routing_plan"] == ["CHAT"]
    assert response.metadata["semantic_understanding"]["ambiguities"] == []


def test_multi_intent_keeps_analysis_and_chart(chat_service: ChatService) -> None:
    response = chat_service.handle("Phân tích FPT và cho tôi biểu đồ 6 tháng", ConversationState())
    assert response.response_type is ResponseType.ANALYSIS
    assert response.charts[0].request.ticker == "FPT"
    assert response.charts[0].request.period == "6M"
    assert response.metadata["routing_plan"][-1] == "CHART"


def test_multi_intent_runs_requested_risk_secondary(chat_service: ChatService) -> None:
    query = ParsedQuery(
        raw_query="Kỹ thuật FPT và rủi ro",
        intent=QueryIntent.TECHNICAL_ANALYSIS,
        secondary_intents=[QueryIntent.RISK_ANALYSIS],
        tickers=["FPT"],
        confidence=0.9,
    )
    response = chat_service.handle(query.raw_query, ConversationState(), parsed_query=query)
    assert "risk" in response.metadata["secondary_outputs"]
    assert response.metadata["secondary_outputs"]["risk"]["risk_level"]
    assert "RISK" in response.metadata["routing_plan"]


def test_research_route_does_not_invent_evidence() -> None:
    store = ResearchStore(Path(__file__).resolve().parents[1] / ".runtime" / "test-artifacts" / "empty-research.sqlite3")
    query = ParsedQuery(raw_query="Tìm báo cáo FPT", intent=QueryIntent.RESEARCH, tickers=["FPT"], confidence=0.9)
    response = ChatService(
        research_store=store,
        web_research=WebResearchService(
            DisabledWebSearchProvider(), DisabledIssuerDiscoveryProvider(), DisabledCompanyNameResolver(),
        ),
    ).handle(
        query.raw_query, ConversationState(), parsed_query=query,
    )
    assert response.metadata["routing_plan"] == ["RESEARCH"]
    assert response.metadata["evidence_bundle"]["citations"] == []
    assert "Chưa có tài liệu" in response.message


class FakeWebSearchProvider:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, limit: int) -> list[WebSearchResult]:
        self.calls += 1
        return [WebSearchResult("FPT annual report", "https://fpt.com.vn/investor", "Official investor relations report.")]


class FakeIssuerDiscoveryProvider:
    name = "hnx_issuer_page"

    def search(self, ticker: str) -> list[WebSearchResult]:
        if ticker != "PNG":
            return []
        return [WebSearchResult(
            "HNX issuer profile and disclosures · PNG",
            "https://www.hnx.vn/vi-vn/cophieu-etfs/chi-tiet-chung-khoan-uc-PNG.html",
            "Official HNX issuer profile and disclosure page for ticker PNG. Annual report available on the source.",
        )]


def test_research_uses_live_discovery_only_after_local_miss() -> None:
    store = ResearchStore(Path(__file__).resolve().parents[1] / ".runtime" / "test-artifacts" / "web-fallback.sqlite3")
    provider = FakeWebSearchProvider()
    query = ParsedQuery(raw_query="Tìm báo cáo FPT", intent=QueryIntent.RESEARCH, tickers=["FPT"], confidence=0.9)
    response = ChatService(
        research_store=store,
        web_research=WebResearchService(provider, DisabledIssuerDiscoveryProvider(), DisabledCompanyNameResolver()),
    ).handle(
        query.raw_query, ConversationState(), parsed_query=query,
    )
    assert provider.calls == 3
    assert response.metadata["retrieval_mode"] == "web_fallback"
    assert response.metadata["evidence_bundle"]["web_citations"][0]["source_url"].startswith("https://fpt.com.vn/")
    assert response.tables[0].title == "Live web results (not verified evidence)"


def test_research_includes_png_official_hnx_discovery_alongside_web_search() -> None:
    store = ResearchStore(Path(__file__).resolve().parents[1] / ".runtime" / "test-artifacts" / "png-hnx.sqlite3")
    web_provider = FakeWebSearchProvider()
    query = ParsedQuery(raw_query="Tìm cho tôi báo cáo PNG", intent=QueryIntent.RESEARCH, tickers=["PNG"], confidence=0.9)
    response = ChatService(
        research_store=store,
        web_research=WebResearchService(web_provider, FakeIssuerDiscoveryProvider()),
    ).handle(query.raw_query, ConversationState(), parsed_query=query)
    assert web_provider.calls == 3
    assert response.metadata["retrieval_mode"] == "web_fallback"
    assert "https://www.hnx.vn/vi-vn/cophieu-etfs/chi-tiet-chung-khoan-uc-PNG.html" in response.sources
    assert response.metadata["evidence_bundle"]["web_citations"][0]["search_provider"] == "hnx_issuer_page"


def test_research_keeps_a_local_citation_ahead_of_web_fallback() -> None:
    store = ResearchStore(Path(__file__).resolve().parents[1] / ".runtime" / "test-artifacts" / "local-first.sqlite3")
    store.ingest_text(
        text="FPT annual report describes revenue growth.", title="FPT annual report",
        document_type=ResearchDocumentType.ANNUAL_REPORT, source_url="https://example.com/fpt-report",
        publisher="FPT IR", published_at=datetime(2026, 1, 1, tzinfo=timezone.utc), tickers=["FPT"],
    )
    provider = FakeWebSearchProvider()
    query = ParsedQuery(raw_query="Tìm báo cáo FPT", intent=QueryIntent.RESEARCH, tickers=["FPT"], confidence=0.9)
    response = ChatService(
        research_store=store,
        web_research=WebResearchService(provider, DisabledIssuerDiscoveryProvider(), DisabledCompanyNameResolver()),
    ).handle(
        query.raw_query, ConversationState(), parsed_query=query,
    )
    assert provider.calls == 0
    assert response.metadata["retrieval_mode"] == "local"
    assert response.metadata["evidence_bundle"]["citations"]
