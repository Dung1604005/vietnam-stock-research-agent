from __future__ import annotations

import pytest
from pathlib import Path
from datetime import date, datetime, timezone

from stock_multi_agent.models.analysis import FinancialMetrics, Recommendation, RiskLevel
from stock_multi_agent.models.chat import ConversationState, ParsedQuery, QueryIntent, ResponseType
from stock_multi_agent.models.chart import ChartType
from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.models.screening import (
    DualRecommendation,
    RecommendationBucket,
    RecommendationCandidate,
)
from stock_multi_agent.models.research import ResearchDocumentType
from stock_multi_agent.providers.market_provider import MarketDataError
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.chat_service import ChatService
from stock_multi_agent.services.llm_analyst import LLMAnalystService
from stock_multi_agent.services.agent_llm import AgentLLMAdvisor
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


class FakeAnalystProvider:
    def parse_structured(self, system, payload, output_model):
        del system
        return output_model.model_validate({
            "summary": "Dữ liệu hiện tại cho thấy cần theo dõi xác nhận thêm.",
            "bull_case": ["Tín hiệu kỹ thuật đã được cung cấp trong facts."],
            "bear_case": ["Rủi ro trong facts vẫn cần được tôn trọng."],
            "data_limitations": ["Không suy diễn ngoài dữ liệu được cung cấp."],
            "cited_source_ids": [payload["allowed_source_ids"][0], "invented:source"],
        })


class FakeSpecialistProvider:
    def __init__(self) -> None:
        self.payloads = []

    def parse_structured(self, system, payload, output_model):
        del system
        self.payloads.append(payload)
        citations = payload["allowed_source_ids"][:1] + ["invented:source"]
        if output_model.__name__ == "_TechnicalIndicatorPlan":
            return output_model.model_validate({
                "selected_indicators": ["MAX_DRAWDOWN", "SUPPORT_RESISTANCE"],
                "rationale": "Cần đo mức giảm và vùng giá quan trọng từ dữ liệu hiện có.",
                "cited_source_ids": citations,
            })
        return output_model.model_validate({
            "assessment": f"Đánh giá có cấu trúc cho {payload['specialist_role']}.",
            "supporting_observations": [
                "Chỉ sử dụng facts đã cung cấp.",
                "Các tín hiệu định lượng được giữ nguyên.",
                "Nguồn dữ liệu được giới hạn theo danh sách cho phép.",
            ],
            "counterarguments": [
                "Kết quả còn phụ thuộc chất lượng dữ liệu.",
                "Dự báo không bảo đảm diễn biến tương lai.",
            ],
            "missing_information": ["Không suy diễn dữ liệu chưa có."],
            "cited_source_ids": citations,
        })


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
    assert called_tools == ["get_stock_data", "price_forecast", "RSI"]

    comparison = chat_service.handle("So với CMG?", conversation)
    assert comparison.response_type is ResponseType.COMPARISON
    assert comparison.tickers == ["FPT", "CMG"]
    assert len(comparison.tables[0].rows) == 2
    assert "Confidence" not in comparison.tables[0].columns
    assert all("Confidence" not in row for row in comparison.tables[0].rows)


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
    assert len(response.tickers) == 12
    assert any("Requested 20" in warning for warning in response.warnings)


def test_recommendation_dispatch_renders_two_buckets_and_exact_count() -> None:
    def candidate(ticker: str, bucket: RecommendationBucket) -> RecommendationCandidate:
        return RecommendationCandidate(
            ticker=ticker,
            bucket=bucket,
            rank_in_bucket=1,
            technical_score=8.0,
            fundamental_score=7.0,
            risk_score=6.0 if bucket is RecommendationBucket.HIGH_REWARD_HIGH_RISK else 3.0,
            risk_level=RiskLevel.HIGH if bucket is RecommendationBucket.HIGH_REWARD_HIGH_RISK else RiskLevel.LOW,
            recommendation=Recommendation.BUY,
            confidence=0.7,
            volatility=0.4 if bucket is RecommendationBucket.HIGH_REWARD_HIGH_RISK else 0.18,
            reasoning="Meets the disclosed deterministic bucket policy.",
            as_of=date(2026, 10, 9),
            source_ids=[f"fixture:{ticker}"],
        )

    class FakeRecommendationService:
        def recommend(self, **kwargs):
            assert kwargs["requested_count"] == 2
            assert kwargs["max_quick_screen"] == 100
            return DualRecommendation(
                high_reward=[candidate("FPT", RecommendationBucket.HIGH_REWARD_HIGH_RISK)],
                safe_picks=[candidate("VNM", RecommendationBucket.SAFE_MODERATE_RETURN)],
                universe_key="default",
                universe_size=100,
                screened_count=20,
                analyzed_count=4,
                requested_count=2,
                returned_count=2,
                horizon="3 months",
            )

    service = ChatService(
        analysis_service=AnalysisService(provider=MultiTickerProvider(["FPT", "VNM"])),
        recommendation_service=FakeRecommendationService(),
        universe={"default": ["FPT", "VNM"]},
    )
    query = ParsedQuery(
        raw_query="Tìm 2 mã tốt nhất để mua",
        intent=QueryIntent.STOCK_RECOMMENDATION,
        requested_result_count=2,
        confidence=0.95,
    )
    response = service.handle(query.raw_query, ConversationState(), parsed_query=query)
    assert response.tickers == ["FPT", "VNM"]
    assert len(response.tables) == 2
    assert all("Confidence" not in table.columns for table in response.tables)
    assert all("Confidence" not in row for table in response.tables for row in table.rows)
    assert response.metadata["returned_result_count"] == 2
    assert response.metadata["routing_plan"] == ["SCREENING", "RECOMMENDATION"]


def test_manager_llm_can_promote_natural_buy_question_to_recommendation_route() -> None:
    class FakeManagerProvider:
        def parse_structured(self, system, payload, output_model):
            assert "Manager Agent" in system
            assert payload["requested_actions"] == ["BUY"]
            return output_model.model_validate({
                "intent": "STOCK_RECOMMENDATION",
                "confidence": 0.96,
            })

    class EmptyRecommendationService:
        def recommend(self, **kwargs):
            return DualRecommendation(
                universe_key="default",
                universe_size=100,
                screened_count=0,
                analyzed_count=0,
                requested_count=kwargs["requested_count"],
                returned_count=0,
                horizon=kwargs["horizon"],
            )

    manager_provider = FakeManagerProvider()
    analysis = AnalysisService(
        provider=MultiTickerProvider(["FPT"]),
        llm_advisor=AgentLLMAdvisor(provider=manager_provider, enabled=False),
        manager_routing_enabled=True,
    )
    service = ChatService(
        analysis_service=analysis,
        recommendation_service=EmptyRecommendationService(),
        universe={"default": ["FPT"]},
    )
    query = ParsedQuery(
        raw_query="Tôi nên mua cổ phiếu nào hôm nay",
        normalized_query="toi nen mua co phieu nao hom nay",
        intent=QueryIntent.GENERAL_CHAT,
        requested_actions=["BUY"],
        confidence=0.55,
    )
    response = service.handle(query.raw_query, ConversationState(), parsed_query=query)

    assert response.metadata["routing_plan"] == ["SCREENING", "RECOMMENDATION"]
    assert response.metadata["parsed_intent"] == "STOCK_RECOMMENDATION"
    assert response.metadata["manager_routing"]["accepted"] is True
    assert response.metadata["manager_routing"]["llm_used"] is True


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


def test_llm_analyst_adds_grounded_commentary_without_overriding_policy() -> None:
    service = ChatService(
        analysis_service=AnalysisService(provider=MultiTickerProvider(["FPT"])),
        universe={"default": ["FPT"]},
        llm_analyst=LLMAnalystService(provider=FakeAnalystProvider(), enabled=True),
    )
    response = service.handle("Phân tích FPT", ConversationState())

    analyst = response.metadata["llm_analyst"]
    assert analyst["status"] == "grounded_commentary"
    assert analyst["does_not_override_deterministic_decision"] is True
    assert analyst["cited_source_ids"] == ["test:ohlcv:FPT"]
    assert "Nhận định bổ sung từ Local LLM" in response.message


def test_only_technical_agent_uses_llm_for_bounded_tool_planning() -> None:
    provider = FakeSpecialistProvider()
    advisor = AgentLLMAdvisor(provider=provider, enabled=True)
    advisor.roles = {'technical'}  # Role allowlist still preserves the lightweight profile.
    service = ChatService(
        analysis_service=AnalysisService(
            provider=MultiTickerProvider(["FPT", "VNINDEX"]),
            llm_advisor=advisor,
        ),
        universe={"default": ["FPT"]},
    )
    response = service.handle("Phân tích FPT", ConversationState())

    reviews = response.metadata["llm_specialist_reviews"]
    roles = {item["agent"] for item in reviews if item["status"] == "GROUNDED"}
    assert roles == {"technical"}
    assert all("invented:source" not in item["cited_source_ids"] for item in reviews)
    assert response.metadata["recommendation"] in {"BUY", "HOLD", "WATCH", "AVOID"}
    assert all(item["controls_numeric_output"] is False for item in reviews)
    assert "Nhận định từ LLM" in response.message
    assert not [item for item in provider.payloads if item.get("specialist_role") in {"data", "fundamental", "risk", "manager"}]
    technical_plans = [item for item in provider.payloads if item.get("task") == "Chọn phép tính kỹ thuật bổ sung"]
    assert technical_plans
    assert technical_plans[0]["facts"]["forecast_agent"] is not None


def test_specialist_llm_retries_non_vietnamese_output_once() -> None:
    class LanguageRepairProvider:
        def __init__(self) -> None:
            self.calls = 0

        def parse_structured(self, system, payload, output_model):
            del system, payload
            self.calls += 1
            if output_model.__name__ == "_CompactVietnameseReview":
                return output_model.model_validate({
                    "lines": [
                        "Dữ liệu cho thấy cần tránh rủi ro giảm giá hiện tại.",
                        "Kết luận định lượng của hệ thống được giữ nguyên.",
                        "Dữ liệu giá đã được cung cấp để đánh giá.",
                        "Rủi ro đã được đo lường từ các chỉ số đầu vào.",
                        "Các tín hiệu định lượng được đối chiếu trước khi kết luận.",
                        "Dự báo vẫn còn bất định và không bảo đảm diễn biến tương lai.",
                        "Chất lượng dữ liệu có thể ảnh hưởng đến độ tin cậy.",
                        "Cần bổ sung thông tin tài chính cập nhật.",
                    ],
                    "cited_source_ids": [],
                })
            assessment = "这是中文分析。" if self.calls == 1 else "Dữ liệu cho thấy cần TRÁNH rủi ro drawdown hiện tại."
            return output_model.model_validate({
                "assessment": assessment,
                "supporting_observations": ["Dữ liệu giá đã được cung cấp.", "Rủi ro đã được đo lường.", "Kết luận định lượng được giữ nguyên."],
                "counterarguments": ["Dự báo còn bất định.", "Dữ liệu có thể chưa đầy đủ."],
                "missing_information": ["Thiếu thông tin cập nhật."],
                "cited_source_ids": [],
            })

    provider = LanguageRepairProvider()
    review = AgentLLMAdvisor(provider=provider, enabled=True).review(
        AgentName.MANAGER,
        "Tổng hợp kết quả",
        {"ticker": "TCB"},
        [],
    )
    assert provider.calls == 2
    assert review is not None
    assert review.status == "GROUNDED"
    assert "Dữ liệu" in review.assessment
    assert "drawdown" not in review.assessment
    assert "giảm giá" in review.assessment


def test_specialist_llm_rejects_persistently_non_vietnamese_output() -> None:
    class ChineseOnlyProvider:
        def parse_structured(self, system, payload, output_model):
            del system, payload
            if output_model.__name__ == "_CompactVietnameseReview":
                return output_model.model_validate({
                    "lines": ["这是中文分析。"] * 8,
                    "cited_source_ids": [],
                })
            return output_model.model_validate({
                "assessment": "这是中文分析。",
                "supporting_observations": ["这是证据。", "这是证据。", "这是证据。"],
                "counterarguments": ["这是风险。", "这是风险。"],
                "missing_information": ["这是缺失信息。"],
                "cited_source_ids": [],
            })

    review = AgentLLMAdvisor(provider=ChineseOnlyProvider(), enabled=True).review(
        AgentName.MANAGER,
        "Tổng hợp kết quả",
        {"ticker": "TCB"},
        [],
    )
    assert review is not None
    assert review.status == "UNAVAILABLE"


def test_specialist_llm_retries_english_heavy_technical_output() -> None:
    class MixedLanguageProvider:
        def __init__(self) -> None:
            self.calls = 0

        def parse_structured(self, system, payload, output_model):
            del system, payload
            self.calls += 1
            if output_model.__name__ == "_CompactVietnameseReview":
                return output_model.model_validate({
                    "lines": [
                        "Tín hiệu kỹ thuật hiện nghiêng về chiều hướng tiêu cực.",
                        "Kết luận định lượng của hệ thống được giữ nguyên.",
                        "Giá đang nằm dưới các đường trung bình quan trọng.",
                        "Động lượng chưa cho thấy sự phục hồi rõ ràng.",
                        "Thanh khoản cần được theo dõi để xác nhận xu hướng.",
                        "Biến động giá có thể làm tín hiệu thay đổi nhanh.",
                        "Dự báo chỉ mang tính tham khảo và còn bất định.",
                        "Cần thêm dữ liệu của các phiên tiếp theo để xác nhận.",
                    ],
                    "cited_source_ids": [],
                })
            return output_model.model_validate({
                "assessment": "The forecast and technical indicators suggest a strong SELL signal based on recent sessions.",
                "supporting_observations": ["Dữ liệu giá đã được cung cấp."] * 3,
                "counterarguments": ["Dự báo còn bất định."] * 2,
                "missing_information": ["Thiếu thông tin cập nhật."],
                "cited_source_ids": [],
            })

    provider = MixedLanguageProvider()
    review = AgentLLMAdvisor(provider=provider, enabled=True).review(
        AgentName.TECHNICAL,
        "Diễn giải tín hiệu kỹ thuật",
        {"ticker": "FPT"},
        [],
    )
    assert provider.calls == 2
    assert review is not None
    assert review.status == "GROUNDED"
    assert "Tín hiệu kỹ thuật" in review.assessment
    assert "forecast" not in review.assessment.lower()


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


class FakeBraveDataSourceProvider:
    name = "brave_search_api"

    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, limit: int) -> list[WebSearchResult]:
        self.calls += 1
        assert "FPT" in query
        assert limit == 3
        return [WebSearchResult(
            "FPT historical market data",
            "https://example.com/market-data/fpt",
            "FPT historical price and financial data",
            provider_name=self.name,
        )]


def test_analysis_cites_provenance_and_uses_brave_for_missing_source_url() -> None:
    brave = FakeBraveDataSourceProvider()
    service = ChatService(
        analysis_service=AnalysisService(provider=MultiTickerProvider(["FPT", "VNINDEX"])),
        web_research=WebResearchService(
            brave, DisabledIssuerDiscoveryProvider(), DisabledCompanyNameResolver(),
        ),
        universe={"default": ["FPT"]},
    )
    query = ParsedQuery(
        raw_query="Phân tích kỹ thuật FPT",
        intent=QueryIntent.TECHNICAL_ANALYSIS,
        tickers=["FPT"],
        confidence=0.95,
    )

    response = service.handle(query.raw_query, ConversationState(), parsed_query=query)

    assert brave.calls == 1
    assert "#### Nguồn dữ liệu" in response.message
    assert "`test:ohlcv:FPT`" in response.message
    assert "https://example.com/market-data/fpt" in response.message
    assert response.metadata["data_source_citations"]["missing_source_url_count"] == 1
    assert response.metadata["data_source_citations"]["brave_discovered_links"][0]["search_provider"] == "brave_search_api"


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
    store = ResearchStore(Path(__file__).resolve().parents[1] / ".runtime" / "test-artifacts" / "local-first-period.sqlite3")
    store.ingest_text(
        text="FPT annual report 2025 describes revenue growth.", title="FPT annual report 2025",
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
