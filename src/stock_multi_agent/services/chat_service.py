from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from stock_multi_agent.chat.query_parser import QueryParser
from stock_multi_agent.chat.response import ResponseComposer
from stock_multi_agent.chat.router import IntentRouter, RoutingPlanValidator
from stock_multi_agent.agents.manager_agent import ManagerResponseSynthesis, ManagerRouteSelection
from stock_multi_agent.models.chat import (
    AssistantResponse,
    ChatRole,
    ConversationMessage,
    ConversationState,
    ParsedQuery,
    QueryIntent,
    ResponseTable,
    ResponseType,
)
from stock_multi_agent.models.messages import AgentName
from stock_multi_agent.models.analysis import Signal
from stock_multi_agent.models.data import SourceReference
from stock_multi_agent.models.screening import ComparisonSpec, RankingDirection, ScreenSpec, ScreenStatus
from stock_multi_agent.models.state import AgentState, TraceEvent
from stock_multi_agent.providers.market_provider import MarketDataError
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.chart_service import ChartService
from stock_multi_agent.services.knowledge_service import KnowledgeService
from stock_multi_agent.services.llm_analyst import LLMAnalystService
from stock_multi_agent.services.rag_service import RAGService
from stock_multi_agent.services.research_store import ResearchStore
from stock_multi_agent.services.research_orchestrator import ResearchOrchestrator
from stock_multi_agent.services.recommendation_service import RecommendationService
from stock_multi_agent.services.screening_service import ScreeningService
from stock_multi_agent.services.universe_service import UniverseService
from stock_multi_agent.services.web_research import WebResearchError, WebResearchService, WebSearchProvider
from stock_multi_agent.tools.technical_indicators import TechnicalIndicatorToolkit
from stock_multi_agent.models.research_contracts import RequestBudget
from stock_multi_agent.services.request_telemetry import observed, request_scope
from stock_multi_agent.services.reporting_period import period_display
from stock_multi_agent.services.research_workbench import ResearchWorkbench
from stock_multi_agent.services.research_freshness import research_scope, external_to_citation, freshness_reason


class ChatService:
    def __init__(
        self,
        analysis_service: AnalysisService | None = None,
        query_parser: QueryParser | None = None,
        intent_router: IntentRouter | None = None,
        chart_service: ChartService | None = None,
        research_store: ResearchStore | None = None,
        rag_service: RAGService | None = None,
        web_research: WebResearchService | None = None,
        research_orchestrator: ResearchOrchestrator | None = None,
        knowledge_service: KnowledgeService | None = None,
        llm_analyst: LLMAnalystService | None = None,
        recommendation_service: RecommendationService | None = None,
        universe: dict[str, list[str]] | None = None,
    ) -> None:
        self.analysis = analysis_service or AnalysisService()
        self.parser = query_parser or QueryParser()
        self.router = intent_router or IntentRouter()
        self.charts = chart_service or ChartService()
        self.research = research_store or ResearchStore()
        self.rag = rag_service or RAGService(self.research)
        self.web_research = web_research or WebResearchService()
        self._data_source_link_cache: dict[str, list[dict[str, str]]] = {}
        self.research_orchestrator = research_orchestrator or ResearchOrchestrator(discovery=self.web_research)
        self.knowledge = knowledge_service or KnowledgeService()
        self.llm_analyst = llm_analyst or LLMAnalystService()
        self.indicators = TechnicalIndicatorToolkit()
        self.universes = UniverseService(universe or self._load_universe())
        self.universe = self.universes.groups()
        self.screening = ScreeningService(self.analysis, self.universes)
        self.recommendations = recommendation_service or RecommendationService(self.screening, self.analysis)
        self.workbench = ResearchWorkbench(self.research.database_path)

    def handle(
        self, user_message: str, conversation: ConversationState, parsed_query: ParsedQuery | None = None,
        *, execution_mode=None, progress_sink=None, offline_bundle=None,
    ) -> AssistantResponse:
        from stock_multi_agent.services.execution_profiles import execution_scope, current_profile, resolve_profile
        from stock_multi_agent.services.progress_events import progress_scope, current_progress_sink, ProgressSink
        selected = resolve_profile(execution_mode if execution_mode is not None else current_profile())
        sink = progress_sink or current_progress_sink() or ProgressSink()
        with execution_scope(selected), progress_scope(sink):
            if selected.mode.value == 'OFFLINE' and not getattr(self, '_offline_runtime', False):
                try:
                    local_service = self._offline_service(offline_bundle)
                except (ValueError, OSError) as exc:
                    response = AssistantResponse(message='Không mở được replay bundle đã xác minh. Kiểm tra file JSON và checksum.',
                        metadata={'execution_profile': selected.model_dump(mode='json'), 'offline_error_type': type(exc).__name__})
                    conversation.messages.append(ConversationMessage(role=ChatRole.USER, content=user_message))
                    conversation.messages.append(ConversationMessage(role=ChatRole.ASSISTANT, content=response.message, response=response))
                    return response
                response = local_service.handle(user_message, conversation, parsed_query,
                    execution_mode=selected, progress_sink=sink)
                response.metadata['replay_bundle'] = getattr(local_service, '_replay_metadata', {})
                return response
            return self._handle_with_budget(user_message, conversation, parsed_query, selected, sink)

    def _offline_service(self, bundle=None):
        """A request-local service: never replace a shared live provider or environment."""
        from stock_multi_agent.providers.offline_provider import OfflineMarketDataProvider
        from stock_multi_agent.services.agent_llm import AgentLLMAdvisor
        from stock_multi_agent.services.price_forecast import PriceForecastService
        from stock_multi_agent.services.embeddings import LocalHashEmbeddingProvider
        class MissingReplayProvider:
            def get_market_data(self, ticker, include_financials=True):
                raise MarketDataError(f'Chưa chọn replay bundle có dữ liệu cho {ticker}; Offline không gọi Internet.')
        provider = OfflineMarketDataProvider(bundle) if bundle else MissingReplayProvider()
        store = ResearchStore(self.research.database_path, embedding_provider=LocalHashEmbeddingProvider())
        analysis = AnalysisService(provider=provider, llm_advisor=AgentLLMAdvisor(enabled=False),
            price_forecast_service=PriceForecastService(), research_store=store, manager_routing_enabled=False)
        service = ChatService(analysis_service=analysis, query_parser=QueryParser(use_configured_provider=False),
            research_store=store, rag_service=RAGService(store, enabled=False), knowledge_service=self.knowledge,
            universe={'default': list(provider.bundle.datasets) if bundle else []})
        service._offline_runtime = True
        service._replay_metadata = {'origin': 'SNAPSHOT_REPLAY',
            'captured_at': provider.bundle.captured_at.isoformat() if bundle else None,
            'tickers': list(provider.bundle.datasets) if bundle else []}
        return service

    def _handle_with_budget(self, user_message, conversation, parsed_query, selected, sink):
        from stock_multi_agent.services.execution_profiles import constrain_budget
        budget = RequestBudget(max_llm_calls=3 if parsed_query and parsed_query.intent in {
            QueryIntent.RESEARCH, QueryIntent.DOCUMENT_QA} else 2)
        with request_scope(constrain_budget(budget, selected)) as metrics:
            query = parsed_query or self.parser.parse(user_message, conversation)
            if query.intent in {QueryIntent.STOCK_ANALYSIS, QueryIntent.FUNDAMENTAL_ANALYSIS, QueryIntent.RISK_ANALYSIS} or (query.intent == QueryIntent.EXPLANATION and query.tickers):
                # Classroom profile: Technical planner + Fundamental + Risk,
                # plus optional Manager synthesis, within one explicit cap.
                def bounded_setting(key, default, minimum, maximum):
                    try:
                        return max(minimum, min(maximum, int(os.getenv(key, str(default)))))
                    except ValueError:
                        return default
                metrics.budget.max_llm_calls = bounded_setting('STOCK_AGENT_LLM_ANALYSIS_MAX_CALLS', 4, 2, 6)
                metrics.budget.max_output_tokens = 12000
                metrics.budget.deadline_seconds = bounded_setting('STOCK_AGENT_ANALYSIS_DEADLINE_SECONDS', 180, 30, 300)
            if query.intent in {QueryIntent.RESEARCH, QueryIntent.DOCUMENT_QA, QueryIntent.THESIS_REVIEW}:
                metrics.budget.max_llm_calls = 3
            if query.intent in {QueryIntent.STOCK_DISCOVERY, QueryIntent.STOCK_RECOMMENDATION}:
                metrics.budget.max_tool_calls = 250
            if query.intent in {QueryIntent.THEORY, QueryIntent.CURRENT_RULE, QueryIntent.HELP,
                    QueryIntent.FORECAST, QueryIntent.INDICATOR_QUERY, QueryIntent.REPORT_EXPORT}:
                metrics.budget.max_llm_calls = 1
            metrics.budget = constrain_budget(metrics.budget, selected)
            response = self._handle(user_message, conversation, query)
            metrics.source_count = len(response.sources)
            response.metadata["request_metrics"] = metrics.snapshot()
            response.metadata['execution_profile'] = selected.model_dump(mode='json')
            response.metadata['progress_events'] = sink.snapshot()
            # A local KB/help request must not touch the analysis/Manager service,
            # even just for optional audit metadata. This is not a model probe.
            if query.intent not in {QueryIntent.THEORY, QueryIntent.CURRENT_RULE, QueryIntent.HELP}:
                gateway = self.analysis.manager_agent.llm_provider
                if hasattr(gateway, 'health'):
                    health = gateway.health()
                    response.metadata['model_identity'] = {key: health.get(key) for key in
                        ('model', 'model_revision', 'adapter_identity', 'adapter_status', 'prompt_version')}
            if selected.mode.value == 'OFFLINE':
                response.message = '**Chế độ Offline — dữ liệu đã lưu, không phải giá hiện tại. Không gọi LLM/Internet.**\n\n' + response.message
            return response

    @observed("chat_pipeline")
    def _handle(
        self,
        user_message: str,
        conversation: ConversationState,
        parsed_query: ParsedQuery | None = None,
    ) -> AssistantResponse:
        query = parsed_query or self.parser.parse(user_message, conversation)
        local_only_intents = {
            QueryIntent.THEORY,
            QueryIntent.CURRENT_RULE,
            QueryIntent.HELP,
            QueryIntent.EXPLANATION,
        }
        if query.intent in local_only_intents:
            manager_routing = ManagerRouteSelection(
                selected_intent=query.intent,
                confidence=query.confidence,
                rationale="Giữ route local đã xác định; không cần gửi câu hỏi lý thuyết tới LLM.",
                fallback_reason="local_only_intent",
            )
        elif query.confidence >= 0.85 and not query.secondary_intents and not query.needs_clarification:
            # A high-confidence typed recommendation intent already has all
            # information needed by the deterministic router. Reserve the
            # remote Manager LLM for ambiguous/misclassified requests; the
            # final Manager synthesis may still run once after analysis.
            manager_routing = ManagerRouteSelection(
                selected_intent=query.intent,
                confidence=query.confidence,
                rationale="Semantic parser đã xác định rõ yêu cầu chọn cổ phiếu; dùng deterministic route nhanh.",
                fallback_reason="high_confidence_typed_route",
            )
        else:
            manager_routing = self.analysis.manager_agent.choose_chat_intent(query)
        if manager_routing.accepted and manager_routing.selected_intent is not query.intent:
            query = query.model_copy(update={
                "intent": manager_routing.selected_intent,
                "confidence": max(query.confidence, manager_routing.confidence),
                "interpretation": manager_routing.rationale,
                "parser_source": f"{query.parser_source}+manager_llm",
            })
        plan = self.router.plan(query, conversation)
        validation_errors = RoutingPlanValidator.validate(plan, query)
        if validation_errors:
            response = AssistantResponse(
                message="Yêu cầu chưa thể được thực hiện an toàn. Bạn có thể diễn đạt lại yêu cầu.",
                tickers=query.tickers,
                warnings=["Routing plan validation failed: " + " ".join(validation_errors)],
            )
        elif query.needs_clarification:
            response = AssistantResponse(
                message=query.clarification_question or "Bạn có thể nói rõ yêu cầu hơn không?",
                tickers=query.tickers,
            )
        else:
            try:
                if plan.intent == QueryIntent.REPORT_EXPORT:
                    previous = next((item.response for item in reversed(conversation.messages) if item.response), None)
                    response = previous.model_copy(deep=True) if previous else AssistantResponse(message='Hãy phân tích hoặc tìm tài liệu trước, sau đó chọn tải báo cáo Markdown/HTML dưới câu trả lời.')
                    response.metadata['export_ready'] = True
                else:
                    response = self._dispatch(query, plan.intent)
                self._attach_secondary_outputs(response, query)
                self.llm_analyst.enrich(response, query)
            except (MarketDataError, ValueError) as exc:
                from stock_multi_agent.services.audit_export import sanitize_text
                response = self._friendly_error(query.tickers, sanitize_text(str(exc)))
            except Exception as exc:
                response = AssistantResponse(
                    message=(
                        "Data Agent hoặc dịch vụ phân tích đang tạm thời không phản hồi. "
                        "Bạn có thể thử lại sau hoặc kiểm tra kết nối dữ liệu."
                    ),
                    tickers=query.tickers,
                    warnings=[f"Dịch vụ không khả dụng ({type(exc).__name__}); không thay bằng dữ liệu giả."],
                )

        if response.response_type in {
            ResponseType.ANALYSIS,
            ResponseType.COMPARISON,
            ResponseType.MARKET_OVERVIEW,
            ResponseType.STOCK_LIST,
        }:
            manager_synthesis = self.analysis.manager_agent.summarize_response(query, response)
        else:
            manager_synthesis = ManagerResponseSynthesis(
                status="SKIPPED",
                fallback_reason="unsupported_response_type",
            )
        if manager_synthesis.status == "GROUNDED":
            section = ["\n\n---\n\n### Tổng hợp từ Manager LLM"]
            section.extend(f"- {line}" for line in manager_synthesis.lines)
            if manager_synthesis.cited_source_ids:
                section.append(
                    "\nNguồn được tham chiếu: "
                    + ", ".join(f"`{source_id}`" for source_id in manager_synthesis.cited_source_ids)
                )
            response.message += "\n".join(section)

        self._attach_data_source_citations(response)

        if plan.steps:
            response.metadata["routing_plan"] = [step.value for step in plan.steps]
            response.metadata["parsed_intent"] = query.intent.value
            response.metadata["task_plan"] = [task.model_dump(mode="json") for task in plan.tasks]
            response.metadata["task_plan_budget_units"] = plan.total_budget_units
        response.metadata["semantic_understanding"] = {
            "input": query.raw_query,
            "normalized": query.normalized_query,
            "intent": query.intent.value,
            "secondary_intents": [intent.value for intent in query.secondary_intents],
            "tickers": query.tickers,
            "period": query.chart_period,
            "horizon": query.investment_horizon,
            "as_of": query.as_of.isoformat() if query.as_of else None,
            "published_after": query.published_after.isoformat() if query.published_after else None,
            "exchange": query.exchange,
            "risk_preference": query.risk_preference,
            "filters": query.filters,
            "comparison_dimensions": query.comparison_dimensions,
            "indicators": query.requested_indicators,
            "requested_result_count": query.requested_result_count,
            "requested_actions": query.requested_actions,
            "confidence": query.confidence,
            "resolved_context": query.resolved_references,
            "ambiguities": query.ambiguities,
            "interpretation": query.interpretation,
            "parser_source": query.parser_source,
            "parser_latency_ms": query.parser_latency_ms,
            "llm_attempted": query.llm_attempted,
            "llm_latency_ms": query.llm_latency_ms,
            "fallback_reason": query.fallback_reason,
            "prompt_version": query.prompt_version,
            "example_bank_version": query.example_bank_version,
        }
        response.metadata["manager_routing"] = manager_routing.model_dump(mode="json")
        response.metadata["manager_synthesis"] = manager_synthesis.model_dump(mode="json")
        self._update_conversation(conversation, user_message, query, plan.intent, response)
        return response

    def _attach_data_source_citations(self, response: AssistantResponse) -> None:
        """Render exact provenance and use Brave only to discover missing links.

        A discovered page is deliberately labelled as a fallback link instead
        of being written into ``SourceReference.source_url``. This preserves
        the distinction between the endpoint that produced a datum and a page
        found later by search.
        """
        if response.response_type not in {ResponseType.ANALYSIS, ResponseType.COMPARISON}:
            return
        if not response.source_details:
            return

        direct_lines: list[str] = []
        missing_refs = []
        for ref in response.source_details:
            as_of = ref.as_of.isoformat() if ref.as_of else "không được nguồn cung cấp"
            fetched = ref.fetched_at.isoformat()
            label = f"{ref.provider} · {ref.data_kind.value} · `{ref.source_id}`"
            if ref.source_url:
                direct_lines.append(
                    f"- [{label}]({ref.source_url}) — dữ liệu đến `{as_of}`, truy xuất `{fetched}`"
                )
            else:
                direct_lines.append(
                    f"- {label} — dữ liệu đến `{as_of}`, truy xuất `{fetched}`; nguồn không trả URL trực tiếp"
                )
                missing_refs.append(ref)

        fallback_links = self._discover_missing_data_source_links(response, missing_refs)
        response.metadata["data_source_citations"] = {
            "exact_source_count": sum(1 for ref in response.source_details if ref.source_url),
            "missing_source_url_count": len(missing_refs),
            "brave_discovered_links": fallback_links,
        }

        sections = ["\n\n#### Nguồn dữ liệu", *direct_lines]
        if fallback_links:
            sections.append(
                "\n**Liên kết tham khảo tìm bằng Brave** "
                "(dùng để kiểm tra thêm, không được coi là endpoint đã sinh ra dữ liệu):"
            )
            sections.extend(
                f"- [{item['title']}]({item['source_url']}) — `{item['search_provider']}`, "
                f"tìm lúc `{item['retrieved_at']}`"
                for item in fallback_links
            )
        elif missing_refs:
            sections.append(
                "\nKhông tìm được liên kết Brave đủ điều kiện; hệ thống giữ nguyên source_id và timestamp, không tự tạo URL."
            )

        research_citations = response.metadata.get("research_citations", [])
        if research_citations:
            sections.append("\n**Tài liệu nghiên cứu được dùng:**")
            for citation in research_citations[:5]:
                sections.append(
                    f"- [{citation['title']}]({citation['source_url']}) — "
                    f"{citation['publisher']} · công bố `{citation.get('published_at') or 'không rõ'}`"
                )
        response.message += "\n".join(sections)

    def _discover_missing_data_source_links(
        self,
        response: AssistantResponse,
        missing_refs: list[SourceReference],
    ) -> list[dict[str, str]]:
        from stock_multi_agent.services.execution_profiles import current_profile
        if current_profile().mode.value == 'OFFLINE':
            return []
        if not missing_refs or not response.tickers:
            return []
        brave = self._configured_brave_provider()
        if brave is None:
            return []

        tickers = sorted(set(response.tickers[:4]))
        kinds = sorted({ref.data_kind.value for ref in missing_refs})
        cache_key = "|".join([*tickers, *kinds])
        if cache_key in self._data_source_link_cache:
            return self._data_source_link_cache[cache_key]

        query = (
            f"{' '.join(tickers)} Vietnam stock official data source "
            f"{' '.join(kinds)} historical price financial statements"
        )
        try:
            results = brave.search(query, 3)
        except WebResearchError:
            self._data_source_link_cache[cache_key] = []
            return []

        retrieved_at = datetime.now(timezone.utc).isoformat()
        links: list[dict[str, str]] = []
        ticker_tokens = {ticker.casefold() for ticker in tickers}
        for result in results:
            parsed = urlparse(result.source_url)
            haystack = f"{result.title} {result.snippet} {result.source_url}".casefold()
            if parsed.scheme != "https" or not parsed.netloc:
                continue
            if not any(ticker in haystack for ticker in ticker_tokens):
                continue
            links.append({
                "title": result.title,
                "source_url": result.source_url,
                "search_provider": result.provider_name or brave.name,
                "retrieved_at": retrieved_at,
            })
        self._data_source_link_cache[cache_key] = links
        return links

    def _configured_brave_provider(self) -> WebSearchProvider | None:
        provider = self.web_research.provider
        if getattr(provider, "name", "") == "brave_search_api":
            return provider
        for candidate in getattr(provider, "providers", []):
            if getattr(candidate, "name", "") == "brave_search_api":
                return candidate
        return None

    def _dispatch(self, query, resolved_intent: QueryIntent) -> AssistantResponse:
        if query.as_of and resolved_intent not in {
            QueryIntent.RESEARCH, QueryIntent.DOCUMENT_QA, QueryIntent.SOURCE_REQUEST, QueryIntent.THESIS_REVIEW,
            QueryIntent.THEORY, QueryIntent.CURRENT_RULE, QueryIntent.HELP,
        }:
            return AssistantResponse(message=(
                f"Yêu cầu phân tích tại ngày {query.as_of.isoformat()} cần dữ liệu point-in-time. "
                "Luồng chat hiện hỗ trợ lọc ngày cho tài liệu; bạn có thể hỏi báo cáo được công bố trước ngày này."
            ), metadata={"status": "UNSUPPORTED_POINT_IN_TIME", "as_of": query.as_of.isoformat()})
        if resolved_intent == QueryIntent.HELP:
            return self._help_response()
        if resolved_intent == QueryIntent.GENERAL_CHAT:
            return AssistantResponse(
                message=(
                    "Mình là trợ lý Multi-Agent chuyên phân tích chứng khoán Việt Nam. "
                    "Bạn có thể yêu cầu phân tích một mã, xem RSI/MACD, vẽ biểu đồ, so sánh mã, "
                    "xem tổng quan thị trường, sàng lọc watchlist hoặc hỏi kiến thức chứng khoán có nguồn."
                )
            )
        if resolved_intent == QueryIntent.UNKNOWN:
            return AssistantResponse(message=query.clarification_question or "Bạn có thể nói rõ yêu cầu hơn không?")
        if resolved_intent == QueryIntent.EXPLANATION:
            if query.tickers:
                state = self.analysis.analyze(query.tickers[0], query.investment_horizon)
                response = ResponseComposer.full_analysis(state)
                self._attach_specialist_llm_reviews(response, state)
                return response
            return self._explain(query)
        if resolved_intent == QueryIntent.THEORY:
            return self.knowledge.answer(query.raw_query)
        if resolved_intent == QueryIntent.CURRENT_RULE:
            return self.knowledge.answer(query.raw_query, current_rule=True)
        if resolved_intent == QueryIntent.MIXED:
            return self._mixed_knowledge_analysis(query)
        if resolved_intent in {QueryIntent.EVENT_IMPACT, QueryIntent.REPORT_CHANGE, QueryIntent.SCENARIO_ANALYSIS,
                QueryIntent.CHANGE_EXPLANATION, QueryIntent.PORTFOLIO_RISK, QueryIntent.PEER_COMPARISON}:
            return self._workbench_query(query, resolved_intent)
        if resolved_intent == QueryIntent.THESIS_REVIEW:
            response = self._research(query)
            if not query.tickers:
                return response
            bundle = response.metadata.get('evidence_bundle', {})
            record = self.workbench.track_thesis(query.tickers[0], query.raw_query, bundle.get('citations', []))
            response.metadata['thesis'] = record
            response.message += f'\n\nLuận điểm được lưu để theo dõi: `{record["thesis_id"]}` · {record["status"]}. Chưa tự xác nhận luận điểm chỉ vì có tài liệu liên quan.'
            return response
        if resolved_intent in {QueryIntent.RESEARCH, QueryIntent.DOCUMENT_QA, QueryIntent.SOURCE_REQUEST}:
            return self._research(query)
        if resolved_intent == QueryIntent.FORECAST:
            if not query.tickers:
                return AssistantResponse(message="Bạn muốn dự báo mã cổ phiếu nào?")
            state = self.analysis.fetch_data(query.tickers[0], query.investment_horizon, include_forecast=True)
            return AssistantResponse(message=ResponseComposer._forecast_summary(state), tickers=query.tickers,
                sources=ResponseComposer._sources(state), source_details=ResponseComposer._source_details(state),
                data_timestamp=state.market_data.as_of if state.market_data else None,
                data_fetched_at=state.market_data.fetched_at if state.market_data else None,
                agent_trace=state.trace, metadata={"price_forecast": ResponseComposer._forecast_metadata(state)})
        if resolved_intent == QueryIntent.MARKET_OVERVIEW:
            return self._market_overview(query.investment_horizon)
        if resolved_intent == QueryIntent.STOCK_DISCOVERY:
            return self._screen(
                query.sector,
                query.investment_horizon,
                requested_count=query.requested_result_count,
                requested_actions=query.requested_actions,
                filters=query.filters, exchange=query.exchange,
            )
        if resolved_intent == QueryIntent.STOCK_RECOMMENDATION:
            if query.filters or query.exchange:
                return self._screen(query.sector, query.investment_horizon, query.requested_result_count or 6,
                    query.requested_actions, filters=query.filters, exchange=query.exchange)
            return self._recommend(
                query.sector,
                query.investment_horizon,
                requested_count=query.requested_result_count or 6,
            )
        if resolved_intent == QueryIntent.STOCK_COMPARISON:
            if len(query.tickers) < 2:
                raise ValueError("Cần ít nhất hai mã để so sánh.")
            if len(query.tickers) > 5:
                return AssistantResponse(message='Mỗi lần so sánh tối đa 5 mã; hãy chọn 2–5 mã để giữ ngân sách truy xuất.', tickers=query.tickers)
            return self._compare(query.tickers, query.investment_horizon)
        if not query.tickers:
            raise ValueError("Tôi chưa xác định được mã cổ phiếu. Bạn có thể nêu mã, ví dụ FPT hoặc VNM.")
        ticker = query.tickers[0]
        if resolved_intent == QueryIntent.STOCK_ANALYSIS:
            state = self.analysis.analyze(ticker, query.investment_horizon)
            response = ResponseComposer.full_analysis(state)
            brief = self.workbench.capture(state)
            response.metadata['research_brief'] = brief
            events = self.workbench.event_impact(state.market_data)
            if events:
                response.tables.append(ResponseTable(title='Sự kiện và tác động có điều kiện', columns=list(events[0]), rows=events))
            response.message += f'\n\nSnapshot nghiên cứu: `{brief["snapshot_id"]}` · dữ liệu đến {brief["as_of"]}. '
            response.message += 'Chưa ghi nhận sự kiện có nguồn.' if not events else 'Tác động sự kiện chưa được định lượng.'
            self._attach_specialist_llm_reviews(response, state)
            return response
        if resolved_intent == QueryIntent.TECHNICAL_ANALYSIS:
            state = self.analysis.analyze_technical(ticker, query.investment_horizon, query.requested_indicators)
            response = ResponseComposer.technical(state)
            self._attach_specialist_llm_reviews(response, state, {"technical"})
            return response
        if resolved_intent == QueryIntent.FUNDAMENTAL_ANALYSIS:
            state = self.analysis.analyze_fundamental(ticker, query.investment_horizon)
            response = ResponseComposer.fundamental(state)
            self._attach_specialist_llm_reviews(response, state, {"fundamental"})
            return response
        if resolved_intent == QueryIntent.INDICATOR_QUERY:
            fundamental = {"PE", "PB", "ROE", "ROA", "EPS"}
            if fundamental.intersection(query.requested_indicators):
                state = self.analysis.analyze_fundamental(ticker, query.investment_horizon)
                response = ResponseComposer.fundamental(state)
                self._attach_specialist_llm_reviews(response, state, {"fundamental"})
                return response
            state = self.analysis.analyze_technical(ticker, query.investment_horizon, query.requested_indicators)
            response = ResponseComposer.technical(state)
            self._attach_specialist_llm_reviews(response, state, {"technical"})
            return response
        if resolved_intent == QueryIntent.RISK_ANALYSIS:
            state = self.analysis.analyze_risk(ticker, query.investment_horizon)
            response = ResponseComposer.risk(state)
            self._attach_specialist_llm_reviews(response, state, {"risk"})
            return response
        if resolved_intent == QueryIntent.STOCK_CHART:
            return self._chart(query)
        return self._help_response()

    def _workbench_query(self, query, intent):
        if not query.tickers:
            return AssistantResponse(message='Bạn hãy nêu mã cổ phiếu cần nghiên cứu.')
        if intent == QueryIntent.CHANGE_EXPLANATION:
            return self.workbench.explain_change(query.tickers[0], query.investment_horizon)
        if intent == QueryIntent.PEER_COMPARISON:
            response = self._compare(query.tickers[:5], query.investment_horizon)
            return response
        if intent == QueryIntent.PORTFOLIO_RISK:
            from stock_multi_agent.models.data import PortfolioPosition
            positions = [PortfolioPosition(ticker=ticker, weight=weight) for ticker, weight in query.portfolio_weights.items()]
            if not positions or len(positions) > 20:
                return AssistantResponse(message='Nhập danh mục tối đa 20 mã cùng tỷ trọng %, tổng không quá 100%.')
            datasets, failures = {}, []
            for position in positions:
                try:
                    state = self.analysis.fetch_data(position.ticker, include_forecast=False)
                    if state.market_data:
                        datasets[position.ticker] = state.market_data
                except Exception as exc:
                    failures.append(f'{position.ticker}: {type(exc).__name__}')
            report = self.workbench.portfolio(positions, datasets)
            return AssistantResponse(message='### Rủi ro danh mục nhập tay\n\n'
                f'Tỷ trọng lớn nhất: {report["largest_position"]:.1%}; HHI cổ phiếu: {report["hhi"]:.4f}; '
                f'tiền mặt chưa phân bổ: {report["cash_weight"]:.1%}.\n\n'
                + (f'Biến động lịch sử năm hóa: {report["annualized_volatility"]:.2%}.' if report['quantitative_status'] == 'ASSESSED' else 'Chưa đủ dữ liệu đồng nhất để tính biến động lịch sử.'),
                tickers=query.tickers, metadata={'portfolio_risk': report}, warnings=report['limitations'] + failures,
                source_details=self._unique_sources([s for data in datasets.values() for s in data.source_refs]))
        state = self.analysis.fetch_data(query.tickers[0], query.investment_horizon,
            include_financials=intent in {QueryIntent.REPORT_CHANGE, QueryIntent.EVENT_IMPACT}, include_forecast=False)
        data = state.market_data
        if data is None:
            raise ValueError('Không có dữ liệu có nguồn cho công cụ nghiên cứu.')
        if intent == QueryIntent.SCENARIO_ANALYSIS:
            rows = self.workbench.scenarios(data, query.scenario_shocks or (-.05, 0, .05))
            title = 'Kịch bản giả định — không phải dự báo'
        elif intent == QueryIntent.REPORT_CHANGE:
            rows = self.workbench.report_changes(data)
            title = 'Đối chiếu hai kỳ báo cáo gần nhất trong dữ liệu đã lấy'
        else:
            rows = self.workbench.event_impact(data)
            title = 'Sự kiện và tác động có điều kiện'
        response = AssistantResponse(message=f'### {title} · {state.ticker}\n\n'
            + ('Mọi hàng đều có source ID và công thức/điều kiện áp dụng.' if rows else 'Chưa đủ dữ liệu có nguồn phù hợp; không tự điền dữ kiện còn thiếu.'),
            tickers=[state.ticker], source_details=data.source_refs, sources=[data.source], data_timestamp=data.as_of,
            metadata={'tool': f'ResearchWorkbench.{intent.value.lower()}'}, warnings=list(data.warnings),
            tables=[ResponseTable(title=title, columns=list(rows[0]), rows=rows)] if rows else [])
        if intent == QueryIntent.REPORT_CHANGE:
            response.warnings.append('Đối chiếu theo metric/unit/frequency. Chưa xác minh cùng cơ sở kế toán hoặc số quý riêng lẻ/lũy kế; không tự gắn nhãn YoY/QoQ.')
        if not rows and intent in {QueryIntent.EVENT_IMPACT, QueryIntent.REPORT_CHANGE}:
            research = self._research(query)
            response.tables.extend(research.tables)
            response.sources.extend(research.sources)
            response.metadata['evidence_bundle'] = research.metadata.get('evidence_bundle', {})
            response.message += '\n\nĐã tìm thêm tài liệu; các trích đoạn chưa được tự chuyển thành số liệu so sánh.'
        return response

    @staticmethod
    def _attach_specialist_llm_reviews(
        response: AssistantResponse,
        state: AgentState,
        roles: set[str] | None = None,
    ) -> None:
        if state.research_freshness_note:
            response.message += "\n\n" + state.research_freshness_note
            response.metadata["research_freshness_note"] = state.research_freshness_note
        response.metadata["llm_specialist_review_history"] = [
            review.model_dump(mode="json") for review in state.llm_reviews
        ]
        # Manager/UI must not reuse a pre-refresh GROUNDED review after a newer
        # invalidation or unavailable post-refresh review for the same role.
        response.metadata['llm_specialist_reviews'] = [review.model_dump(mode='json')
            for review in {item.agent: item for item in state.llm_reviews}.values()]
        response.metadata['evidence_follow_up_outcomes'] = state.evidence_follow_up_outcomes
        if state.evidence_follow_up_outcomes:
            response.tables.append(ResponseTable(title='Vòng bổ sung bằng chứng được Manager duyệt',
                columns=['Tác vụ', 'Trạng thái', 'Kiểm soát'], rows=[
                    {'Tác vụ': item['task_kind'], 'Trạng thái': item['status'], 'Kiểm soát': item['approval']}
                    for item in state.evidence_follow_up_outcomes]))
        selected_reviews = [
            review for review in state.llm_reviews
            if roles is None or review.agent.value in roles
        ]
        # Re-analysis may emit multiple reviews; show the newest result per role.
        selected_reviews = list({review.agent: review for review in selected_reviews}.values())
        reviews = [review for review in selected_reviews if review.status == "GROUNDED"]
        if not reviews:
            if selected_reviews:
                response.message += (
                    "\n\n---\n\n### Nhận định từ LLM\n"
                    "LLM không tạo được nhận định tiếng Việt hợp lệ trong giới hạn thời gian; "
                    "kết quả deterministic vẫn được giữ nguyên."
                )
            return
        sections = ["\n\n---\n\n### Nhận định từ LLM"]
        for review in reviews:
            sections.append(f"\n**{review.agent.value.title()} Agent:** {review.assessment}")
            if review.model_role == "indicator_planner":
                continue
            if review.grounding_status == 'CLAIM_ANCHORS_VERIFIED':
                sections.append('\nĐã đối chiếu các anchor nguồn của claim; chưa xác minh tính đúng của suy luận hay quan hệ nhân quả.')
                for claim in review.claims:
                    sections.append(f'\n- {claim.claim_vi} — `{claim.source_id}`; kỳ {claim.period_end or "chưa rõ"}; trích đoạn: “{claim.exact_quote}”')
            else:
                sections.append('\nReview chưa được kiểm chứng đến từng claim; chỉ dùng làm diễn giải tham khảo.')
            if review.supporting_observations:
                sections.append("\n**Luận cứ chính:**\n" + "\n".join(f"- {item}" for item in review.supporting_observations))
            if review.counterarguments:
                sections.append("\n**Phản biện và rủi ro:**\n" + "\n".join(f"- {item}" for item in review.counterarguments))
            if review.missing_information:
                sections.append("\n**Thông tin còn thiếu:**\n" + "\n".join(f"- {item}" for item in review.missing_information))
            if review.cited_source_ids:
                sections.append("\nNguồn: " + ", ".join(f"`{item}`" for item in review.cited_source_ids))
        for review in selected_reviews:
            if review.status != 'GROUNDED':
                sections.append(f'\n**{review.agent.value.title()} Agent:** chưa có nhận xét LLM hợp lệ; dùng kết quả Python và bằng chứng bên trên.')
        if all(review.model_role == "indicator_planner" for review in reviews):
            sections.append("\nLLM chỉ chọn công cụ bổ sung; Python tính chỉ báo và pipeline deterministic kiểm soát điểm số, dự báo và quyết định.")
        else:
            sections.append("\nLLM chỉ diễn giải evidence; điểm số, forecast và quyết định vẫn do pipeline deterministic kiểm soát.")
        response.message += "\n".join(sections)

    def _mixed_knowledge_analysis(self, query: ParsedQuery) -> AssistantResponse:
        if not query.tickers:
            raise ValueError("Cần mã cổ phiếu để áp dụng kiến thức vào dữ liệu thực.")
        response = self.knowledge.answer(query.raw_query)
        ticker = query.tickers[0]
        fundamental = {"PE", "PB", "ROE", "ROA", "EPS"}
        if fundamental.intersection(query.requested_indicators) or any(
            item in query.normalized_query for item in ("dinh gia", "dcf", "ev ebitda", "bien loi nhuan")
        ):
            applied = ResponseComposer.fundamental(
                self.analysis.analyze_fundamental(ticker, query.investment_horizon)
            )
            component = "fundamental"
        else:
            applied = ResponseComposer.technical(
                self.analysis.analyze_technical(ticker, query.investment_horizon, query.requested_indicators)
            )
            component = "technical"
        applied.message = f"### Áp dụng vào dữ liệu thật · {ticker}\n\n{applied.message}"
        self._merge_secondary_response(response, applied, component)
        response.tickers = [ticker]
        response.metadata["mixed_components"] = ["knowledge", component]
        response.metadata["knowledge_does_not_control_recommendation"] = True
        return response

    def _research(self, query: ParsedQuery) -> AssistantResponse:
        ticker = query.tickers[0] if query.tickers else None
        scope = research_scope(query.raw_query, ticker=ticker, as_of=query.as_of)
        rag_execution = self.rag.answer(query.raw_query, ticker=ticker, as_of=query.as_of, published_after=query.published_after)
        bundle = rag_execution.bundle
        if bundle.citations:
            rows = [{
                "Title": citation.title,
                "Publisher": citation.publisher,
                "Published": citation.published_at.date().isoformat(),
                "Dữ liệu đến kỳ": period_display(citation),
                "Source": str(citation.source_url),
                "Similarity": round(citation.retrieval_score, 3),
                "Retrieval": ", ".join(citation.retrieval_modes),
                "Excerpt": citation.quote,
            } for citation in bundle.citations]
            return AssistantResponse(
                message=(
                    rag_execution.message
                    if rag_execution.status == "GROUNDED"
                    else f"### Research evidence{f' · {ticker}' if ticker else ''}\n\n{bundle.retrieval_note}\n\n{rag_execution.message}"
                ),
                tickers=query.tickers,
                tables=[ResponseTable(title="Verified local citations", columns=list(rows[0]), rows=rows)],
                sources=[str(citation.source_url) for citation in bundle.citations],
                warnings=[rag_execution.error] if rag_execution.error else [],
                metadata={
                    "evidence_bundle": bundle.model_dump(mode="json"),
                    "retrieval_mode": "local",
                    "rag": {
                        "status": rag_execution.status,
                        "embedding_model": getattr(self.research.embedding_provider, "model_id", None),
                        "retrieval": "vector+fts5",
                        "cited_chunk_ids": rag_execution.cited_chunk_ids,
                        "audit_issues": rag_execution.audit_issues,
                        "answer_contract": rag_execution.answer_contract.model_dump(mode="json") if rag_execution.answer_contract else None,
                    },
                },
            )

        execution = self.research_orchestrator.research(query.raw_query, ticker=ticker)
        # Persist discovery metadata, then let the separate sync worker enforce fetch/storage policy.
        try:
            from stock_multi_agent.services.research_sync import ResearchSyncService
            queued_research = ResearchSyncService(self.research).enqueue_chat(execution.discovery_citations, ticker=ticker)
        except Exception:
            queued_research = []
        bundle.web_citations = execution.discovery_citations
        bundle.external_citations = execution.external_citations
        if query.as_of:
            bundle.external_citations = [item for item in bundle.external_citations
                if item.published_at is not None and item.published_at.date() <= query.as_of
                and (not query.published_after or item.published_at.date() >= query.published_after)]
            # Search snippets have no trustworthy publication date.
            bundle.web_citations = []
        if execution.failures:
            bundle.retrieval_note = f"{bundle.retrieval_note} External retrieval issues: {' | '.join(execution.failures[:2])}"

        research_metadata = {
            "plan": execution.plan.model_dump(mode="json"),
            "trace": execution.trace,
            "failures": execution.failures,
            "persistent_candidates": queued_research,
            "freshness_scope": scope.mode,
            "freshness_status": bundle.freshness_status,
        }
        history_rows = [{"Title": item.title, "Dữ liệu đến kỳ": period_display(item),
            "Published": item.published_at.date().isoformat(), "Source": str(item.source_url)} for item in bundle.historical_citations]
        history_tables = [ResponseTable(title="Tài liệu lịch sử — không dùng làm dữ liệu hiện tại", columns=list(history_rows[0]), rows=history_rows)] if history_rows else []
        missing_note = (bundle.freshness_note + " Đã tìm thêm trên Internet; chưa xác minh được tài liệu cập nhật phù hợp.\n\n") if bundle.freshness_status == "INSUFFICIENT_CURRENT_DATA" else ""
        queue_note = (
            f"\n\nĐã lưu {len(queued_research)} URL vào thư viện báo cáo. "
            "Tài liệu đủ điều kiện được đưa vào RAG ở lần đồng bộ tiếp theo; "
            "bạn có thể xem và duyệt các mục đang chờ trong Thư viện báo cáo / Crawler."
            if queued_research else ""
        )

        if bundle.external_citations:
            grounded_external = [citation for item in bundle.external_citations if (citation := external_to_citation(item, ticker)) is not None]
            external_rag = self.rag.synthesize(query.raw_query, bundle.model_copy(update={"citations": grounded_external}), ticker=ticker, scope=scope, as_of=query.as_of)
            if external_rag.bundle.citations:
                missing_note = ""
            elif scope.mode == "CURRENT":
                missing_note = external_rag.bundle.freshness_note + " Đã tìm thêm trên Internet; chưa xác minh được tài liệu cập nhật phù hợp.\n\n"
            rows = [{
                "Title": citation.title,
                "Published": citation.published_at.date().isoformat() if citation.published_at else "Not supplied by source",
                "Dữ liệu đến kỳ": citation.reporting_period.label if citation.reporting_period.status == "DETECTED" else "Chưa xác định",
                "Kiểm tra độ mới": (freshness_reason(converted, scope, query.as_of) or "Phù hợp phạm vi yêu cầu") if (converted := external_to_citation(citation, ticker)) else "Chưa xác minh ngày công bố",
                "Retrieved (UTC)": citation.retrieved_at.isoformat(),
                "Source": str(citation.canonical_url),
                "Extracted evidence": citation.quote,
            } for citation in bundle.external_citations]
            return AssistantResponse(
                message=missing_note + (
                    external_rag.message if external_rag.status == "GROUNDED" else
                    f"### Retrieved external evidence{f' · {ticker}' if ticker else ''}\n\n"
                    "These excerpts were fetched and extracted during this request. They are not automatically "
                    "saved to the verified local knowledge base."
                ) + queue_note,
                tickers=query.tickers,
                tables=[*history_tables, ResponseTable(title="Extracted external evidence", columns=list(rows[0]), rows=rows)],
                sources=[str(citation.canonical_url) for citation in bundle.external_citations],
                metadata={
                    "evidence_bundle": bundle.model_dump(mode="json"),
                    "retrieval_mode": "external_evidence",
                    "freshness_status": external_rag.bundle.freshness_status,
                    "research_execution": research_metadata,
                    "rag": {"status": external_rag.status, "cited_chunk_ids": external_rag.cited_chunk_ids,
                        "answer_contract": external_rag.answer_contract.model_dump(mode="json") if external_rag.answer_contract else None},
                },
            )

        if bundle.web_citations:
            rows = [{
                "Title": citation.title,
                "Domain": citation.source_domain,
                "Retrieved (UTC)": citation.retrieved_at.isoformat(),
                "Source": str(citation.source_url),
                "Search snippet": citation.snippet,
            } for citation in bundle.web_citations]
            return AssistantResponse(
                message=missing_note + (
                    f"### Internet research discovery{f' · {ticker}' if ticker else ''}\n\n"
                    "No matching verified offline document was found. These live search results are not ingested "
                    "or independently verified; open the source and ingest an approved primary document before "
                    "using it as evidence."
                ) + queue_note,
                tickers=query.tickers,
                tables=[*history_tables, ResponseTable(title="Live web results (not verified evidence)", columns=list(rows[0]), rows=rows)],
                sources=[str(citation.source_url) for citation in bundle.web_citations],
                metadata={
                    "evidence_bundle": bundle.model_dump(mode="json"),
                    "retrieval_mode": "web_fallback",
                    "research_execution": research_metadata,
                },
            )

        return AssistantResponse(
            message=missing_note + (
                "Chưa có tài liệu đã xác minh phù hợp trong research store và không tìm thấy kết quả Internet an toàn. "
                "Hãy ingest báo cáo/công bố được phép sử dụng kèm URL nguồn, publisher và published timestamp trước."
            ),
            tickers=query.tickers,
            tables=history_tables,
            warnings=[bundle.retrieval_note],
            metadata={
                "evidence_bundle": bundle.model_dump(mode="json"),
                "retrieval_mode": "none",
                "research_execution": research_metadata,
            },
        )

    def _chart(self, query) -> AssistantResponse:
        artifacts = []
        trace = [TraceEvent(
            agent=AgentName.MANAGER,
            phase="PLAN",
            action="SELECT_ROUTE",
            detail="Chart request needs Data Agent and chart_service only",
        )]
        sources: list[str] = []
        source_details = []
        warnings: list[str] = []
        newest: date | None = None
        newest_fetch = None
        from stock_multi_agent.services.financial_charts import FINANCIAL_TYPES
        missing = []
        for ticker in query.tickers[:4]:
            requests = self.charts.requests_from_query(query, ticker)
            state = self.analysis.fetch_data(ticker, query.investment_horizon,
                include_financials=any(request.chart_type in FINANCIAL_TYPES for request in requests), include_forecast=False)
            if state.market_data is None:
                raise ValueError("Data Agent không trả về dữ liệu giá.")
            for request in requests:
                try:
                    artifact = self.charts.build_artifact(request, state.market_data)
                except ValueError as exc:
                    missing.append(f'{ticker} · {request.chart_type.value}: {exc}')
                    continue
                artifacts.append(artifact)
                sources.append(artifact.source)
                source_details.extend(artifact.source_details)
                warnings.extend(artifact.warnings)
                newest = max(newest, artifact.data_timestamp) if newest else artifact.data_timestamp
                if artifact.data_fetched_at:
                    newest_fetch = max(newest_fetch, artifact.data_fetched_at) if newest_fetch else artifact.data_fetched_at
            trace.extend(state.trace)
            trace.append(TraceEvent(
                agent=AgentName.MANAGER,
                phase="ACT",
                action="CALL_TOOL",
                detail=f"Render {request.chart_type.value} chart for {ticker}",
                tool="chart_service",
            ))
        names = ", ".join(query.tickers[:4])
        return AssistantResponse(
            message=(
                f"### Biểu đồ {names}\n\n"
                + (f'Đã vẽ {len(artifacts)} biểu đồ từ số liệu có nguồn.' if artifacts else 'Chưa có đủ dữ liệu đúng chỉ tiêu/kỳ để vẽ; không tạo lịch sử giả.')
                + ('\n\nChưa vẽ được: ' + '; '.join(missing) if missing else '')
            ),
            response_type=ResponseType.CHART,
            tickers=query.tickers[:4], charts=artifacts, agent_trace=trace,
            sources=list(dict.fromkeys(sources)), source_details=self._unique_sources(source_details),
            warnings=list(dict.fromkeys(warnings)), data_timestamp=newest, data_fetched_at=newest_fetch,
            metadata={'unavailable_charts': missing},
        )

    def _attach_secondary_outputs(self, response: AssistantResponse, query: ParsedQuery) -> None:
        if query.as_of:
            return
        if QueryIntent.RESEARCH in query.secondary_intents:
            self._merge_secondary_response(response, self._research(query), "research")
        if QueryIntent.TECHNICAL_ANALYSIS in query.secondary_intents and query.tickers and "technical_score" not in response.metadata:
            technical = self.analysis.analyze_technical(query.tickers[0], query.investment_horizon, query.requested_indicators)
            self._merge_secondary_response(response, ResponseComposer.technical(technical), "technical")
        if QueryIntent.STOCK_CHART in query.secondary_intents and query.tickers:
            chart_response = self._chart(query)
            self._merge_secondary_response(response, chart_response, "chart")
        if QueryIntent.FUNDAMENTAL_ANALYSIS in query.secondary_intents and query.tickers and "fundamental_score" not in response.metadata:
            fundamental_response = ResponseComposer.fundamental(
                self.analysis.analyze_fundamental(query.tickers[0], query.investment_horizon)
            )
            self._merge_secondary_response(response, fundamental_response, "fundamental")
        if QueryIntent.RISK_ANALYSIS in query.secondary_intents and query.tickers and "risk_score" not in response.metadata:
            risk_response = ResponseComposer.risk(
                self.analysis.analyze_risk(query.tickers[0], query.investment_horizon)
            )
            self._merge_secondary_response(response, risk_response, "risk")

    def _merge_secondary_response(
        self,
        response: AssistantResponse,
        secondary: AssistantResponse,
        output_name: str,
    ) -> None:
        if output_name != "chart":
            response.message += f"\n\n---\n\n{secondary.message}"
        response.charts.extend(secondary.charts)
        response.tables.extend(secondary.tables)
        response.agent_trace.extend(secondary.agent_trace)
        response.sources = list(dict.fromkeys([*response.sources, *secondary.sources]))
        response.source_details = self._unique_sources([*response.source_details, *secondary.source_details])
        response.warnings = list(dict.fromkeys([*response.warnings, *secondary.warnings]))
        if secondary.data_timestamp:
            response.data_timestamp = max(response.data_timestamp, secondary.data_timestamp) if response.data_timestamp else secondary.data_timestamp
        if secondary.data_fetched_at:
            response.data_fetched_at = max(response.data_fetched_at, secondary.data_fetched_at) if response.data_fetched_at else secondary.data_fetched_at
        response.metadata.setdefault("secondary_outputs", {})[output_name] = {
            key: value
            for key, value in secondary.metadata.items()
            if key in {"fundamental_score", "fundamental_signal", "risk_score", "risk_level", "veto"}
        }

    @staticmethod
    def _explain(query: ParsedQuery) -> AssistantResponse:
        definitions = {
            "RSI": "RSI đo động lượng trên thang 0–100; vùng cao/thấp là tín hiệu tham khảo, không tự động đồng nghĩa mua hoặc bán.",
            "MACD": "MACD so sánh hai đường trung bình lũy thừa để mô tả động lượng và giao cắt xu hướng.",
            "PE": "P/E là giá thị trường chia cho lợi nhuận trên mỗi cổ phiếu; cần so với lịch sử, ngành và chất lượng lợi nhuận.",
        }
        selected = [definitions[item] for item in query.requested_indicators if item in definitions]
        if selected:
            return AssistantResponse(message="\n\n".join(selected), tickers=query.tickers)
        return AssistantResponse(
            message=(
                "Mình có thể giải thích kết quả trước đó theo dữ liệu, tín hiệu kỹ thuật, nền tảng cơ bản "
                "và cảnh báo rủi ro. Hãy mở **View agent activity** để xem route và công cụ đã dùng."
            ),
            tickers=query.tickers,
        )

    def _market_overview(self, horizon: str) -> AssistantResponse:
        state = self.analysis.fetch_data("VNINDEX", horizon, include_financials=False)
        if state.market_data is None:
            raise ValueError("Không có dữ liệu VN-Index.")
        bars = state.market_data.bars
        latest = bars[-1]
        day_change = (latest.close / bars[-2].close - 1) if len(bars) >= 2 else 0.0
        change_20 = (latest.close / bars[-21].close - 1) if len(bars) >= 21 else 0.0
        avg_volume = sum(bar.volume for bar in bars[-20:]) / min(20, len(bars))
        volume_ratio = latest.volume / avg_volume if avg_volume else 0.0
        trend = "tăng" if change_20 > 0.02 else "giảm" if change_20 < -0.02 else "đi ngang"
        warnings = ResponseComposer._warnings(state)
        trace = [
            TraceEvent(agent=AgentName.MANAGER, phase="PLAN", action="MARKET_OVERVIEW", detail="Use VN-Index price and liquidity; do not invent breadth/sector data"),
            *state.trace,
            TraceEvent(agent=AgentName.MANAGER, phase="ACT", action="SYNTHESIZE", detail="Compose market overview from available fields"),
        ]
        message = (
            "### Tổng quan thị trường Việt Nam\n\n"
            f"**VN-Index:** {latest.close:,.2f} · phiên gần nhất {day_change:+.2%}  \n"
            f"**Xu hướng 20 phiên:** {trend} ({change_20:+.2%})  \n"
            f"**Thanh khoản:** {volume_ratio:.2f} lần trung bình 20 phiên\n\n"
            "**Độ rộng thị trường:** Không có dữ liệu từ provider hiện tại.  \n"
            "**Hiệu suất ngành:** Không có dữ liệu từ provider hiện tại."
        )
        return AssistantResponse(
            message=message, response_type=ResponseType.MARKET_OVERVIEW,
            tickers=["VNINDEX"], agent_trace=trace, sources=[state.market_data.source],
            source_details=state.market_data.source_refs,
            warnings=warnings, data_timestamp=latest.date,
            data_fetched_at=state.market_data.fetched_at,
            metadata={"index": latest.close, "day_change": day_change, "change_20": change_20, "volume_ratio": volume_ratio},
        )

    def _screen(
        self,
        sector: str | None,
        horizon: str,
        requested_count: int | None = None,
        requested_actions: list[str] | None = None,
        filters: dict | None = None, exchange: str | None = None,
    ) -> AssistantResponse:
        universe = self.universes.select(sector)
        actions = list(dict.fromkeys(requested_actions or []))
        eligible_signals = [Signal.BUY, Signal.HOLD]
        minimum_score = 4.0
        ranking_direction = RankingDirection.HIGHER_SCORE_FIRST
        if filters and not actions:
            # A neutral numeric screen is not a BUY/HOLD recommendation. Do not
            # silently remove SELL/UNKNOWN signals that satisfy the user's rule.
            eligible_signals = list(Signal)
            minimum_score = 0.0
        if actions == [Signal.BUY.value]:
            eligible_signals = [Signal.BUY]
        elif actions == [Signal.SELL.value]:
            eligible_signals = [Signal.SELL]
            minimum_score = 0.0
            ranking_direction = RankingDirection.LOWER_SCORE_FIRST
        target_count = requested_count or 5
        candidate_budget = min(len(universe.tickers), 100)
        spec = ScreenSpec(
            universe_key=universe.key,
            maximum_candidates=candidate_budget,
            eligible_signals=eligible_signals,
            minimum_technical_score=minimum_score,
            max_price_age_days=3,
            ranking_direction=ranking_direction,
            candidate_selection_policy="progressive_successive_halving_v1",
            factor_policy_version="screen-v3-progressive",
            filters=filters or {}, exchange=exchange,
        )
        run = self.screening.run_progressive(spec, horizon, target_count=min(target_count, candidate_budget))
        ranked_candidates = [candidate for candidate in run.candidates if candidate.status is ScreenStatus.PASS]
        selected_set = set(run.selected_tickers)
        selected_candidates = [candidate for candidate in ranked_candidates if candidate.ticker in selected_set]
        rows = [
            {
                "Ticker": candidate.ticker,
                "Rank": str(candidate.rank) if candidate.rank is not None else "-",
                "Status": candidate.status.value,
                "Technical": candidate.technical_signal.value if candidate.technical_signal else "UNKNOWN",
                "Score": f"{candidate.technical_score:.1f}" if candidate.technical_score is not None else "N/A",
                "As of": candidate.as_of.isoformat() if candidate.as_of else "unavailable",
                "Reason": " ".join(candidate.reasons),
            }
            for candidate in selected_candidates
        ]
        metric_labels = {'rsi': 'RSI14', 'pe': 'P/E', 'pb': 'P/B', 'roe': 'ROE (%)'}
        metric_columns = []
        for metric in sorted({field.rsplit('_', 1)[0] for field in spec.filters}):
            label = metric_labels[metric]
            metric_columns.append(label)
            for row, candidate in zip(rows, selected_candidates):
                value = candidate.filter_values.get(metric)
                row[label] = value * 100 if metric == 'roe' and value is not None else value
        returned_count = len(selected_candidates)
        action_label = actions[0] if len(actions) == 1 else ('điều kiện người dùng' if filters else 'BUY/HOLD')
        warnings = list(dict.fromkeys(run.warnings))
        if requested_count is not None and returned_count < requested_count:
            warnings.append(
                f"Requested {requested_count} {action_label} candidates, but only {returned_count} passed the disclosed policy "
                f"within the evaluated universe of {len(run.candidates)} symbols."
            )
        table = ResponseTable(
            title=f"Screen results - {universe.key}",
            columns=["Ticker", "Rank", *metric_columns, "Status", "Technical", "Score", "As of", "Reason"],
            rows=rows,
            note=(
                f"Universe: {len(universe.tickers)} symbols; evaluated: {len(run.candidates)}; "
                f"requested: {requested_count if requested_count is not None else 'all'}; returned: {returned_count}; "
                f"matching {action_label}: {len(ranked_candidates)}; policy: {spec.factor_policy_version}; "
                f"price-as-of aligned: {run.price_as_of_aligned}; "
                f"candidate selection: {spec.candidate_selection_policy}; "
                "not a market-wide universe or investment recommendation. không đại diện toàn thị trường."
            ),
        )
        filter_note = ''
        if filters:
            operators = {'lt': '<', 'gt': '>', 'lte': '≤', 'gte': '≥', 'eq': '='}
            conditions = [f'{metric_labels[field.rsplit("_", 1)[0]]} {operators[field.rsplit("_", 1)[1]]} {threshold:g}' for field, threshold in filters.items()]
            filter_note = '\n\n**Điều kiện lọc:** ' + ' và '.join(conditions) + '.'
            if any(field.startswith('rsi_') for field in filters):
                filter_note += ' RSI14 được Python tính từ 14 biến động giá đóng cửa gần nhất; so sánh ngưỡng trước khi làm tròn. Đạt bộ lọc không đồng nghĩa là mã nên mua.'
        return AssistantResponse(
            message=self._screen_message(
                universe.key, len(run.candidates), requested_count, returned_count, action_label,
            ) + filter_note,
            response_type=ResponseType.STOCK_LIST,
            tickers=[candidate.ticker for candidate in selected_candidates if candidate.status is ScreenStatus.PASS],
            tables=[table], agent_trace=run.traces,
            sources=[universe.source, *list(dict.fromkeys(item.provider for item in run.source_details))],
            source_details=self._unique_sources([
                *([universe.source_reference] if universe.source_reference else []), *run.source_details,
            ]),
            warnings=warnings, data_timestamp=run.newest_as_of,
            data_fetched_at=run.newest_fetched_at,
            metadata={
                "screen_spec": spec.model_dump(mode="json"),
                "universe": universe.model_dump(mode="json"),
                "evaluated_count": len(run.candidates),
                "pass_count": run.pass_count,
                "requested_result_count": requested_count,
                "returned_result_count": returned_count,
                "requested_actions": actions,
                "matching_candidate_count": len(ranked_candidates),
                "missing_data_policy": spec.missing_data_policy.value,
                "price_as_of_aligned": run.price_as_of_aligned,
                "price_as_of_comparable": run.price_as_of_comparable,
                "refreshed_tickers": run.refreshed_tickers,
                "universe_capture": universe.captured_at.isoformat() if universe.captured_at else None,
                "progressive_rounds": [item.model_dump(mode="json") for item in run.progressive_rounds],
                "screen_exclusions": [item.model_dump(mode='json') for item in run.candidates if item.status is not ScreenStatus.PASS],
            },
        )

    def _recommend(
        self,
        sector: str | None,
        horizon: str,
        requested_count: int,
    ) -> AssistantResponse:
        universe_key = sector if sector in self.universe else "default"
        result = self.recommendations.recommend(
            horizon=horizon,
            universe_key=universe_key,
            requested_count=requested_count,
            max_quick_screen=100,
        )

        def make_table(title: str, candidates, *, reference_only: bool = False) -> ResponseTable:
            return ResponseTable(
                title=title,
                columns=[
                    "Ticker", "Rank", "Tier", "Technical", "Fundamental", "Risk",
                    "Volatility", "Decision", "As of", "Why",
                ],
                rows=[{
                    "Ticker": item.ticker,
                    "Rank": item.rank_in_bucket,
                    "Tier": item.selection_tier.value,
                    "Technical": f"{item.technical_score:.1f}/10",
                    "Fundamental": f"{item.fundamental_score:.1f}/10",
                    "Risk": f"{item.risk_score:.1f}/10 ({item.risk_level.value})",
                    "Volatility": f"{item.volatility:.1%}",
                    "Decision": item.recommendation.value,
                    "As of": item.as_of.isoformat() if item.as_of else "unavailable",
                    "Why": item.reasoning + (
                        " Gaps: " + "; ".join(item.policy_gaps) if item.policy_gaps else ""
                    ),
                } for item in candidates],
                note=(
                    "REFERENCE_ONLY: các mã này chỉ để nghiên cứu tiếp và chưa vượt toàn bộ chính sách."
                    if reference_only else
                    "Phân loại theo chính sách deterministic dual-recommendation-v2; không phải chỉ dẫn giao dịch."
                ),
            )

        candidates = [*result.high_reward, *result.safe_picks, *result.reference_only]
        newest = max((item.as_of for item in candidates if item.as_of), default=None)
        tables = []
        if result.high_reward:
            tables.append(make_table("Lợi nhuận cao – rủi ro cao", result.high_reward))
        if result.safe_picks:
            tables.append(make_table("An toàn – lợi nhuận vừa phải", result.safe_picks))
        if result.reference_only:
            tables.append(make_table(
                "Tham khảo gần đạt chuẩn", result.reference_only, reference_only=True,
            ))
        if result.returned_count == requested_count and not result.reference_only:
            count_note = f"Đã trả về đúng **{result.returned_count}** mã vượt toàn bộ chính sách."
        elif result.returned_count == requested_count:
            count_note = (
                f"Có **{result.qualified_count}** mã vượt toàn bộ chính sách và bổ sung "
                f"**{result.reference_count}** mã chỉ để tham khảo, đủ **{result.returned_count}** mã yêu cầu."
            )
        else:
            count_note = (
                f"Chỉ có **{result.returned_count}/{requested_count}** mã có đủ dữ liệu để xếp hạng; "
                "hệ thống không tạo kết quả khi phân tích sâu chưa hoàn thành."
            )
        return AssistantResponse(
            message=(
                f"### Gợi ý cổ phiếu theo hai nhóm — {result.universe_key}\n\n"
                f"Đã xét universe **{result.universe_size}** mã, có **{result.screened_count}** mã qua sàng lọc "
                f"kỹ thuật và phân tích sâu **{result.analyzed_count}** mã. {count_note}\n\n"
                f"Thời gian pipeline: quick screen **{result.quick_screen_seconds:.1f}s**, "
                f"deep analysis **{result.deep_analysis_seconds:.1f}s**.\n\n"
                "Kết quả phục vụ học tập và nghiên cứu, không phải khuyến nghị đầu tư."
            ),
            response_type=ResponseType.STOCK_LIST,
            tickers=[item.ticker for item in candidates],
            tables=tables,
            sources=list(dict.fromkeys(source_id for item in candidates for source_id in item.source_ids)),
            warnings=result.warnings,
            data_timestamp=newest,
            metadata={
                "dual_recommendation": result.model_dump(mode="json"),
                "requested_result_count": requested_count,
                "returned_result_count": result.returned_count,
                "qualified_count": result.qualified_count,
                "reference_count": result.reference_count,
                "screened_count": result.screened_count,
                "analyzed_count": result.analyzed_count,
                "progressive_rounds": [item.model_dump(mode="json") for item in result.progressive_rounds],
                "deep_analysis_limits": result.deep_analysis_limits,
                "refreshed_tickers": result.refreshed_tickers,
                "price_as_of_comparable": result.price_as_of_comparable,
                "quick_screen_seconds": result.quick_screen_seconds,
                "deep_analysis_seconds": result.deep_analysis_seconds,
                "total_recommendation_seconds": result.total_seconds,
                "policy_version": result.policy_version,
            },
        )

    @staticmethod
    def _screen_message(
        universe_key: str,
        evaluated_count: int,
        requested_count: int | None,
        returned_count: int,
        action_label: str,
    ) -> str:
        if requested_count is None:
            return (
                f"### Bounded screen - {universe_key}\n\n"
                f"Evaluated **{evaluated_count}** symbols. The table preserves all statuses so unavailable or failed "
                "symbols are not silently presented as ranked candidates."
            )
        if returned_count == requested_count:
            result_note = f"Returned exactly **{returned_count}** matching candidates."
        else:
            result_note = (
                f"Returned **{returned_count}** of the **{requested_count}** requested candidates because no additional "
                "symbols passed the disclosed screen policy."
            )
        return (
            f"### Bounded {action_label} screen - {universe_key}\n\n"
            f"Evaluated **{evaluated_count}** symbols. {result_note} This is a technical screen, not a trade instruction."
        )

    def _discover(self, sector: str | None, horizon: str) -> AssistantResponse:
        universe_key = sector if sector in self.universe else "default"
        symbols = self.universe.get(universe_key, self.universe["default"])
        rows: list[dict[str, object]] = []
        traces: list[TraceEvent] = [
            TraceEvent(agent=AgentName.MANAGER, phase="PLAN", action="SELECT_UNIVERSE", detail=f"Screen configured {universe_key} watchlist: {', '.join(symbols)}")
        ]
        sources: list[str] = []
        source_details = []
        warnings: list[str] = []
        newest: date | None = None
        newest_fetch = None
        for ticker in symbols:
            try:
                state = self.analysis.analyze_technical(ticker, horizon)
                if state.market_data is None or state.technical_result is None:
                    continue
                bars = state.market_data.bars
                change_20 = bars[-1].close / bars[-21].close - 1 if len(bars) >= 21 else 0.0
                rows.append({
                    "Ticker": ticker,
                    "Technical": state.technical_result.signal.value,
                    "Score": state.technical_result.score,
                    "20D": f"{change_20:+.1%}",
                    "Last": f"{bars[-1].close:,.2f}",
                })
                traces.extend(state.trace)
                sources.append(state.market_data.source)
                source_details.extend(state.market_data.source_refs)
                newest = max(newest, bars[-1].date) if newest else bars[-1].date
                newest_fetch = max(newest_fetch, state.market_data.fetched_at) if newest_fetch else state.market_data.fetched_at
            except (MarketDataError, ValueError) as exc:
                warnings.append(f"{ticker}: {exc}")
        rows.sort(key=lambda item: float(item["Score"]), reverse=True)
        rows = rows[:5]
        if not rows:
            raise ValueError("Không lấy được dữ liệu cho watchlist đang theo dõi.")
        table = ResponseTable(
            title="Kết quả sàng lọc kỹ thuật",
            columns=["Ticker", "Technical", "Score", "20D", "Last"],
            rows=rows,
            note=f"Chỉ sàng lọc trong tập {len(symbols)} mã cấu hình, không đại diện toàn thị trường.",
        )
        return AssistantResponse(
            message=(
                f"### Các mã đáng chú ý trong watchlist {universe_key}\n\n"
                f"Data và Technical Agent đã sàng lọc **{len(symbols)} mã**. "
                "Thứ hạng dựa trên tín hiệu kỹ thuật hiện tại, không phải dự báo lợi nhuận."
            ),
            response_type=ResponseType.STOCK_LIST,
            tickers=[str(row["Ticker"]) for row in rows], tables=[table], agent_trace=traces,
            sources=list(dict.fromkeys(sources)), source_details=self._unique_sources(source_details),
            warnings=warnings, data_timestamp=newest, data_fetched_at=newest_fetch,
        )

    def _compare(self, tickers: list[str], horizon: str) -> AssistantResponse:
        spec = ComparisonSpec(tickers=tickers)
        rows: list[dict[str, object]] = []
        compared_states = []
        traces: list[TraceEvent] = [
            TraceEvent(agent=AgentName.MANAGER, phase="PLAN", action="COMPARE", detail=f"Run comparable full analyses for {', '.join(tickers)}")
        ]
        sources: list[str] = []
        source_details = []
        warnings: list[str] = []
        newest: date | None = None
        newest_fetch = None
        for ticker in tickers:
            try:
                state = self.analysis.analyze(ticker, horizon, allow_llm_planning=False)
                if not all((state.final_decision, state.technical_result, state.fundamental_result, state.risk_result, state.market_data)):
                    continue
                rows.append({
                    "Ticker": ticker,
                    "Technical": f"{state.technical_result.signal.value} ({state.technical_result.score:.1f})",
                    "Fundamental": f"{state.fundamental_result.signal.value} ({state.fundamental_result.score:.1f})",
                    "Risk": f"{state.risk_result.risk_level.value} ({state.risk_result.risk_score:.1f})",
                    "Recommendation": state.final_decision.recommendation.value,
                    "Price as of": state.market_data.as_of.isoformat() if state.market_data.as_of else "unavailable",
                    "Financial period": self._latest_financial_period(state),
                })
                compared_states.append(state)
                traces.extend(state.trace)
                sources.append(state.market_data.source)
                source_details.extend(state.market_data.source_refs)
                warnings.extend(state.market_data.warnings)
                newest = max(newest, state.market_data.bars[-1].date) if newest else state.market_data.bars[-1].date
                newest_fetch = max(newest_fetch, state.market_data.fetched_at) if newest_fetch else state.market_data.fetched_at
            except (MarketDataError, ValueError) as exc:
                warnings.append(f"{ticker}: {exc}")
        if len(rows) < 2:
            raise ValueError("Không có đủ dữ liệu hợp lệ cho ít nhất hai mã để so sánh.")
        currencies = {
            state.market_data.instrument.currency if state.market_data.instrument else "VND"
            for state in compared_states
        }
        price_dates = {state.market_data.as_of for state in compared_states}
        financial_periods = {self._latest_financial_period(state) for state in compared_states}
        units = {state.market_data.price_unit for state in compared_states}
        unit_aligned = len(units) == 1 and 'UNKNOWN' not in units
        price_aligned = len(price_dates) == 1
        currency_aligned = len(currencies) == 1
        financial_aligned = None not in financial_periods and len(financial_periods) == 1
        if spec.require_same_currency and not currency_aligned:
            warnings.append("Currency mismatch: numerical comparison is withheld.")
        if not currency_aligned or not unit_aligned:
            warnings.append('Price units/currency are missing or incompatible: numerical score comparisons are withheld.')
            for row in rows:
                for key in ('Technical', 'Fundamental', 'Risk'):
                    row[key] = 'UNAVAILABLE (unit/currency mismatch)'
        if spec.require_same_price_as_of and not price_aligned:
            warnings.append("Price as-of dates differ: technical-score comparison is withheld.")
            for row in rows:
                row["Technical"] = "UNAVAILABLE (price dates differ)"
        if spec.require_same_financial_period and not financial_aligned:
            warnings.append("Financial reporting periods are missing or differ: fundamental-score comparison is withheld.")
            for row in rows:
                row["Fundamental"] = "UNAVAILABLE (period mismatch)"
        table = ResponseTable(
            title=f"So sánh {', '.join(tickers)}",
            columns=["Ticker", "Technical", "Fundamental", "Risk", "Recommendation", "Price as of", "Financial period"],
            rows=rows,
            note=(
                "Scores are descriptive, not a return prediction. Fields with incompatible currency, price as-of, "
                "or financial reporting periods are withheld. Policy: comparison-v1."
            ),
        )
        peers = self.workbench.peer_percentiles(compared_states)
        peer_tables = [ResponseTable(title='Percentile trong tập peer đã nhập (không phải toàn ngành)', columns=list(peers[0]), rows=peers)] if peers else []
        return AssistantResponse(
            message=(
                f"### So sánh {' và '.join(tickers)}\n\n"
                "Mỗi mã đã được chạy qua cùng quy trình Multi-Agent. Bảng dưới đây cho thấy trade-off "
                "giữa động lượng, chất lượng cơ bản và rủi ro; không có kết luận chắc chắn một mã luôn tốt hơn."
            ),
            response_type=ResponseType.COMPARISON,
            tickers=tickers, tables=[table, *peer_tables], agent_trace=traces,
            sources=list(dict.fromkeys(sources)), source_details=self._unique_sources(source_details),
            warnings=list(dict.fromkeys(warnings)), data_timestamp=newest, data_fetched_at=newest_fetch,
            metadata={
                "comparison_spec": spec.model_dump(mode="json"),
                "comparison_alignment": {
                    "currency_aligned": currency_aligned,
                    "unit_aligned": unit_aligned,
                    "price_as_of_aligned": price_aligned,
                    "financial_period_aligned": financial_aligned,
                },
            },
        )

    @staticmethod
    def _latest_financial_period(state) -> str | None:
        if state.market_data is None:
            return None
        periods = [item.period_end for item in state.market_data.financial_observations if item.period_end]
        periods.extend(item.period_end for item in state.market_data.financial_statement_observations if item.period_end)
        return max(periods).isoformat() if periods else None

    @staticmethod
    def _unique_sources(items):
        unique = {}
        for item in items:
            unique[(item.source_id, item.content_sha256, item.data_kind.value)] = item
        return list(unique.values())

    @staticmethod
    def _help_response() -> AssistantResponse:
        return AssistantResponse(
            message=(
                "### Mình có thể giúp gì?\n\n"
                "- `Phân tích FPT trong 3 tháng`\n"
                "- `RSI FPT hiện thế nào?`\n"
                "- `Cho tôi biểu đồ FPT 6 tháng`\n"
                "- `So sánh FPT và CMG`\n"
                "- `Thị trường Việt Nam hiện tại thế nào?`\n"
                "- `Có mã công nghệ nào đáng chú ý?`\n"
                "- `P/E là gì?`\n"
                "- `ATO khác ATC thế nào?`"
            )
        )

    @staticmethod
    def _friendly_error(tickers: list[str], detail: str) -> AssistantResponse:
        ticker_text = f" cho mã **{tickers[0]}**" if tickers else ""
        return AssistantResponse(
            message=(
                f"Mình chưa lấy được dữ liệu hợp lệ{ticker_text}. "
                "Bạn hãy kiểm tra lại mã cổ phiếu hoặc thử lại khi nguồn dữ liệu ổn định."
            ),
            tickers=tickers,
            warnings=[detail],
        )

    @staticmethod
    def _update_conversation(conversation, user_message, query, resolved_intent, response) -> None:
        if not conversation.messages:
            conversation.title = user_message[:42] + ("…" if len(user_message) > 42 else "")
        conversation.add_message(ConversationMessage(role=ChatRole.USER, content=user_message))
        conversation.add_message(ConversationMessage(role=ChatRole.ASSISTANT, content=response.message, response=response))
        if response.tickers:
            conversation.active_tickers = response.tickers[:5]
        conversation.last_intent = resolved_intent
        conversation.investment_horizon = query.investment_horizon
        if query.chart_period:
            conversation.last_chart_period = query.chart_period
        if response.response_type == ResponseType.STOCK_LIST:
            conversation.last_stock_list = response.tickers
        if response.tickers:
            conversation.entity_history.append(response.tickers[:5])
            conversation.entity_history = conversation.entity_history[-8:]
        conversation.last_semantic_query = {
            "intent": query.intent.value,
            "secondary_intents": [intent.value for intent in query.secondary_intents],
            "tickers": query.tickers,
            "confidence": query.confidence,
            "resolved_references": query.resolved_references,
        }
        if response.response_type in {ResponseType.ANALYSIS, ResponseType.COMPARISON}:
            conversation.last_analysis = response.metadata

    @staticmethod
    def _load_universe() -> dict[str, list[str]]:
        path = Path(__file__).resolve().parents[3] / "config" / "watchlist.json"
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
