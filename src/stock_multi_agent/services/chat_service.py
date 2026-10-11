from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from stock_multi_agent.chat.query_parser import QueryParser
from stock_multi_agent.chat.response import ResponseComposer
from stock_multi_agent.chat.router import IntentRouter, RoutingPlanValidator
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
from stock_multi_agent.models.screening import ComparisonSpec, RankingDirection, ScreenSpec, ScreenStatus
from stock_multi_agent.models.state import AgentState, TraceEvent
from stock_multi_agent.providers.market_provider import MarketDataError
from stock_multi_agent.services.analysis_service import AnalysisService
from stock_multi_agent.services.chart_service import ChartService
from stock_multi_agent.services.knowledge_service import KnowledgeService
from stock_multi_agent.services.llm_analyst import LLMAnalystService
from stock_multi_agent.services.research_store import ResearchStore
from stock_multi_agent.services.research_orchestrator import ResearchOrchestrator
from stock_multi_agent.services.screening_service import ScreeningService
from stock_multi_agent.services.universe_service import UniverseService
from stock_multi_agent.services.web_research import WebResearchError, WebResearchService
from stock_multi_agent.tools.technical_indicators import TechnicalIndicatorToolkit


class ChatService:
    def __init__(
        self,
        analysis_service: AnalysisService | None = None,
        query_parser: QueryParser | None = None,
        intent_router: IntentRouter | None = None,
        chart_service: ChartService | None = None,
        research_store: ResearchStore | None = None,
        web_research: WebResearchService | None = None,
        research_orchestrator: ResearchOrchestrator | None = None,
        knowledge_service: KnowledgeService | None = None,
        llm_analyst: LLMAnalystService | None = None,
        universe: dict[str, list[str]] | None = None,
    ) -> None:
        self.analysis = analysis_service or AnalysisService()
        self.parser = query_parser or QueryParser()
        self.router = intent_router or IntentRouter()
        self.charts = chart_service or ChartService()
        self.research = research_store or ResearchStore()
        self.web_research = web_research or WebResearchService()
        self.research_orchestrator = research_orchestrator or ResearchOrchestrator(discovery=self.web_research)
        self.knowledge = knowledge_service or KnowledgeService()
        self.llm_analyst = llm_analyst or LLMAnalystService()
        self.indicators = TechnicalIndicatorToolkit()
        self.universes = UniverseService(universe or self._load_universe())
        self.universe = self.universes.groups()
        self.screening = ScreeningService(self.analysis, self.universes)

    def handle(
        self,
        user_message: str,
        conversation: ConversationState,
        parsed_query: ParsedQuery | None = None,
    ) -> AssistantResponse:
        query = parsed_query or self.parser.parse(user_message, conversation)
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
                response = self._dispatch(query, plan.intent)
                self._attach_secondary_outputs(response, query)
                self.llm_analyst.enrich(response, query)
            except (MarketDataError, ValueError) as exc:
                response = self._friendly_error(query.tickers, str(exc))
            except Exception as exc:
                response = AssistantResponse(
                    message=(
                        "Data Agent hoặc dịch vụ phân tích đang tạm thời không phản hồi. "
                        "Bạn có thể thử lại sau hoặc kiểm tra kết nối dữ liệu."
                    ),
                    tickers=query.tickers,
                    warnings=[f"{type(exc).__name__}: {exc}"],
                )

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
        self._update_conversation(conversation, user_message, query, plan.intent, response)
        return response

    def _dispatch(self, query, resolved_intent: QueryIntent) -> AssistantResponse:
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
            return self._explain(query)
        if resolved_intent == QueryIntent.THEORY:
            return self.knowledge.answer(query.raw_query)
        if resolved_intent == QueryIntent.CURRENT_RULE:
            return self.knowledge.answer(query.raw_query, current_rule=True)
        if resolved_intent == QueryIntent.MIXED:
            return self._mixed_knowledge_analysis(query)
        if resolved_intent == QueryIntent.RESEARCH:
            return self._research(query)
        if resolved_intent == QueryIntent.MARKET_OVERVIEW:
            return self._market_overview(query.investment_horizon)
        if resolved_intent == QueryIntent.STOCK_DISCOVERY:
            return self._screen(
                query.sector,
                query.investment_horizon,
                requested_count=query.requested_result_count,
                requested_actions=query.requested_actions,
            )
        if resolved_intent == QueryIntent.STOCK_COMPARISON:
            if len(query.tickers) < 2:
                raise ValueError("Cần ít nhất hai mã để so sánh.")
            return self._compare(query.tickers[:4], query.investment_horizon)
        if not query.tickers:
            raise ValueError("Tôi chưa xác định được mã cổ phiếu. Bạn có thể nêu mã, ví dụ FPT hoặc VNM.")
        ticker = query.tickers[0]
        if resolved_intent == QueryIntent.STOCK_ANALYSIS:
            state = self.analysis.analyze(ticker, query.investment_horizon)
            response = ResponseComposer.full_analysis(state)
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

    @staticmethod
    def _attach_specialist_llm_reviews(
        response: AssistantResponse,
        state: AgentState,
        roles: set[str] | None = None,
    ) -> None:
        response.metadata["llm_specialist_reviews"] = [
            review.model_dump(mode="json") for review in state.llm_reviews
        ]
        selected_reviews = [
            review for review in state.llm_reviews
            if roles is None or review.agent.value in roles
        ]
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
            if review.supporting_observations:
                sections.append("\n**Luận cứ chính:**\n" + "\n".join(f"- {item}" for item in review.supporting_observations))
            if review.counterarguments:
                sections.append("\n**Phản biện và rủi ro:**\n" + "\n".join(f"- {item}" for item in review.counterarguments))
            if review.missing_information:
                sections.append("\n**Thông tin còn thiếu:**\n" + "\n".join(f"- {item}" for item in review.missing_information))
            if review.cited_source_ids:
                sections.append("\nNguồn: " + ", ".join(f"`{item}`" for item in review.cited_source_ids))
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
        bundle = self.research.search(query.raw_query, ticker=ticker)
        if bundle.citations:
            rows = [{
                "Title": citation.title,
                "Publisher": citation.publisher,
                "Published": citation.published_at.date().isoformat(),
                "Source": str(citation.source_url),
                "Excerpt": citation.quote,
            } for citation in bundle.citations]
            return AssistantResponse(
                message=f"### Research evidence{f' · {ticker}' if ticker else ''}\n\n{bundle.retrieval_note}",
                tickers=query.tickers,
                tables=[ResponseTable(title="Verified local citations", columns=list(rows[0]), rows=rows)],
                sources=[str(citation.source_url) for citation in bundle.citations],
                metadata={"evidence_bundle": bundle.model_dump(mode="json"), "retrieval_mode": "local"},
            )

        execution = self.research_orchestrator.research(query.raw_query, ticker=ticker)
        bundle.web_citations = execution.discovery_citations
        bundle.external_citations = execution.external_citations
        if execution.failures:
            bundle.retrieval_note = f"{bundle.retrieval_note} External retrieval issues: {' | '.join(execution.failures[:2])}"

        research_metadata = {
            "plan": execution.plan.model_dump(mode="json"),
            "trace": execution.trace,
            "failures": execution.failures,
        }

        if bundle.external_citations:
            rows = [{
                "Title": citation.title,
                "Published": citation.published_at.date().isoformat() if citation.published_at else "Not supplied by source",
                "Retrieved (UTC)": citation.retrieved_at.isoformat(),
                "Source": str(citation.canonical_url),
                "Extracted evidence": citation.quote,
            } for citation in bundle.external_citations]
            return AssistantResponse(
                message=(
                    f"### Retrieved external evidence{f' · {ticker}' if ticker else ''}\n\n"
                    "These excerpts were fetched and extracted during this request. They are not automatically "
                    "saved to the verified local knowledge base."
                ),
                tickers=query.tickers,
                tables=[ResponseTable(title="Extracted external evidence", columns=list(rows[0]), rows=rows)],
                sources=[str(citation.canonical_url) for citation in bundle.external_citations],
                metadata={
                    "evidence_bundle": bundle.model_dump(mode="json"),
                    "retrieval_mode": "external_evidence",
                    "research_execution": research_metadata,
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
                message=(
                    f"### Internet research discovery{f' · {ticker}' if ticker else ''}\n\n"
                    "No matching verified offline document was found. These live search results are not ingested "
                    "or independently verified; open the source and ingest an approved primary document before "
                    "using it as evidence."
                ),
                tickers=query.tickers,
                tables=[ResponseTable(title="Live web results (not verified evidence)", columns=list(rows[0]), rows=rows)],
                sources=[str(citation.source_url) for citation in bundle.web_citations],
                metadata={
                    "evidence_bundle": bundle.model_dump(mode="json"),
                    "retrieval_mode": "web_fallback",
                    "research_execution": research_metadata,
                },
            )

        return AssistantResponse(
            message=(
                "Chưa có tài liệu đã xác minh phù hợp trong research store và không tìm thấy kết quả Internet an toàn. "
                "Hãy ingest báo cáo/công bố được phép sử dụng kèm URL nguồn, publisher và published timestamp trước."
            ),
            tickers=query.tickers,
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
        for ticker in query.tickers[:4]:
            request = self.charts.request_from_query(query, ticker)
            state = self.analysis.fetch_data(request.ticker, query.investment_horizon, include_financials=False)
            if state.market_data is None:
                raise ValueError("Data Agent không trả về dữ liệu giá.")
            artifact = self.charts.build_artifact(request, state.market_data)
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
                f"Đã vẽ **{artifacts[0].request.chart_type.value}** cho giai đoạn "
                f"**{artifacts[0].request.period}**."
            ),
            response_type=ResponseType.CHART,
            tickers=query.tickers[:4], charts=artifacts, agent_trace=trace,
            sources=list(dict.fromkeys(sources)), source_details=self._unique_sources(source_details),
            warnings=list(dict.fromkeys(warnings)), data_timestamp=newest, data_fetched_at=newest_fetch,
        )

    def _attach_secondary_outputs(self, response: AssistantResponse, query: ParsedQuery) -> None:
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
    ) -> AssistantResponse:
        universe = self.universes.select(sector)
        actions = list(dict.fromkeys(requested_actions or []))
        eligible_signals = [Signal.BUY, Signal.HOLD]
        minimum_score = 4.0
        ranking_direction = RankingDirection.HIGHER_SCORE_FIRST
        if actions == [Signal.BUY.value]:
            eligible_signals = [Signal.BUY]
        elif actions == [Signal.SELL.value]:
            eligible_signals = [Signal.SELL]
            minimum_score = 0.0
            ranking_direction = RankingDirection.LOWER_SCORE_FIRST
        # Guest vnstock access is rate-limited. Evaluate only the requested
        # cardinality (five for an open-ended request), never a hidden 3x pool.
        candidate_budget = min(len(universe.tickers), min(10, requested_count or 5))
        spec = ScreenSpec(
            universe_key=universe.key,
            maximum_candidates=candidate_budget,
            eligible_signals=eligible_signals,
            minimum_technical_score=minimum_score,
            ranking_direction=ranking_direction,
            candidate_selection_policy=(
                "ticker_ascending_prefix_v1" if universe.source_reference else "configured_order_v1"
            ),
        )
        run = self.screening.run(spec, horizon)
        ranked_candidates = [candidate for candidate in run.candidates if candidate.status is ScreenStatus.PASS]
        selected_candidates = ranked_candidates[:requested_count] if requested_count else list(run.candidates)
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
        returned_count = len(selected_candidates)
        action_label = actions[0] if len(actions) == 1 else "BUY/HOLD"
        warnings = list(dict.fromkeys(run.warnings))
        if requested_count is not None and returned_count < requested_count:
            warnings.append(
                f"Requested {requested_count} {action_label} candidates, but only {returned_count} passed the disclosed policy "
                f"within the evaluated universe of {len(run.candidates)} symbols."
            )
        table = ResponseTable(
            title=f"Screen results - {universe.key}",
            columns=["Ticker", "Rank", "Status", "Technical", "Score", "As of", "Reason"],
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
        return AssistantResponse(
            message=self._screen_message(
                universe.key, len(run.candidates), requested_count, returned_count, action_label,
            ),
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
                "universe_capture": universe.captured_at.isoformat() if universe.captured_at else None,
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
                state = self.analysis.analyze(ticker, horizon)
                if not all((state.final_decision, state.technical_result, state.fundamental_result, state.risk_result, state.market_data)):
                    continue
                rows.append({
                    "Ticker": ticker,
                    "Technical": f"{state.technical_result.signal.value} ({state.technical_result.score:.1f})",
                    "Fundamental": f"{state.fundamental_result.signal.value} ({state.fundamental_result.score:.1f})",
                    "Risk": f"{state.risk_result.risk_level.value} ({state.risk_result.risk_score:.1f})",
                    "Recommendation": state.final_decision.recommendation.value,
                    "Confidence": f"{state.final_decision.confidence:.0%}",
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
        price_aligned = len(price_dates) == 1
        currency_aligned = len(currencies) == 1
        financial_aligned = None not in financial_periods and len(financial_periods) == 1
        if spec.require_same_currency and not currency_aligned:
            warnings.append("Currency mismatch: numerical comparison is withheld.")
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
            columns=["Ticker", "Technical", "Fundamental", "Risk", "Recommendation", "Confidence", "Price as of", "Financial period"],
            rows=rows,
            note=(
                "Scores are descriptive, not a return prediction. Fields with incompatible currency, price as-of, "
                "or financial reporting periods are withheld. Policy: comparison-v1."
            ),
        )
        return AssistantResponse(
            message=(
                f"### So sánh {' và '.join(tickers)}\n\n"
                "Mỗi mã đã được chạy qua cùng quy trình Multi-Agent. Bảng dưới đây cho thấy trade-off "
                "giữa động lượng, chất lượng cơ bản và rủi ro; không có kết luận chắc chắn một mã luôn tốt hơn."
            ),
            response_type=ResponseType.COMPARISON,
            tickers=tickers, tables=[table], agent_trace=traces,
            sources=list(dict.fromkeys(sources)), source_details=self._unique_sources(source_details),
            warnings=list(dict.fromkeys(warnings)), data_timestamp=newest, data_fetched_at=newest_fetch,
            metadata={
                "comparison_spec": spec.model_dump(mode="json"),
                "comparison_alignment": {
                    "currency_aligned": currency_aligned,
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
            conversation.active_tickers = response.tickers[:4]
        conversation.last_intent = resolved_intent
        conversation.investment_horizon = query.investment_horizon
        if query.chart_period:
            conversation.last_chart_period = query.chart_period
        if response.response_type == ResponseType.STOCK_LIST:
            conversation.last_stock_list = response.tickers
        if response.tickers:
            conversation.entity_history.append(response.tickers[:4])
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
