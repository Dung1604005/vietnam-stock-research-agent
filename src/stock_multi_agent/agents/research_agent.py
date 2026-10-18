from __future__ import annotations

from stock_multi_agent.agents.base_agent import BaseAgent
from stock_multi_agent.models.agent_contracts import AgentSpec, ResearchAgentContext
from stock_multi_agent.models.messages import AgentName, MessageType
from stock_multi_agent.models.research import EvidenceBundle
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.research_store import ResearchStore
from stock_multi_agent.services.research_freshness import research_scope, filter_citations, external_to_citation, freshness_note, today
import time


class ResearchAgent(BaseAgent):
    """Retrieve bounded, provenance-preserving context for other specialists."""

    name = AgentName.RESEARCH
    goal = "Retrieve relevant company evidence with citations without issuing a trading signal."
    spec = AgentSpec(
        role=name,
        goal=goal,
        required_context=["ticker", "investment_horizon", "user_goal"],
        allowed_tools=["hybrid_vector_fts_retrieval", "bounded_web_retrieval", 'structured_llm_review'],
        output_type="list[Citation]",
        can_use_llm=True,
        decision_constraints=[
            "Use current-period local evidence or fetched web evidence with source, publication date and period checks",
            "Never convert retrieved prose into a BUY or SELL signal",
            "Preserve document ID, chunk ID, source URL, publisher, and publication time",
        ],
    )

    def __init__(self, store: ResearchStore | None = None, web_research=None, llm_advisor=None) -> None:
        self.store = store or ResearchStore()
        self.web_research = web_research
        self._web_cache = {}
        self.llm_advisor = llm_advisor

    def run(self, state: AgentState, expanded: bool = False, follow_up_task: str | None = None,
            follow_up_metric: str | None = None, follow_up_period: str | None = None) -> dict[str, object]:
        context = ResearchAgentContext.from_state(state)
        scope = research_scope(state.user_goal, ticker=context.ticker, current=True)
        queries = self._queries(context, expanded=expanded)
        objectives = {
            'FIND_RECENT_REPORT': f'{context.ticker} báo cáo tài chính mới nhất công bố chính thức',
            'VERIFY_FINANCIAL_FACT': f'{context.ticker} báo cáo tài chính doanh thu lợi nhuận đơn vị kỳ hợp nhất',
            'CHECK_RISK_EVENT': f'{context.ticker} công bố thông tin rủi ro kiện tụng nợ kiểm toán mới nhất',
        }
        if follow_up_task is not None:
            if follow_up_task not in objectives:
                raise ValueError('Unsupported research follow-up task')
            target = f' metric {follow_up_metric}' if follow_up_metric else ''
            target += f' period {follow_up_period}' if follow_up_period else ''
            objectives[follow_up_task] += target
            queries = [objectives[follow_up_task], *queries[:1]]
        bundles: list[EvidenceBundle] = []
        traces = [self.event("PLAN", "CREATE_RETRIEVAL_PLAN", f"Prepared {len(queries)} bounded research queries")]
        for query in queries:
            bundle = self.store.search(query, ticker=context.ticker, limit=4, scope=scope)
            bundles.append(bundle)
            traces.append(self.event(
                "ACT", "CALL_TOOL", f"Retrieved {len(bundle.citations)} citation(s) for: {query}",
                "hybrid_vector_fts_retrieval",
            ))
            if len(bundle.citations) >= (4 if expanded else 3):
                break

        citations = []
        seen_chunks: set[str] = set()
        for bundle in bundles:
            for citation in bundle.citations:
                if citation.chunk_id not in seen_chunks:
                    seen_chunks.add(citation.chunk_id)
                    citations.append(citation)
        citations.sort(key=lambda item: item.retrieval_score, reverse=True)
        citations = citations[:8 if expanded else 4]
        archive = [item for bundle in bundles for item in bundle.historical_citations]
        if not archive and not citations and isinstance(self.store, ResearchStore):
            archive = self.store.period_overview(context.ticker)
        citations, excluded = filter_citations(citations, scope)
        archive.extend(excluded)
        if (not citations or follow_up_task is not None) and self.web_research is not None and state.allow_llm_planning:
            key = (context.ticker, scope, today(), follow_up_task, follow_up_metric, follow_up_period)
            cached = self._web_cache.get(key)
            try:
                if cached and cached[0] > time.monotonic():
                    citations = cached[1]
                else:
                    execution = self.web_research.research(objectives.get(follow_up_task,
                        f"{context.ticker} báo cáo tài chính mới nhất"), ticker=context.ticker)
                    fetched = [citation for item in execution.external_citations if (citation := external_to_citation(item, context.ticker))]
                    fresh, old = filter_citations(fetched, scope)
                    citations = list({item.chunk_id: item for item in [*citations, *fresh]}.values())[:8]
                    archive.extend(old)
                    self._web_cache[key] = (time.monotonic() + 600, citations)
                    if len(self._web_cache) > 100:
                        self._web_cache.pop(next(iter(self._web_cache)))
                    if isinstance(self.store, ResearchStore):
                        from stock_multi_agent.services.research_sync import ResearchSyncService
                        ResearchSyncService(self.store).enqueue_chat(execution.discovery_citations, ticker=context.ticker)
                traces.append(self.event("ACT", "FRESHNESS_WEB_FALLBACK", f"Found {len(citations)} current, dated web citations", "bounded_web_retrieval"))
            except Exception:
                traces.append(self.event("ACT", "FRESHNESS_WEB_FALLBACK_UNAVAILABLE", "No updated document could be verified"))
        note = freshness_note(citations or archive, insufficient=not citations, historical=scope.mode == "HISTORY")
        citation_ids = [citation.chunk_id for citation in citations]
        status = "COMPLETED" if citations else "PARTIAL"
        summary = (
            f"Retrieved {len(citations)} provenance-linked research citation(s) for {context.ticker}."
            if citations else f"Chưa có dữ liệu nghiên cứu cập nhật cho {context.ticker}. {note}"
        )
        review = self.llm_review('Đối chiếu bằng chứng và nêu mâu thuẫn, không đưa ra kết luận giao dịch',
            {'ticker': state.ticker, 'citations': [item.model_dump(mode='json') for item in citations]},
            citation_ids) if expanded and state.allow_llm_planning and citations else None
        messages = [
            self.message(
                AgentName.FUNDAMENTAL,
                MessageType.RESULT,
                "Research evidence ready",
                {"ticker": context.ticker, "citation_ids": citation_ids},
                confidence=max((item.retrieval_score for item in citations), default=0.0),
            ),
            self.message(
                AgentName.RISK,
                MessageType.RESULT,
                "Research evidence ready",
                {"ticker": context.ticker, "citation_ids": citation_ids},
                confidence=max((item.retrieval_score for item in citations), default=0.0),
            ),
        ] if citations else []
        return {
            "research_evidence": citations,
            "research_freshness_note": note,
            "llm_reviews": [review] if review else [],
            "messages": messages,
            "trace": traces,
            "completed_tasks": ["research_reanalysis" if expanded else "research_retrieval"],
            "agent_reports": [self.report(
                stage="research_reanalysis" if expanded else "research_retrieval",
                status=status,
                summary=summary,
                confidence=max((item.retrieval_score for item in citations), default=0.0),
                output_key="research_evidence",
                tools_used=["hybrid_vector_fts_retrieval"],
                missing_information=[] if citations else ["No matching verified local document"],
                evidence_count=len(citations),
            )],
        }

    @staticmethod
    def _queries(context: ResearchAgentContext, expanded: bool = False) -> list[str]:
        queries = [
            f"{context.ticker} kết quả kinh doanh định giá dòng tiền rủi ro",
            f"{context.ticker} triển vọng doanh nghiệp trong {context.investment_horizon}",
        ]
        if expanded:
            queries.extend([
                f"{context.ticker} annual report financial statements revenue profit debt",
                f"{context.ticker} official disclosure business outlook valuation",
                f"{context.ticker} business liquidity dilution governance risks",
            ])
        return queries
