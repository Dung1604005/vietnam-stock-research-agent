from __future__ import annotations

from stock_multi_agent.agents.base_agent import BaseAgent
from stock_multi_agent.models.agent_contracts import AgentSpec, ResearchAgentContext
from stock_multi_agent.models.messages import AgentName, MessageType
from stock_multi_agent.models.research import EvidenceBundle
from stock_multi_agent.models.state import AgentState
from stock_multi_agent.services.research_store import ResearchStore


class ResearchAgent(BaseAgent):
    """Retrieve bounded, provenance-preserving context for other specialists."""

    name = AgentName.RESEARCH
    goal = "Retrieve relevant company evidence with citations without issuing a trading signal."
    spec = AgentSpec(
        role=name,
        goal=goal,
        required_context=["ticker", "investment_horizon", "user_goal"],
        allowed_tools=["hybrid_vector_fts_retrieval"],
        output_type="list[Citation]",
        can_use_llm=False,
        decision_constraints=[
            "Return only citations already present in the verified research store",
            "Never convert retrieved prose into a BUY or SELL signal",
            "Preserve document ID, chunk ID, source URL, publisher, and publication time",
        ],
    )

    def __init__(self, store: ResearchStore | None = None) -> None:
        self.store = store or ResearchStore()

    def run(self, state: AgentState) -> dict[str, object]:
        context = ResearchAgentContext.from_state(state)
        queries = self._queries(context)
        bundles: list[EvidenceBundle] = []
        traces = [self.event("PLAN", "CREATE_RETRIEVAL_PLAN", f"Prepared {len(queries)} bounded research queries")]
        for query in queries:
            bundle = self.store.search(query, ticker=context.ticker, limit=4)
            bundles.append(bundle)
            traces.append(self.event(
                "ACT", "CALL_TOOL", f"Retrieved {len(bundle.citations)} citation(s) for: {query}",
                "hybrid_vector_fts_retrieval",
            ))
            if len(bundle.citations) >= 3:
                break

        citations = []
        seen_chunks: set[str] = set()
        for bundle in bundles:
            for citation in bundle.citations:
                if citation.chunk_id not in seen_chunks:
                    seen_chunks.add(citation.chunk_id)
                    citations.append(citation)
        citations.sort(key=lambda item: item.retrieval_score, reverse=True)
        citations = citations[:4]
        citation_ids = [citation.chunk_id for citation in citations]
        status = "COMPLETED" if citations else "PARTIAL"
        summary = (
            f"Retrieved {len(citations)} provenance-linked research citation(s) for {context.ticker}."
            if citations else f"No verified local research citation matched {context.ticker}."
        )
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
            "messages": messages,
            "trace": traces,
            "completed_tasks": ["research_retrieval"],
            "agent_reports": [self.report(
                stage="research_retrieval",
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
    def _queries(context: ResearchAgentContext) -> list[str]:
        return [
            f"{context.ticker} kết quả kinh doanh định giá dòng tiền rủi ro",
            f"{context.ticker} triển vọng doanh nghiệp trong {context.investment_horizon}",
        ]
