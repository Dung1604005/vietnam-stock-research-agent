from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit

from stock_multi_agent.models.research import (
    ExternalEvidenceCitation, ExtractedDocument, FetchedDocument, ResearchPlan, WebCitation,
)
from stock_multi_agent.services.content_fetcher import ContentFetchError, SafeContentFetcher
from stock_multi_agent.services.content_extractor import HtmlContentExtractor
from stock_multi_agent.services.reporting_period import infer_reporting_period
from stock_multi_agent.services.research_planner import ResearchPlanner
from stock_multi_agent.services.web_research import WebResearchError, WebResearchService


class WebDiscovery(Protocol):
    def search(self, query: str, ticker: str | None = None, limit: int = 5) -> list[WebCitation]: ...


class ContentFetcher(Protocol):
    def fetch(self, source_url: str, *, timeout_seconds: int | None = None, max_bytes: int | None = None) -> FetchedDocument: ...


class ContentExtractor(Protocol):
    def extract(self, fetched: FetchedDocument) -> ExtractedDocument: ...


@dataclass
class ResearchExecution:
    plan: ResearchPlan
    external_citations: list[ExternalEvidenceCitation] = field(default_factory=list)
    discovery_citations: list[WebCitation] = field(default_factory=list)
    trace: list[dict[str, object]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


class ResearchOrchestrator:
    """A bounded, non-agentic research loop for externally retrieved evidence."""

    def __init__(
        self,
        planner: ResearchPlanner | None = None,
        discovery: WebDiscovery | None = None,
        fetcher: ContentFetcher | None = None,
        extractor: ContentExtractor | None = None,
    ) -> None:
        self.planner = planner or ResearchPlanner()
        self.discovery = discovery or WebResearchService()
        self.fetcher = fetcher or SafeContentFetcher()
        self.extractor = extractor or HtmlContentExtractor()

    def research(self, user_query: str, ticker: str | None = None) -> ResearchExecution:
        plan = self.planner.plan(user_query, ticker)
        execution = ResearchExecution(plan=plan)
        candidates: list[WebCitation] = []
        seen_urls: set[str] = set()

        for query in plan.queries[:plan.max_searches]:
            try:
                discovered = self.discovery.search(query.query, ticker=plan.ticker, limit=5)
            except WebResearchError as exc:
                execution.failures.append(f"search: {exc}")
                execution.trace.append({"phase": "SEARCH", "query": query.query, "outcome": "error"})
                continue
            execution.trace.append({"phase": "SEARCH", "query": query.query, "outcome": "ok", "result_count": len(discovered)})
            for citation in discovered:
                url = str(citation.source_url)
                if url not in seen_urls:
                    seen_urls.add(url)
                    candidates.append(citation)

        execution.discovery_citations = candidates
        bytes_used = 0
        seen_content: set[str] = set()
        for candidate in candidates[:plan.max_pages]:
            remaining = plan.max_total_bytes - bytes_used
            if remaining <= 0:
                execution.trace.append({"phase": "FETCH", "outcome": "budget_exhausted"})
                break
            try:
                fetched = self.fetcher.fetch(
                    str(candidate.source_url), timeout_seconds=plan.timeout_seconds, max_bytes=remaining,
                )
                bytes_used += len(fetched.body.encode("utf-8"))
                extracted = self.extractor.extract(fetched)
            except (ContentFetchError, ValueError) as exc:
                execution.failures.append(f"fetch: {candidate.source_url}: {exc}")
                execution.trace.append({"phase": "FETCH", "url": str(candidate.source_url), "outcome": "error"})
                continue
            if extracted.content_sha256 in seen_content:
                execution.trace.append({"phase": "EVALUATE", "url": str(extracted.canonical_url), "outcome": "duplicate"})
                continue
            seen_content.add(extracted.content_sha256)
            score, quote = self._select_evidence(extracted, plan.goal, plan.ticker)
            execution.trace.append({
                "phase": "EVALUATE", "url": str(extracted.canonical_url), "outcome": "selected" if quote else "irrelevant",
                "relevance_score": round(score, 3),
            })
            if not quote:
                continue
            execution.external_citations.append(ExternalEvidenceCitation(
                title=extracted.title,
                source_url=extracted.final_url,
                canonical_url=extracted.canonical_url,
                source_domain=urlsplit(str(extracted.canonical_url)).netloc.lower(),
                published_at=extracted.published_at,
                retrieved_at=extracted.retrieved_at,
                content_sha256=extracted.content_sha256,
                quote=quote,
                evidence_location="Extracted HTML text",
                relevance_score=score,
                reporting_period=infer_reporting_period(extracted.title, extracted.text),
            ))

        execution.external_citations.sort(key=lambda item: item.relevance_score, reverse=True)
        return execution

    @staticmethod
    def _select_evidence(document: ExtractedDocument, goal: str, ticker: str | None) -> tuple[float, str | None]:
        terms = ResearchOrchestrator._terms(goal)
        if ticker:
            terms.add(ticker.lower())
        if not terms:
            return 0.0, None
        title_terms = set(ResearchOrchestrator._terms(document.title))
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", document.text) if part.strip()]
        best_sentence = ""
        best_matches = 0
        for sentence in sentences:
            matches = len(terms.intersection(ResearchOrchestrator._terms(sentence)))
            if matches > best_matches:
                best_sentence, best_matches = sentence, matches
        title_matches = len(terms.intersection(title_terms))
        score = min(1.0, (best_matches + title_matches * 0.5) / max(1, len(terms)))
        if best_matches == 0:
            return score, None
        return score, best_sentence[:1400]

    @staticmethod
    def _terms(value: str) -> set[str]:
        decomposed = unicodedata.normalize("NFD", value.casefold())
        normalized = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
        stop_words = {"tim", "kiem", "cho", "toi", "ve", "cua", "va", "the", "nao", "thong", "tin", "bao", "cao"}
        return {term for term in re.findall(r"[a-z0-9]{3,}", normalized) if term not in stop_words}
