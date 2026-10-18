"""Bounded breadth-first discovery with resumable URL frontier."""
from __future__ import annotations

import hashlib
from urllib.parse import unquote, urlsplit

from stock_multi_agent.models.research_crawler import DocumentCandidate
from stock_multi_agent.services.document_metadata import plain, publication_date
from stock_multi_agent.services.source_adapters.html_listing import HtmlListingAdapter
from stock_multi_agent.services.source_adapters.rss import RssAdapter
from stock_multi_agent.services.source_adapters.sitemap import SitemapAdapter
from stock_multi_agent.services.source_adapters.json_listing import JsonListingAdapter
from stock_multi_agent.services.source_adapters.brave_discovery import BraveDiscoveryAdapter
from stock_multi_agent.services.source_adapters.fpt import FptListingAdapter
from stock_multi_agent.services.source_registry import canonical_url
from stock_multi_agent.services.content_fetcher import ContentFetchError
from stock_multi_agent.services.research_crawl_transport import CrawlPolicyError
from stock_multi_agent.services.reporting_period import infer_reporting_period
from stock_multi_agent.services.research_freshness import classify_document_type


def is_navigation(source, url):
    parsed = urlsplit(url)
    path = parsed.path.rstrip("/")
    collection = path.rsplit("/", 1)[-1] in {"bao-cao", "bao-cao-tai-chinh", "bao-cao-thuong-nien", "cong-bo-thong-tin",
                                               "thong-tin-cong-bo", "reports", "financial-reports"}
    return collection or any(path == urlsplit(str(item.url)).path.rstrip("/") for item in source.entrypoints)


class ResearchDiscovery:
    adapters = {"listing": HtmlListingAdapter(), "rss": RssAdapter(), "sitemap": SitemapAdapter(), "json": JsonListingAdapter()}

    @staticmethod
    def candidate(source, link, parent):
        timestamp = publication_date(link.published, source.timezone)
        kind = classify_document_type(link.title)
        period = infer_reporting_period(link.title, document_type=kind)
        return DocumentCandidate(source_id=source.source_id, source_url=canonical_url(link.url), parent_url=parent,
            title=link.title[:500], publisher=source.publisher, tickers=source.ticker_map,
            document_type=kind, reporting_period=period, period_end=period.end.isoformat() if period.end else None,
            published_at=timestamp, date_evidence=link.date_evidence if timestamp else "", storage_mode=source.storage_mode)

    def discover(self, source, transport, repository, state, *, force=False, max_fetches=15):
        roots = [{"url": str(item.url), "kind": item.kind, "depth": 0} for item in source.entrypoints]
        # Check fresh landing pages first; historical backfill must not delay new reports.
        frontier = roots + state.get("frontier", [])
        def unique_frontier(values, seen):
            unique = {}
            for value in values:
                key = canonical_url(value["url"])
                if key not in seen and value["depth"] <= source.max_depth:
                    unique.setdefault(key, value)
            return list(unique.values())[:1000]
        frontier = unique_frontier(frontier, set())
        headers = state.setdefault("listing_headers", {})
        state["discovery_errors"] = []
        state["last_discovered_links"] = 0
        seen, created = set(), 0
        fetched_count = 0
        while frontier and fetched_count < max_fetches:
            entry = frontier[0]
            url = canonical_url(entry["url"])
            if url in seen or not source.accepts(url):
                frontier.pop(0)
                continue
            checkpoint = headers.get(url, {})
            try:
                response = transport.fetch(url, etag=None if force else checkpoint.get("etag"),
                                           last_modified=None if force else checkpoint.get("last_modified"))
            except (ContentFetchError, CrawlPolicyError) as exc:
                # A dead listing must not permanently poison the persisted frontier.
                frontier.pop(0)
                seen.add(url)
                fetched_count += 1
                state["discovery_errors"].append({"path": urlsplit(url).path,
                    "reason": f"HTTP_{exc.status_code}" if isinstance(exc, ContentFetchError) and exc.status_code else type(exc).__name__})
                state["frontier"] = frontier[:1000]
                repository.save_state(source.source_id, state)
                continue
            frontier.pop(0)
            seen.add(url)
            fetched_count += 1
            if response.status_code == 304:
                # A 304 listing still has downstream links to revisit for revisions/new children.
                links = checkpoint.get("children", [])
                frontier.extend(item for item in links if item["url"] not in seen)
                frontier = unique_frontier(frontier, seen)
                continue
            adapter = FptListingAdapter() if source.source_id == "fpt-ir" and entry["kind"] == "listing" else self.adapters[entry["kind"]]
            links = adapter.discover(response.body, str(response.final_url))
            children = []
            for link in links[:5000]:
                if not source.accepts(link.url):
                    continue
                target = canonical_url(link.url)
                value = plain(unquote(link.title + " " + target))
                is_document = not is_navigation(source, target) and any(plain(pattern) in value for pattern in source.document_patterns)
                if is_document:
                    candidate = self.candidate(source, link, str(response.final_url))
                    repository.upsert(candidate)
                    created += 1
                    state["last_discovered_links"] = created
                # Do not parse binary PDFs as listing pages; HTML document pages may contain report downloads.
                if entry["depth"] < source.max_depth and not urlsplit(target).path.lower().endswith(".pdf"):
                    if is_document or link.kind == "sitemap" or "page=" in target or "nha-dau-tu" in target or "quan-he-co-dong" in target:
                        children.append({"url": target, "kind": link.kind, "depth": entry["depth"] + 1})
            frontier.extend(children)
            frontier = unique_frontier(frontier, seen)
            headers[url] = {"etag": response.etag, "last_modified": response.last_modified,
                            "fingerprint": hashlib.sha256(response.body.encode()).hexdigest(), "children": children[:500]}
            # Bound checkpoint size; evict oldest listing validators.
            if len(headers) > 200:
                del headers[next(iter(headers))]
            state["frontier"] = frontier[:1000]
            repository.save_state(source.source_id, state)
        state["frontier"] = frontier[:1000]
        repository.save_state(source.source_id, state)
        if not created and source.brave_fallback:
            for link in BraveDiscoveryAdapter().discover_source(source):
                repository.upsert(self.candidate(source, link, link.url))
                created += 1
            state["last_discovered_links"] = created
            repository.save_state(source.source_id, state)
        return created
