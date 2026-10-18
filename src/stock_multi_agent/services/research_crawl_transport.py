"""Bounded HTTP transport: every redirect passes the same robots/path/rate gates."""
from __future__ import annotations

import time
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from stock_multi_agent.services.content_fetcher import ContentFetchError, SafeContentFetcher


class CrawlBudgetExceeded(RuntimeError):
    pass


class CrawlPolicyError(ValueError):
    pass


class CrawlTransport:
    USER_AGENT = "StockResearchAssistant"
    TYPES = {"text/html", "application/xhtml+xml", "application/pdf", "application/xml", "text/xml",
             "application/rss+xml", "application/atom+xml", "text/plain", "application/json"}

    def __init__(self, source, *, deadline, max_total_bytes=64_000_000):
        self.source, self.deadline = source, deadline
        self.max_total_bytes = max_total_bytes
        self.requests = self.bytes_downloaded = 0
        self.last_request = {}
        self.robots = {}

    def _pace(self, url, delay=None):
        if self.requests >= self.source.max_pages_per_run or self.bytes_downloaded >= self.max_total_bytes:
            raise CrawlBudgetExceeded("REQUEST_OR_BYTE_BUDGET")
        host = urlsplit(url).netloc
        delay = max(self.source.min_request_interval_seconds, delay or 0)
        wait = max(0, delay - (time.monotonic() - self.last_request.get(host, 0)))
        if time.monotonic() + wait >= self.deadline:
            raise CrawlBudgetExceeded("DEADLINE")
        if wait:
            time.sleep(wait)
        self.last_request[host] = time.monotonic()
        self.requests += 1

    def _robots(self, url):
        parsed = urlsplit(url)
        origin = f"https://{parsed.netloc}"
        if origin not in self.robots:
            robots_url = origin + "/robots.txt"
            def robots_guard(target):
                other = urlsplit(target)
                if other.scheme != "https" or other.netloc != parsed.netloc or other.path != "/robots.txt":
                    raise CrawlPolicyError("ROBOTS_REDIRECT_NOT_ALLOWED")
                self._pace(target)
            fetcher = SafeContentFetcher(url_guard=robots_guard, content_types=self.TYPES)
            try:
                response = fetcher.fetch(robots_url, timeout_seconds=self._timeout(), max_bytes=256_000)
                self.bytes_downloaded += response.byte_size
                lines = response.body.splitlines()
            except ContentFetchError as exc:
                if exc.status_code in {404, 410}:
                    lines = []
                elif exc.status_code is None or exc.status_code == 429 or exc.status_code >= 500:
                    raise
                else:
                    raise CrawlPolicyError("ROBOTS_UNAVAILABLE") from exc
            parser = RobotFileParser(robots_url)
            parser.parse(lines)
            self.robots[origin] = parser
        return self.robots[origin]

    def _timeout(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise CrawlBudgetExceeded("DEADLINE")
        return max(1, min(12, int(remaining)))

    def _guard(self, url):
        if not self.source.fetch_allowed or not self.source.accepts(url):
            raise CrawlPolicyError("URL_OUTSIDE_SOURCE")
        robots = self._robots(url)
        if not robots.can_fetch(self.USER_AGENT, url):
            raise CrawlPolicyError("ROBOTS_DISALLOWED")
        self._pace(url, robots.crawl_delay(self.USER_AGENT) or robots.crawl_delay("*"))

    def fetch(self, url, *, etag=None, last_modified=None):
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        fetcher = SafeContentFetcher(url_guard=self._guard, content_types=self.TYPES, request_headers=headers)
        response = fetcher.fetch(url, timeout_seconds=self._timeout(),
                                 max_bytes=min(self.source.max_document_bytes, self.max_total_bytes - self.bytes_downloaded))
        self.bytes_downloaded += response.byte_size
        return response
