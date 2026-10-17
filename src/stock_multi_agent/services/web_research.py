from __future__ import annotations

import os
import re
import ssl
import unicodedata
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, unquote, urlencode, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from stock_multi_agent.models.research import WebCitation


class WebResearchError(RuntimeError):
    """The optional live discovery service could not return a safe result."""


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    source_url: str
    snippet: str
    provider_name: str | None = None


class WebSearchProvider(Protocol):
    name: str

    def search(self, query: str, limit: int) -> list[WebSearchResult]: ...


class IssuerDiscoveryProvider(Protocol):
    name: str

    def search(self, ticker: str) -> list[WebSearchResult]: ...


class CompanyNameResolver(Protocol):
    def resolve(self, ticker: str) -> str | None: ...


class DisabledWebSearchProvider:
    name = "disabled"

    def search(self, query: str, limit: int) -> list[WebSearchResult]:
        raise WebResearchError("Live web discovery is disabled by configuration.")


class DisabledIssuerDiscoveryProvider:
    name = "disabled"

    def search(self, ticker: str) -> list[WebSearchResult]:
        return []


class DisabledCompanyNameResolver:
    def resolve(self, ticker: str) -> str | None:
        return None


class VnstockCompanyNameResolver:
    """Use the project's existing VCI source to disambiguate a ticker for web search."""

    def resolve(self, ticker: str) -> str | None:
        try:
            from vnstock.api.company import Company

            source = os.getenv("VNSTOCK_SOURCE", "VCI").upper()
            overview = Company(symbol=ticker, source=source).overview()
            if getattr(overview, "empty", True):
                return None
            value = overview.iloc[0].get("organ_name") or overview.iloc[0].get("organ_short_name")
            if not isinstance(value, str) or not value.strip():
                return None
            return re.sub(r"\s+", " ", value).strip()
        except Exception:
            return None


class HnxIssuerDiscoveryProvider:
    """Find the official HNX issuer/disclosure page for a known ticker.

    This is intentionally a discovery link, not an assertion that any linked
    report is current or suitable evidence. It prevents short ticker symbols
    such as PNG from being treated as ordinary web-search words.
    """

    name = "hnx_issuer_page"
    max_response_bytes = 1_000_000
    endpoints = (
        "https://www.hnx.vn/vi-vn/cophieu-etfs/chi-tiet-chung-khoan-uc-{ticker}.html",
        "https://cbtd.hnx.vn/vi-vn/cophieu-etfs/chi-tiet-chung-khoan-uc-{ticker}.html",
    )

    def search(self, ticker: str) -> list[WebSearchResult]:
        symbol = ticker.upper().strip()
        if not re.fullmatch(r"[A-Z0-9]{3,10}", symbol):
            return []
        for template in self.endpoints:
            source_url = template.format(ticker=symbol)
            page_text = self._fetch_text(source_url)
            if not page_text or not self._matches_ticker(page_text, symbol):
                continue
            return [WebSearchResult(
                title=f"HNX issuer profile and disclosures · {symbol}",
                source_url=source_url,
                snippet=(
                    f"Official HNX issuer profile and disclosure page for ticker {symbol}. "
                    "Open the source to select the relevant annual report, financial statement, or disclosure."
                ),
            )]
        return []

    def _fetch_text(self, source_url: str) -> str:
        request = Request(
            source_url,
            headers={"User-Agent": "StockResearchAssistant/0.1 (official issuer discovery)"},
        )
        try:
            with urlopen(request, timeout=12, context=self._ssl_context()) as response:  # noqa: S310 - fixed HNX endpoints
                payload = response.read(self.max_response_bytes + 1)
        except OSError:
            return ""
        if len(payload) > self.max_response_bytes:
            return ""
        return re.sub(r"\s+", " ", payload.decode("utf-8", errors="replace"))

    @staticmethod
    def _ssl_context() -> ssl.SSLContext:
        """Use certifi when available; certificate verification is never disabled."""
        try:
            import certifi
        except ImportError:
            return ssl.create_default_context()
        return ssl.create_default_context(cafile=certifi.where())

    @staticmethod
    def _matches_ticker(page_text: str, ticker: str) -> bool:
        normalized = unescape(re.sub(r"<[^>]+>", " ", page_text))
        return bool(re.search(rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])", normalized, flags=re.IGNORECASE))


class _DuckDuckGoResultParser(HTMLParser):
    """Small, dependency-free parser for DuckDuckGo's public HTML result page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture: str | None = None
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = attributes.get("class", "") or ""
        if tag == "a" and "result__a" in classes:
            self._flush()
            self._current = {"url": attributes.get("href", ""), "title": "", "snippet": ""}
            self._capture = "title"
            self._depth = 1
        elif self._current is not None and "result__snippet" in classes:
            self._capture = "snippet"
            self._depth = 1
        elif self._capture is not None:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._capture is not None:
            self._depth -= 1
            if self._depth <= 0:
                self._capture = None
                self._depth = 0

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._capture is not None:
            self._current[self._capture] += data

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        if self._current is not None and self._current["url"] and self._current["title"]:
            self.results.append({key: re.sub(r"\s+", " ", value).strip() for key, value in self._current.items()})
        self._current = None
        self._capture = None
        self._depth = 0


class DuckDuckGoHtmlSearchProvider:
    """Unauthenticated web discovery; it never persists or ingests result pages."""

    name = "duckduckgo_html"
    endpoint = "https://html.duckduckgo.com/html/?q="
    max_response_bytes = 1_000_000

    def search(self, query: str, limit: int) -> list[WebSearchResult]:
        request = Request(
            self.endpoint + quote_plus(query),
            headers={"User-Agent": "StockResearchAssistant/0.1 (citation-first web discovery)"},
        )
        try:
            with urlopen(request, timeout=12) as response:  # noqa: S310 - fixed search endpoint
                payload = response.read(self.max_response_bytes + 1)
        except OSError as exc:
            raise WebResearchError("Live web search is temporarily unavailable.") from exc
        if len(payload) > self.max_response_bytes:
            raise WebResearchError("Live web search response exceeded the safety limit.")

        parser = _DuckDuckGoResultParser()
        parser.feed(payload.decode("utf-8", errors="replace"))
        parser.close()
        results: list[WebSearchResult] = []
        for item in parser.results:
            source_url = self._unwrap_result_url(item["url"])
            parsed = urlparse(source_url)
            if parsed.scheme != "https" or not parsed.netloc:
                continue
            snippet = item["snippet"] or "Search result snippet was not supplied by the provider."
            results.append(WebSearchResult(
                title=unescape(item["title"]),
                source_url=source_url,
                snippet=unescape(snippet),
                provider_name=self.name,
            ))
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _unwrap_result_url(value: str) -> str:
        parsed = urlparse(unescape(value))
        if "duckduckgo.com" in parsed.netloc:
            redirect_target = parse_qs(parsed.query).get("uddg", [""])[0]
            if redirect_target:
                return unquote(redirect_target)
        return value


class BingRssSearchProvider:
    """Structured public RSS search results with no key or third-party package."""

    name = "bing_rss"
    endpoint = "https://www.bing.com/search?format=rss&q="
    max_response_bytes = 1_000_000

    def search(self, query: str, limit: int) -> list[WebSearchResult]:
        request = Request(
            self.endpoint + quote_plus(query),
            headers={"User-Agent": "StockResearchAssistant/0.1 (citation-first web discovery)"},
        )
        try:
            with urlopen(request, timeout=12) as response:  # noqa: S310 - fixed search endpoint
                payload = response.read(self.max_response_bytes + 1)
        except OSError as exc:
            raise WebResearchError("Live web search is temporarily unavailable.") from exc
        if len(payload) > self.max_response_bytes:
            raise WebResearchError("Live web search response exceeded the safety limit.")
        try:
            root = ElementTree.fromstring(payload)
        except ElementTree.ParseError as exc:
            raise WebResearchError("Live web search returned an invalid RSS response.") from exc

        results: list[WebSearchResult] = []
        for item in root.findall("./channel/item"):
            title = self._text(item, "title")
            source_url = self._text(item, "link")
            snippet = self._strip_html(self._text(item, "description"))
            if not title or not source_url:
                continue
            results.append(WebSearchResult(
                title=title,
                source_url=source_url,
                snippet=snippet or "Search result snippet was not supplied by the provider.",
                provider_name=self.name,
            ))
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _text(item: ElementTree.Element, name: str) -> str:
        return (item.findtext(name) or "").strip()

    @staticmethod
    def _strip_html(value: str) -> str:
        return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", value))).strip()


class GdeltNewsSearchProvider:
    """Free, keyless discovery of recent news articles through GDELT DOC 2.0."""

    name = "gdelt_doc_api"
    endpoint = "https://api.gdeltproject.org/api/v2/doc/doc"
    max_response_bytes = 1_000_000

    def search(self, query: str, limit: int) -> list[WebSearchResult]:
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(max(1, min(limit, 250))),
        }
        request = Request(
            self.endpoint + "?" + urlencode(params),
            headers={"Accept": "application/json", "User-Agent": "StockResearchAssistant/0.1 (research discovery)"},
        )
        try:
            with urlopen(request, timeout=12) as response:  # noqa: S310 - fixed GDELT public API endpoint
                payload = response.read(self.max_response_bytes + 1)
        except OSError as exc:
            raise WebResearchError("GDELT news discovery is temporarily unavailable.") from exc
        if len(payload) > self.max_response_bytes:
            raise WebResearchError("GDELT response exceeded the safety limit.")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebResearchError("GDELT returned invalid JSON.") from exc
        articles = decoded.get("articles", []) if isinstance(decoded, dict) else []
        results: list[WebSearchResult] = []
        for article in articles:
            if not isinstance(article, dict):
                continue
            title = str(article.get("title", "")).strip()
            source_url = str(article.get("url", "")).strip()
            snippet = "GDELT news discovery result; open the source for article content."
            if title and source_url:
                results.append(WebSearchResult(
                    title=title,
                    source_url=source_url,
                    snippet=snippet,
                    provider_name=self.name,
                ))
            if len(results) >= limit:
                break
        return results


class FreeSearchProvider:
    """Ordered provider chain; defaults to GDELT then Bing without credentials."""

    name = "free_gdelt_then_bing"

    def __init__(self, providers: list[WebSearchProvider] | None = None) -> None:
        self.providers = providers or [GdeltNewsSearchProvider(), BingRssSearchProvider()]

    def search(self, query: str, limit: int) -> list[WebSearchResult]:
        results: list[WebSearchResult] = []
        seen_urls: set[str] = set()
        failures: list[WebResearchError] = []
        for provider in self.providers:
            try:
                candidates = provider.search(query, limit)
            except WebResearchError as exc:
                failures.append(exc)
                continue
            for candidate in candidates:
                if candidate.source_url not in seen_urls:
                    seen_urls.add(candidate.source_url)
                    results.append(WebSearchResult(
                        title=candidate.title,
                        source_url=candidate.source_url,
                        snippet=candidate.snippet,
                        provider_name=candidate.provider_name or provider.name,
                    ))
                if len(results) >= limit:
                    return results
        if results:
            return results
        if failures:
            raise failures[-1]
        return []


class BraveSearchApiProvider:
    """Official Brave Search API adapter, enabled only with an API key."""

    name = "brave_search_api"
    endpoint = "https://api.search.brave.com/res/v1/web/search"
    max_response_bytes = 1_000_000

    def __init__(self, api_key: str, *, country: str | None = None, search_lang: str = "vi") -> None:
        if not api_key.strip():
            raise ValueError("Brave Search API key cannot be empty.")
        self.api_key = api_key
        self.country = country
        self.search_lang = search_lang

    def search(self, query: str, limit: int) -> list[WebSearchResult]:
        params: dict[str, str] = {
            "q": query,
            "count": str(max(1, min(limit, 20))),
            "search_lang": self.search_lang,
            "safesearch": "moderate",
            "result_filter": "web",
        }
        # Brave supports Vietnamese search results, but VN is not currently
        # in its country-bias enum. Omitting country avoids HTTP 422 while the
        # query itself still carries the Vietnam/ticker context.
        if self.country:
            params["country"] = self.country
        freshness = os.getenv("STOCK_AGENT_WEB_SEARCH_FRESHNESS", "").strip()
        if freshness:
            params["freshness"] = freshness
        payload = self._request_json(self.endpoint + "?" + urlencode(params))
        raw_results = payload.get("web", {}).get("results", [])
        results: list[WebSearchResult] = []
        for item in raw_results:
            title = str(item.get("title", "")).strip()
            source_url = str(item.get("url", "")).strip()
            snippet = str(item.get("description", "")).strip() or "Search result snippet was not supplied by the provider."
            if title and source_url:
                results.append(WebSearchResult(
                    title=title,
                    source_url=source_url,
                    snippet=snippet,
                    provider_name=self.name,
                ))
            if len(results) >= limit:
                break
        return results

    def _request_json(self, request_url: str) -> dict[str, object]:
        request = Request(
            request_url,
            headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
        )
        try:
            with urlopen(request, timeout=12) as response:  # noqa: S310 - fixed official Brave API endpoint
                payload = response.read(self.max_response_bytes + 1)
        except HTTPError as exc:
            detail = self._http_error_detail(exc)
            label = {
                401: "authentication failed",
                403: "access was denied",
                422: "request parameters were rejected",
                429: "quota or rate limit was reached",
            }.get(exc.code, "request failed")
            suffix = f": {detail}" if detail else ""
            raise WebResearchError(f"Brave Search API {label} (HTTP {exc.code}){suffix}") from exc
        except (URLError, OSError) as exc:
            raise WebResearchError(f"Brave Search API network error: {type(exc).__name__}") from exc
        if len(payload) > self.max_response_bytes:
            raise WebResearchError("Brave Search API response exceeded the safety limit.")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebResearchError("Brave Search API returned invalid JSON.") from exc
        if not isinstance(decoded, dict):
            raise WebResearchError("Brave Search API returned an unexpected response.")
        return decoded

    @staticmethod
    def _http_error_detail(exc: HTTPError) -> str | None:
        try:
            payload = exc.read(65_537)
            if len(payload) > 65_536:
                return None
            decoded = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        error = decoded.get("error") if isinstance(decoded, dict) else None
        if not isinstance(error, dict):
            return None
        detail = error.get("detail")
        return str(detail)[:300] if detail else None


class WebResearchService:
    """Internet fallback for a local research corpus.

    It returns discovery metadata only. It does not crawl result pages, create
    claims, or write unlicensed third-party content to the local corpus.
    """

    def __init__(
        self,
        provider: WebSearchProvider | None = None,
        issuer_provider: IssuerDiscoveryProvider | None = None,
        company_name_resolver: CompanyNameResolver | None = None,
    ) -> None:
        self.provider = provider or self._provider_from_environment()
        self.issuer_provider = issuer_provider or self._issuer_provider_from_environment()
        self.company_name_resolver = company_name_resolver or self._company_name_resolver_from_environment()

    @staticmethod
    def _provider_from_environment() -> WebSearchProvider:
        enabled = os.getenv("STOCK_AGENT_WEB_SEARCH_ENABLED", "true").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return DisabledWebSearchProvider()
        provider = os.getenv("STOCK_AGENT_WEB_SEARCH_PROVIDER", "free").strip().lower()
        brave_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        key_is_placeholder = brave_key.casefold() in {"your_key", "your_api_key", "brave_api_key"}
        if provider == "brave" and brave_key and not key_is_placeholder:
            return BraveSearchApiProvider(brave_key)
        if provider == "brave":
            return DisabledWebSearchProvider()
        if provider == "auto" and brave_key and not key_is_placeholder:
            return FreeSearchProvider([
                BraveSearchApiProvider(brave_key),
                GdeltNewsSearchProvider(),
                BingRssSearchProvider(),
            ])
        if provider == "bing":
            return BingRssSearchProvider()
        if provider == "duckduckgo":
            return DuckDuckGoHtmlSearchProvider()
        if provider in {"auto", "free"}:
            return FreeSearchProvider()
        return DisabledWebSearchProvider()

    @staticmethod
    def _issuer_provider_from_environment() -> IssuerDiscoveryProvider:
        enabled = os.getenv("STOCK_AGENT_OFFICIAL_ISSUER_LOOKUP_ENABLED", "true").strip().lower()
        return HnxIssuerDiscoveryProvider() if enabled in {"1", "true", "yes", "on"} else DisabledIssuerDiscoveryProvider()

    @staticmethod
    def _company_name_resolver_from_environment() -> CompanyNameResolver:
        enabled = os.getenv("STOCK_AGENT_COMPANY_NAME_LOOKUP_ENABLED", "true").strip().lower()
        return VnstockCompanyNameResolver() if enabled in {"1", "true", "yes", "on"} else DisabledCompanyNameResolver()

    def search(self, query: str, ticker: str | None = None, limit: int = 5) -> list[WebCitation]:
        if not query or not query.strip():
            raise WebResearchError("Web search query cannot be empty.")
        retrieved_at = datetime.now(timezone.utc)
        issuer_results = self.issuer_provider.search(ticker) if ticker else []
        # An HNX profile is useful, but must not suppress current issuer-IR,
        # news, or industry sources returned by the configured provider.
        # The verified exchange profile already disambiguates the symbol. Avoid
        # an unnecessary VCI lookup in that case; it is only a fallback for an
        # ambiguous ticker when the exchange profile is unavailable.
        company_name = self.company_name_resolver.resolve(ticker) if ticker and not issuer_results else None
        search_query = self._build_search_query(query, ticker, company_name)
        if not search_query:
            raise WebResearchError("Web search query cannot be empty.")
        provider_results: list[WebSearchResult] = []
        provider_error: WebResearchError | None = None
        try:
            provider_results = self.provider.search(search_query, max(1, min(limit, 8)))
        except WebResearchError as exc:
            provider_error = exc
        if provider_error and not issuer_results:
            raise provider_error

        results: list[tuple[str, WebSearchResult]] = [
            (result.provider_name or self.issuer_provider.name, result) for result in issuer_results
        ]
        results.extend((result.provider_name or self.provider.name, result) for result in provider_results)
        citations: list[WebCitation] = []
        seen_urls: set[str] = set()
        for source_provider_name, result in results:
            if result.source_url in seen_urls:
                continue
            seen_urls.add(result.source_url)
            parsed = urlparse(result.source_url)
            if parsed.scheme != "https" or not parsed.netloc:
                continue
            is_official_issuer_result = source_provider_name == self.issuer_provider.name
            if not is_official_issuer_result and not self._is_relevant(result, search_query, ticker=ticker):
                continue
            citations.append(WebCitation(
                title=result.title,
                source_url=result.source_url,
                source_domain=parsed.netloc.lower(),
                snippet=result.snippet,
                search_provider=source_provider_name,
                retrieved_at=retrieved_at,
                rank=len(citations) + 1,
            ))
            if len(citations) >= limit:
                break
        return citations

    @staticmethod
    def _build_search_query(query: str, ticker: str | None, company_name: str | None = None) -> str:
        """Remove chat filler so a provider searches for research concepts."""
        cleaned = re.sub(r"\s+", " ", query).strip()
        normalized = WebResearchService._normalize(cleaned)
        research_terms: list[str] = []
        if "bao cao" in normalized:
            research_terms.extend(["annual report", "financial disclosure"])
        if "annual report" in normalized:
            research_terms.append("annual report")
        if "financial statement" in normalized:
            research_terms.append("financial statement")
        if "cong bo" in normalized:
            research_terms.append("disclosure")
        if "nghi quyet" in normalized:
            research_terms.append("resolution")
        if ticker:
            # The user may be asking about expansion, governance, operations,
            # or a specific risk.  Preserve that subject instead of collapsing
            # every ticker request into the same generic stock query.
            return " ".join([
                ticker.upper(), company_name or "Vietnam stock", cleaned, *research_terms,
            ]).strip()[:500]
        return cleaned

    @staticmethod
    def _is_relevant(result: WebSearchResult, query: str, ticker: str | None = None) -> bool:
        """Reject obvious ticker-name collisions before they reach the user."""
        stop_words = {"tim", "kiem", "cho", "toi", "ve", "cua", "va", "the", "nao", "thong", "tin"}
        query_terms = set(WebResearchService._tokens(query)) - stop_words
        normalized_query = WebResearchService._normalize(query)
        if "bao cao" in normalized_query:
            query_terms.update({"annual", "report"})
        if "cong bo" in normalized_query or "filing" in query_terms:
            query_terms.update({"disclosure", "filing"})
        if "nghi quyet" in normalized_query:
            query_terms.update({"resolution"})
        if not query_terms:
            return True
        haystack = set(WebResearchService._tokens(f"{result.title} {result.snippet} {result.source_url}"))
        # Do not let another issuer's annual-report result pass simply because
        # it shares generic words such as "annual" and "report". The exchange
        # result above is trusted as a ticker-specific lookup; generic web
        # candidates must also name the requested ticker.
        if ticker and ticker.casefold() not in haystack:
            return False
        matched = query_terms.intersection(haystack)
        minimum_matches = 2 if len(query_terms) >= 2 else 1
        if len(matched) >= minimum_matches:
            return True
        # Result titles/snippets often omit the full question even for a valid
        # issuer page. Keep symbol-matched candidates for content extraction;
        # a result becomes evidence only after its fetched body is checked.
        query_tokens = WebResearchService._tokens(query)
        return bool(query_tokens and query_tokens[0] in haystack)

    @staticmethod
    def _tokens(value: str) -> list[str]:
        return re.findall(r"[a-z0-9]{3,}", WebResearchService._normalize(value))

    @staticmethod
    def _normalize(value: str) -> str:
        decomposed = unicodedata.normalize("NFD", value.casefold())
        return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
