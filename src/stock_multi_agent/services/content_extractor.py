from __future__ import annotations

import hashlib
import re
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from stock_multi_agent.models.research import ExtractedDocument, FetchedDocument


class _ArticleTextParser(HTMLParser):
    _blocked_tags = frozenset({"script", "style", "noscript", "svg", "canvas", "iframe", "form"})
    _boilerplate_tags = frozenset({"nav", "footer", "header", "aside"})
    _block_tags = frozenset({"article", "main", "section", "p", "h1", "h2", "h3", "h4", "li", "div", "br"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._title_depth = 0
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.canonical_href: str | None = None
        self.published_value: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): (value or "") for key, value in attrs}
        lowered = tag.lower()
        if lowered in self._blocked_tags or lowered in self._boilerplate_tags:
            self._ignored_depth += 1
            return
        if lowered == "title":
            self._title_depth += 1
        if lowered == "link" and attributes.get("rel", "").lower() == "canonical":
            self.canonical_href = attributes.get("href") or self.canonical_href
        if lowered == "meta":
            key = (attributes.get("property") or attributes.get("name") or attributes.get("itemprop") or "").lower()
            if key in {"article:published_time", "date", "datepublished", "publishdate", "publication_date"}:
                self.published_value = attributes.get("content") or self.published_value
        if self._ignored_depth == 0 and lowered in self._block_tags:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in self._blocked_tags or lowered in self._boilerplate_tags:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if lowered == "title":
            self._title_depth = max(0, self._title_depth - 1)
        if self._ignored_depth == 0 and lowered in self._block_tags:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._title_depth:
            self.title_parts.append(data)
        self.text_parts.append(data)


class HtmlContentExtractor:
    """Extract human-readable HTML text and conservative provenance metadata."""

    def extract(self, fetched: FetchedDocument) -> ExtractedDocument:
        parser = _ArticleTextParser()
        parser.feed(fetched.body)
        parser.close()
        text = self._clean_text(" ".join(parser.text_parts))
        if not text:
            raise ValueError("Fetched HTML has no extractable text.")
        title = self._clean_text(" ".join(parser.title_parts)) or urlsplit(str(fetched.final_url)).netloc
        canonical_url = self._canonical_url(str(fetched.final_url), parser.canonical_href)
        return ExtractedDocument(
            requested_url=fetched.requested_url,
            final_url=fetched.final_url,
            canonical_url=canonical_url,
            title=title[:500],
            published_at=self._parse_aware_datetime(parser.published_value),
            retrieved_at=fetched.retrieved_at,
            text=text,
            content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    @staticmethod
    def _clean_text(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _canonical_url(final_url: str, candidate: str | None) -> str:
        if not candidate:
            return final_url
        resolved = urljoin(final_url, candidate)
        parsed = urlsplit(resolved)
        return resolved if parsed.scheme == "https" and parsed.netloc else final_url

    @staticmethod
    def _parse_aware_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None
