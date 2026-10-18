"""Conservative metadata extraction. Modification/creation times are not publication dates."""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from zoneinfo import ZoneInfo

from stock_multi_agent.models.research import ResearchDocumentType
from stock_multi_agent.services.reporting_period import infer_reporting_period


def plain(value):
    return "".join(char for char in unicodedata.normalize("NFD", value.lower().replace("đ", "d"))
                   if unicodedata.category(char) != "Mn")


def publication_date(value: str, timezone: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    parsed = None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (ValueError, TypeError):
            for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
                try:
                    parsed = datetime.strptime(value, fmt)
                    break
                except ValueError:
                    continue
    if parsed is None:
        return None
    # Date-only values use midnight in the source timezone, with precision recorded in evidence.
    return parsed if parsed.utcoffset() is not None else parsed.replace(tzinfo=ZoneInfo(timezone))


class _Metadata(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.dates, self.json_parts = [], []
        self.in_json = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        key = (attrs.get("property") or attrs.get("name") or attrs.get("itemprop") or "").lower()
        if tag == "meta" and key in {"article:published_time", "datepublished", "publishdate", "publication_date"}:
            self.dates.append((attrs.get("content", ""), "HTML " + key))
        if tag == "script" and attrs.get("type") == "application/ld+json":
            self.in_json = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_json = False

    def handle_data(self, data):
        if self.in_json:
            self.json_parts.append(data)


def enrich_candidate(candidate, extracted, fetched, source):
    candidate = candidate.model_copy(deep=True)
    previous_date, previous_evidence = candidate.published_at, candidate.date_evidence
    if not candidate.date_evidence.startswith("User reviewed:"):
        candidate.published_at, candidate.date_evidence = None, ""
    if extracted.title not in {"PDF report", source.domain} and len(extracted.title.strip()) > 5:
        candidate.title = extracted.title[:500]
    if not candidate.title:
        candidate.title = extracted.title[:500]
    candidate.publisher = source.publisher
    candidate.tickers = candidate.tickers or source.ticker_map
    if not candidate.published_at:
        dates = []
        if fetched.content_type != "application/pdf":
            parser = _Metadata()
            parser.feed(fetched.body)
            dates.extend(parser.dates)
            def walk(value):
                if isinstance(value, dict):
                    if isinstance(value.get("datePublished"), str):
                        dates.append((value["datePublished"], "JSON-LD datePublished"))
                    for sub in value.values():
                        walk(sub)
                elif isinstance(value, list):
                    for sub in value:
                        walk(sub)
            for part in parser.json_parts:
                try:
                    walk(json.loads(part))
                except (ValueError, RecursionError):
                    continue
        # Only labelled publication dates from body; fiscal period/signature dates are deliberately excluded.
        match = re.search(r"(?:ngày công bố|ngày đăng|published(?: on)?|publication date)\s*[:：]?\s*(\d{4}-\d{2}-\d{2}|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4})",
                          extracted.text, re.IGNORECASE)
        if match:
            dates.append((match[1], "Labelled publication in document: " + match[0]))
        for value, origin in dates:
            timestamp = publication_date(value, source.timezone)
            if timestamp:
                candidate.published_at = timestamp
                candidate.date_evidence = f"{origin}: {value}; source timezone={source.timezone} (date-only → midnight)"[:500]
                break
        if not candidate.published_at:
            candidate.published_at, candidate.date_evidence = previous_date, previous_evidence
    text = plain(candidate.title + " " + extracted.text[:3000])
    if candidate.document_type == ResearchDocumentType.OTHER:
        for terms, kind in (
            (("bao cao thuong nien", "annual report"), ResearchDocumentType.ANNUAL_REPORT),
            (("bao cao tai chinh", "financial statement", "bctc"), ResearchDocumentType.FINANCIAL_STATEMENT),
            (("investor presentation", "trinh bay nha dau tu"), ResearchDocumentType.INVESTOR_PRESENTATION),
            (("ket qua kinh doanh", "earnings", "business results"), ResearchDocumentType.EARNINGS_RELEASE),
            (("cong bo thong tin", "disclosure"), ResearchDocumentType.REGULATORY_DISCLOSURE),
        ):
            if any(term in text for term in terms):
                candidate.document_type = kind
                break
    # Re-read the fetched document: a previous revision's period must not constrain a new PDF.
    candidate.reporting_period = infer_reporting_period(candidate.title, extracted.text, candidate.document_type)
    candidate.period_end = candidate.reporting_period.end.isoformat() if candidate.reporting_period.end else None
    candidate.metadata_confidence = sum((bool(candidate.title), bool(candidate.publisher), bool(candidate.tickers),
                                         bool(candidate.published_at), candidate.document_type != ResearchDocumentType.OTHER)) / 5
    return candidate


def metadata_problem(candidate, text):
    normalized = plain(text)
    if len(text.strip()) < 120:
        return "INSUFFICIENT_TEXT"
    if any(marker in normalized[:1500] for marker in ("access denied", "captcha", "403 forbidden", "please sign in", "login required", "verify you are human")):
        return "LOGIN_OR_ERROR_PAGE"
    if not candidate.title or not candidate.publisher:
        return "MISSING_IDENTITY"
    if not candidate.published_at or not candidate.date_evidence:
        return "MISSING_PUBLICATION_DATE"
    if candidate.published_at.date() > datetime.now(candidate.published_at.tzinfo).date():
        return "FUTURE_PUBLICATION_DATE"
    if not candidate.tickers:
        return "MISSING_TICKER"
    if any(not re.fullmatch(r"[A-Z][A-Z0-9]{1,9}", item) for item in candidate.tickers):
        return "INVALID_TICKER"
    if candidate.document_type == ResearchDocumentType.OTHER:
        return "UNKNOWN_DOCUMENT_TYPE"
    if candidate.reporting_period.status == "AMBIGUOUS":
        return "AMBIGUOUS_REPORTING_PERIOD"
    if candidate.period_end:
        try:
            date.fromisoformat(candidate.period_end)
        except ValueError:
            return "INVALID_REPORTING_PERIOD"
    return None
