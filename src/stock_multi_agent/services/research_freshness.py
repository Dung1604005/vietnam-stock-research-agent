"""Shared freshness policy for local SQL retrieval and live web evidence."""
from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from stock_multi_agent.models.research import Citation, ResearchDocumentType
from stock_multi_agent.services.reporting_period import infer_reporting_period, normalize, period_display


def today():
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date()


@dataclass(frozen=True)
class ResearchScope:
    mode: str = "ARCHIVE"
    years: tuple[int, ...] = ()
    quarter: int | None = None
    document_type: str | None = None
    period_kind: str | None = None
    period_pairs: tuple[tuple[int, int], ...] = ()


def research_scope(question, *, ticker=None, as_of=None, current=False):
    value = normalize(question)
    years = tuple(sorted({int(year) for year in re.findall(r"\b(20\d{2})\b", value)}))
    period = infer_reporting_period(question)
    if not years and period.status == "DETECTED" and period.year:
        years = (period.year,)
    document_type = "annual_report" if any(term in value for term in ("bao cao thuong nien", "annual report")) else None
    period_kind = "half_year" if any(term in value for term in ("ban nien", "6 thang", "half-year")) else "nine_months" if "9 thang" in value else None
    quarter_tokens = re.findall(r"(?:\bquy\s*|\bquarter\s*|\bq\s*)(iv|iii|ii|i|[1-4])\b", value)
    quarters = tuple(sorted({int(token) if token.isdigit() else {"i": 1, "ii": 2, "iii": 3, "iv": 4}[token] for token in quarter_tokens}))
    pairs = tuple((years[0], quarter) for quarter in quarters) if len(years) == 1 else ()
    if len(years) > 1:
        explicit_pairs = []
        for match in re.finditer(r"(?:\bquy\s*|\bquarter\s*|\bq\s*)(iv|iii|ii|i|[1-4])\s*(?:[/.\-]\s*|(?:nam|year)\s*)?(20\d{2})\b", value):
            token, year = match.groups()
            number = int(token) if token.isdigit() else {"i": 1, "ii": 2, "iii": 3, "iv": 4}[token]
            explicit_pairs.append((int(year), number))
        pairs = tuple(sorted(set(explicit_pairs)))
    if as_of or years or any(term in value for term in ("lich su", "historical")):
        return ResearchScope("HISTORY", years, period.quarter if len(quarters) <= 1 else None, document_type, period_kind, pairs)
    recent = any(term in value for term in ("moi nhat", "hien tai", "gan day", "hom nay", "latest", "current", "recent"))
    return ResearchScope("CURRENT" if current or ticker or recent else "ARCHIVE", document_type=document_type, period_kind=period_kind)


@lru_cache(maxsize=1)
def policy():
    path = Path(__file__).resolve().parents[3] / "config" / "research_freshness.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    for rule in config["rules"].values():
        if not 1 <= rule["publication_days"] <= 3650 or not 1 <= rule["period_days"] <= 3650:
            raise ValueError("Invalid freshness age limit")
    return config


def citation_period(citation):
    if citation.reporting_period.status != "UNKNOWN":
        return citation.reporting_period
    return infer_reporting_period(citation.title, citation.quote,
        getattr(citation, "document_type", None), declared_end=getattr(citation, "period_end", None))


def classify_document_type(title, text=""):
    value = normalize(title + " " + text[:300])
    for terms, kind in (
        (("annual report", "bao cao thuong nien"), "annual_report"),
        (("financial statement", "bao cao tai chinh", "bctc"), "financial_statement"),
        (("earnings", "ket qua kinh doanh", "tinh hinh kinh doanh"), "earnings_release"),
        (("investor presentation",), "investor_presentation"),
        (("cong bo thong tin", "disclosure"), "regulatory_disclosure"),
    ):
        if any(term in value for term in terms):
            return kind
    if re.search(r"\d", value) and any(term in value for term in ("doanh thu", "loi nhuan", "revenue", "net profit", "earnings per share")):
        return "earnings_release"
    return "other"


def freshness_reason(citation, scope, reference=None):
    reference = reference or today()
    period = citation_period(citation)
    kind = str(getattr(getattr(citation, "document_type", None), "value", getattr(citation, "document_type", None)) or classify_document_type(citation.title))
    if kind == "other":
        kind = classify_document_type(citation.title, citation.quote)
    if citation.published_at.date() > reference:
        return "FUTURE_PUBLICATION_DATE"
    if scope.document_type and kind != scope.document_type:
        return "DOCUMENT_TYPE_MISMATCH"
    if scope.period_kind and period.kind != scope.period_kind:
        return "REPORT_KIND_MISMATCH"
    if scope.mode == "HISTORY":
        if scope.years and (period.status != "DETECTED" or period.year not in scope.years):
            return "REPORT_YEAR_MISMATCH"
        if scope.quarter and (period.kind != "quarter" or period.quarter != scope.quarter):
            return "REPORT_QUARTER_MISMATCH"
        if scope.period_pairs and (period.year, period.quarter) not in scope.period_pairs:
            return "REPORT_QUARTER_MISMATCH"
        return None
    if scope.mode != "CURRENT":
        return None
    rule = policy()["rules"].get(kind, policy()["rules"]["other"])
    if (reference - citation.published_at.date()).days > rule["publication_days"]:
        return "STALE_PUBLICATION"
    if period.status == "AMBIGUOUS":
        return "AMBIGUOUS_REPORTING_PERIOD"
    if rule["require_period"] and (period.status != "DETECTED" or period.end is None):
        return "UNKNOWN_REPORTING_PERIOD"
    if period.end:
        if period.end > min(reference, citation.published_at.date()):
            return "FUTURE_REPORTING_PERIOD"
        maximum = policy()["annual_financial_period_days"] if kind in {"financial_statement", "earnings_release"} and period.kind == "annual" else rule["period_days"]
        if (reference - period.end).days > maximum:
            return "STALE_REPORTING_PERIOD"
    return None


def filter_citations(citations, scope, reference=None):
    eligible, excluded = [], []
    for citation in citations:
        item = citation.model_copy(update={"reporting_period": citation_period(citation)})
        if freshness_reason(item, scope, reference):
            excluded.append(item)
        else:
            eligible.append(item)
    if scope.mode == "CURRENT":
        latest = {}
        for item in eligible:
            kind = getattr(item, "document_type", None) or classify_document_type(item.title)
            key = (tuple(getattr(item, "tickers", [])), str(kind))
            if item.reporting_period.end:
                latest[key] = max(latest.get(key, date.min), item.reporting_period.end)
        result = []
        for item in eligible:
            kind = getattr(item, "document_type", None) or classify_document_type(item.title)
            key = (tuple(getattr(item, "tickers", [])), str(kind))
            if item.reporting_period.end and item.reporting_period.end < latest.get(key, date.min):
                excluded.append(item)
            else:
                result.append(item)
        eligible = result
    return eligible, excluded


def external_to_citation(item, ticker=None):
    if item.published_at is None:
        return None
    from stock_multi_agent.services.source_registry import SourceRegistry
    source = SourceRegistry().resolve(str(item.canonical_url))
    identity = hashlib.sha256(f"{item.canonical_url}:{item.content_sha256}".encode()).hexdigest()
    return Citation(document_id=identity, chunk_id=identity, title=item.title, source_url=item.canonical_url,
        publisher=source.publisher if source else item.source_domain, published_at=item.published_at,
        retrieved_at=item.retrieved_at, content_sha256=item.content_sha256, quote=item.quote,
        retrieval_score=item.relevance_score, retrieval_modes=["web_extracted"], quality_tier=source.quality_tier if source else "T4",
        reporting_period=item.reporting_period, period_end=item.reporting_period.end,
        document_type=classify_document_type(item.title, item.quote), tickers=[ticker] if ticker else [])


def freshness_note(citations, *, insufficient=False, historical=False):
    lines = list(dict.fromkeys(period_display(item) for item in citations))[:5]
    if insufficient:
        return "Chưa có dữ liệu cập nhật đã xác minh đáp ứng yêu cầu. " + ("; ".join(lines) + ". Các tài liệu này chỉ dùng tham khảo lịch sử hoặc chưa đủ thông tin xác nhận kỳ hiện tại." if lines else "Chưa xác định được kỳ dữ liệu phù hợp.")
    return ("Tra cứu lịch sử. " if historical else "") + "; ".join(lines)


def period_sql(scope, reference, alias="d", *, latest=True):
    """Return SQL predicates before candidate LIMIT; thresholds come only from validated config."""
    conditions, params = [], []
    meta = f"{alias}.reporting_period_json"
    year = f"json_extract({meta}, '$.year')"
    kind = f"json_extract({meta}, '$.kind')"
    status = f"json_extract({meta}, '$.status')"
    if scope.document_type:
        conditions.append(f"{alias}.document_type=?")
        params.append(scope.document_type)
    if scope.period_kind:
        conditions.append(f"{kind}=?")
        params.append(scope.period_kind)
    if scope.years:
        conditions.append(f"{year} IN ({','.join('?' for _ in scope.years)}) AND {status}='DETECTED'")
        params.extend(scope.years)
    if scope.quarter:
        conditions.append(f"{kind}='quarter' AND json_extract({meta}, '$.quarter')=?")
        params.append(scope.quarter)
    if scope.period_pairs:
        conditions.append("(" + " OR ".join(f"({year}=? AND json_extract({meta}, '$.quarter')=? AND {kind}='quarter')" for _ in scope.period_pairs) + ")")
        for pair in scope.period_pairs:
            params.extend(pair)
    if scope.mode != "CURRENT":
        return conditions, params
    alternatives = []
    for doc_type, rule in policy()["rules"].items():
        pub_age = f"julianday(?) - julianday(date({alias}.published_at)) BETWEEN 0 AND {rule['publication_days']}"
        limit = f"CASE WHEN {kind}='annual' AND {alias}.document_type IN ('financial_statement','earnings_release') THEN {policy()['annual_financial_period_days']} ELSE {rule['period_days']} END"
        period_age = f"({status}='DETECTED' AND {alias}.period_end IS NOT NULL AND {alias}.period_end<=date({alias}.published_at) AND julianday(?) - julianday({alias}.period_end) BETWEEN 0 AND {limit})"
        period_check = period_age if rule["require_period"] else f"({period_age} OR ({alias}.period_end IS NULL AND {status}='UNKNOWN'))"
        alternatives.append(f"({alias}.document_type='{doc_type}' AND {pub_age} AND {period_check})")
        params.extend([reference.isoformat(), reference.isoformat()])
    conditions.append("(" + " OR ".join(alternatives) + ")")
    if latest:
        # Compare the entire eligible corpus, not merely the chunks that happened to enter top-k.
        newer_conditions, newer_params = period_sql(scope, reference, alias="recent", latest=False)
        conditions.append(f"""NOT EXISTS (SELECT 1 FROM research_documents recent
            WHERE recent.document_type={alias}.document_type AND recent.period_end>{alias}.period_end
              AND date(recent.published_at)<=? AND {' AND '.join(newer_conditions)}
              AND EXISTS (SELECT 1 FROM json_each(recent.tickers_json) a JOIN json_each({alias}.tickers_json) b ON a.value=b.value)
              AND NOT EXISTS (SELECT 1 FROM research_documents correction WHERE correction.supersedes_id=recent.document_id AND date(correction.published_at)<=?))""")
        params.extend([reference.isoformat(), *newer_params, reference.isoformat()])
    return conditions, params
