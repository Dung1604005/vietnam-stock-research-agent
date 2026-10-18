"""Identify the reporting period from headings, never from download/publication time."""
from __future__ import annotations

import calendar
import re
import unicodedata
from datetime import date

from stock_multi_agent.models.research import ReportingPeriod


def normalize(value):
    return "".join(c for c in unicodedata.normalize("NFD", value.lower().replace("đ", "d")) if unicodedata.category(c) != "Mn")


def _period(year, quarter=None, kind="annual", evidence="", end=None):
    month = {"half_year": 6, "nine_months": 9}.get(kind, quarter * 3 if quarter else 12)
    end = end or date(year, month, calendar.monthrange(year, month)[1])
    label = (f"Quý {quarter}/{year}" if kind == "quarter" else f"6 tháng đầu năm {year}" if kind == "half_year"
             else f"9 tháng đầu năm {year}" if kind == "nine_months" else f"Năm {year}" if kind == "annual" else f"{end:%d/%m/%Y}")
    return ReportingPeriod(year=year, quarter=quarter, kind=kind, end=end, label=label, status="DETECTED", evidence=evidence[:500])


def _heading_period(value, *, annual=False):
    text = normalize(value)
    # Non-calendar fiscal years need an explicit end date, not an assumed December 31.
    if re.search(r"(?:nien do|fiscal year)\s*(?:nam\s*)?20\d{2}\s*[-/]\s*20\d{2}", text):
        return ReportingPeriod(status="AMBIGUOUS", evidence=value[:500])
    found = []
    for match in re.finditer(r"(?:\bquy\s*|\bquarter\s*|\bq\s*)(iv|iii|ii|i|[1-4])\s*(?:[/.\-]\s*|(?:nam|year)\s*)?(20\d{2})\b", text):
        token, year = match.groups()
        quarter = int(token) if token.isdigit() else {"i": 1, "ii": 2, "iii": 3, "iv": 4}[token]
        found.append(_period(int(year), quarter, "quarter", match[0]))
    for match in re.finditer(r"\b(6|9)\s*(?:thang|months?)\s*(?:dau\s*)?(?:nam\s*|of\s*)?(20\d{2})\b|\b(?:ban nien|half[- ]year)\s*(?:nam\s*)?(20\d{2})\b", text):
        months, year, half_year = match.groups()
        kind = "half_year" if half_year or months == "6" else "nine_months"
        found.append(_period(int(year or half_year), kind=kind, evidence=match[0]))
    if not found:
        # Restrict annual years to report/year labels; an arbitrary publication date is not a fiscal year.
        patterns = [r"\b(?:nam|year|fy)\s*(20\d{2})\b", r"\b(?:annual report|bao cao thuong nien)\s*(?:nam\s*)?(20\d{2})\b"]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                found.append(_period(int(match[1]), evidence=match[0]))
        if annual and not found:
            for match in re.finditer(r"\b(20\d{2})\b", text):
                found.append(_period(int(match[1]), evidence=match[0]))
    unique = {(item.end, item.kind): item for item in found}
    if len(unique) > 1:
        return ReportingPeriod(status="AMBIGUOUS", evidence=value[:500])
    return next(iter(unique.values()), ReportingPeriod())


def infer_reporting_period(title: str, text: str = "", document_type=None, *, declared_end: date | None = None) -> ReportingPeriod:
    annual = str(getattr(document_type, "value", document_type)) == "annual_report"
    heading = _heading_period(title, annual=annual)
    body = normalize(text[:2200])
    ends = []
    # Only explicitly labelled period end dates, excluding dates in comparison tables and signatures.
    for match in re.finditer(r"(?:period ended|year ended|months ended|as at|ky ket thuc|nam ket thuc|thang ket thuc|tai ngay|giua nien do)\s*(?:ngay\s*)?[:：]?\s*(\d{4}-\d{2}-\d{2}|\d{1,2}[/.-]\d{1,2}[/.-]\d{4}|\d{1,2}\s+thang\s+\d{1,2}\s+nam\s+\d{4}|\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4})", body):
        value = match[1]
        try:
            if re.match(r"\d{4}-", value):
                ends.append(date.fromisoformat(value))
            elif "thang" in value:
                day, month, year = map(int, re.findall(r"\d+", value))
                ends.append(date(year, month, day))
            elif re.search(r"[a-z]", value):
                day, month, year = value.split()
                months = {name: index for index, name in enumerate("January February March April May June July August September October November December".split(), 1)}
                ends.append(date(int(year), months[month.title()], int(day)))
            else:
                day, month, year = map(int, re.split(r"[/.-]", value))
                ends.append(date(year, month, day))
        except ValueError:
            continue
    if declared_end:
        # Explicit structured metadata supplied by the caller has provenance separate from publication time.
        ends.append(declared_end)
    if len(set(ends)) > 1:
        return ReportingPeriod(status="AMBIGUOUS", evidence="Multiple labelled period ends in document heading")
    if ends:
        end = ends[0]
        if heading.end and heading.end != end:
            return ReportingPeriod(status="AMBIGUOUS", evidence=f"Title {heading.label} conflicts with period end {end}")
        kind = heading.kind if heading.end else "annual" if end.month == 12 and end.day == 31 else "dated"
        result = _period(end.year, heading.quarter, kind, f"Explicit reporting period end: {end}", end)
        return result
    if heading.status != "UNKNOWN":
        return heading
    # Scan only the first short heading block. Never infer a period from the entire financial table.
    first_lines = " ".join(line.strip() for line in text[:700].splitlines() if line.strip())
    result = _heading_period(first_lines, annual=annual)
    if result.kind == "annual" and not annual:
        normalized = normalize(first_lines)
        # A company founding year or a publication/copyright year is not its reporting period.
        if not re.search(r"(?:bao cao tai chinh|financial statements?|ket qua kinh doanh|earnings)[^.]{0,90}(?:nam|year|fy)\s*20\d{2}", normalized):
            return ReportingPeriod()
        if any(term in normalize(result.evidence) for term in ("thanh lap", "founded", "copyright")):
            return ReportingPeriod()
    return result


def period_display(item):
    period = item.reporting_period
    if period.status == "DETECTED" and period.end:
        return f"Dữ liệu đến kỳ: {period.label} (kết thúc {period.end:%d/%m/%Y})"
    if item.period_end and period.status != "AMBIGUOUS":
        return f"Dữ liệu đến kỳ: {item.period_end:%d/%m/%Y}"
    return "Dữ liệu đến kỳ: chưa xác định" + (" — thông tin kỳ mâu thuẫn" if period.status == "AMBIGUOUS" else "")
