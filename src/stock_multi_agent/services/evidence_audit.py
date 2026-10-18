"""Conservative mechanical grounding checks. Not a semantic truth verifier."""
from __future__ import annotations

import re
import unicodedata

from stock_multi_agent.models.research import Citation


def _numbers(value: str) -> set[str]:
    value = value.replace("−", "-")
    return {re.sub(r"\s+", "", token.replace(",", ".").rstrip(".")) for token in re.findall(r"(?<!\w)[-+]?\d+(?:[.,]\d+)*(?:\s*%)?", value)}


def audit_claim(claim: str, citations: list[Citation]) -> list[str]:
    if not citations:
        return ["missing_evidence"]
    source = " ".join(f"{item.title} {item.quote} {item.published_at.date().isoformat()}" for item in citations)
    problems = []
    if _numbers(claim).difference(_numbers(source)):
        problems.append("unsupported_numeric_claim")
    def quantities(text):
        plain = "".join(ch for ch in unicodedata.normalize("NFD", text.casefold().replace("đ", "d")) if unicodedata.category(ch) != "Mn")
        return {(number.replace(",", "."), unit) for number, unit in re.findall(
            r"(?<!\w)(\d+(?:[.,]\d+)*)\s*(trieu|ty|nghin|vnd|usd|%)\b", plain)}
    if quantities(claim).difference(quantities(source)):
        problems.append("unsupported_unit")
    if any(not item.publisher.strip() or item.published_at.utcoffset() is None or item.retrieved_at.utcoffset() is None for item in citations):
        problems.append("incomplete_provenance")
    # Require at least one content-bearing anchor. Alias expansion covers common
    # Vietnamese/English financial concepts; this is deliberately conservative.
    from stock_multi_agent.services.embeddings import LocalHashEmbeddingProvider
    aliases = LocalHashEmbeddingProvider._ALIASES
    def terms(text):
        plain = "".join(ch for ch in unicodedata.normalize("NFD", text.casefold().replace("đ", "d")) if unicodedata.category(ch) != "Mn")
        for phrase, values in aliases.items():
            if any(re.search(rf"\b{re.escape(value)}\b", plain) for value in (phrase, *values)):
                plain += " " + phrase + " " + " ".join(values)
        stop = {"cua", "cho", "bao", "cao", "the", "and", "doanh", "nghiep", "duoc", "voi", "trong", "mot", "cac", "nay"}
        return set(re.findall(r"[a-z]{3,}", plain)).difference(stop)
    if not terms(claim).intersection(terms(source)):
        problems.append("no_content_anchor")
    return problems
