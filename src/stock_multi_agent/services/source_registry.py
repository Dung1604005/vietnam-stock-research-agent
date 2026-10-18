from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from stock_multi_agent.models.research_crawler import SourceDefinition


def canonical_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Source must be a public HTTP(S) URL without credentials")
    query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
             if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}]
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", urlencode(sorted(query)), ""))


SourcePolicy = SourceDefinition


class SourceRegistry:
    def __init__(self, path: Path | None = None):
        path = path or Path(__file__).resolve().parents[3] / "config" / "research_sources.json"
        self.sources = [SourcePolicy.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))["sources"]]
        if len({source.source_id for source in self.sources}) != len(self.sources):
            raise ValueError("Duplicate research source_id")

    def by_id(self, source_id: str) -> SourcePolicy | None:
        return next((source for source in self.sources if source.source_id == source_id), None)

    def resolve(self, url: str) -> SourcePolicy | None:
        host = urlsplit(canonical_url(url)).hostname or ""
        matches = [source for source in self.sources if host == source.domain or host.endswith("." + source.domain) or host in source.asset_domains]
        return max(matches, key=lambda item: len(item.domain)) if matches else None
