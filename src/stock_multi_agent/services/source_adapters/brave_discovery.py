import os
from stock_multi_agent.services.source_adapters.base import DiscoveredLink


class BraveDiscoveryAdapter:
    def discover_source(self, source):
        from stock_multi_agent.services.web_research import BraveSearchApiProvider
        key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        if not key:
            return []
        provider = BraveSearchApiProvider(key)
        results = provider.search(f"site:{source.domain} {' '.join(source.ticker_map)} báo cáo tài chính", 5)
        return [DiscoveredLink(item.source_url, item.title) for item in results if source.accepts(item.source_url)]
