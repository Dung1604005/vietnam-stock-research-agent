import json
from urllib.parse import urljoin
from stock_multi_agent.services.source_adapters.base import DiscoveredLink


class JsonListingAdapter:
    def discover(self, body, base_url):
        links = []
        def visit(value, depth=0):
            if depth > 15:
                return
            if isinstance(value, dict):
                url = next((value.get(key) for key in ("url", "href", "download_url", "file_url") if isinstance(value.get(key), str)), None)
                if url:
                    published = next((value[key] for key in ("published_at", "datePublished", "published") if isinstance(value.get(key), str)), "")
                    links.append(DiscoveredLink(urljoin(base_url, url), str(value.get("title", ""))[:500], published,
                                                f"JSON publication: {published}" if published else ""))
                for child in value.values():
                    visit(child, depth + 1)
            elif isinstance(value, list):
                for child in value:
                    visit(child, depth + 1)
        visit(json.loads(body))
        return links
