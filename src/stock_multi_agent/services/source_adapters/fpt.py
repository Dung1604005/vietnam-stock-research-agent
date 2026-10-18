from urllib.parse import parse_qs, urljoin, urlsplit
from stock_multi_agent.services.source_adapters.html_listing import HtmlListingAdapter


class FptListingAdapter(HtmlListingAdapter):
    """FPT's download and view buttons refer to the same /api/media PDF."""
    def discover(self, body, base_url):
        links = super().discover(body, base_url)
        for link in links:
            parsed = urlsplit(link.url)
            if parsed.path == "/api/download":
                media = parse_qs(parsed.query).get("url", [""])[0]
                if media.startswith("/api/media/"):
                    link.url = urljoin(base_url, media)
        return links
