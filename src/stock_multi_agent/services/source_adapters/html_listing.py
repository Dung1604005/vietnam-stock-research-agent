from html.parser import HTMLParser
from urllib.parse import urljoin
import re
from stock_multi_agent.services.source_adapters.base import DiscoveredLink


class _Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.current = [], None
        self.ignore = 0
        self.frames = []

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if tag not in {"br", "img", "meta", "link", "input", "hr", "source", "wbr", "embed", "area", "base", "param", "track", "col"}:
            self.frames.append({"tag": tag, "text": "", "links": [], "scope": tag in {"tr", "li", "div"}})
        if tag in {"script", "style", "noscript"}:
            self.ignore += 1
        if tag == "a" and data.get("href") and not self.ignore:
            self.current = [data["href"], data.get("title", "") or "", data.get("data-published-at", "") or ""]

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self.ignore = max(0, self.ignore - 1)
        if tag == "a" and self.current:
            self.links.append(self.current)
            for frame in self.frames:
                if frame["scope"]:
                    frame["links"].append(self.current)
            self.current = None
        if self.frames and any(frame["tag"] == tag for frame in self.frames):
            while self.frames:
                frame = self.frames.pop()
                if frame["scope"] and len(frame["text"]) <= 1400:
                    dates = set(re.findall(r"(?<!\d)\d{1,2}/\d{1,2}/\d{4}(?!\d)", frame["text"]))
                    if len(dates) == 1:
                        for link in frame["links"]:
                            if not link[2]:
                                link[2] = next(iter(dates))
                            short_title = link[1].strip().lower()
                            if len(short_title) < 12 or short_title in {"xem trực tuyến", "tải xuống"} or short_title.startswith("cập nhật"):
                                contextual_title = " ".join(frame["text"].split())[:500]
                                if len(contextual_title) > len(link[1]):
                                    link[1] = contextual_title
                if frame["tag"] == tag:
                    break

    def handle_data(self, data):
        if self.current and not self.ignore:
            self.current[1] += " " + data
        if not self.ignore:
            for frame in self.frames:
                if frame["scope"] and len(frame["text"]) <= 1400:
                    frame["text"] += " " + data


class HtmlListingAdapter:
    def discover(self, body, base_url):
        parser = _Links()
        parser.feed(body)
        return [DiscoveredLink(urljoin(base_url, url), " ".join(title.split())[:500], published,
                               f"Listing entry publication date: {published}" if published else "")
                for url, title, published in parser.links if not url.startswith(("#", "javascript:", "mailto:"))]
