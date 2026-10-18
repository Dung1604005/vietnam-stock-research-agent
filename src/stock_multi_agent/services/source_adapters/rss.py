from urllib.parse import urljoin
from xml.etree import ElementTree as ET
from stock_multi_agent.services.source_adapters.base import DiscoveredLink


class RssAdapter:
    def discover(self, body, base_url):
        if "<!DOCTYPE" in body.upper() or "<!ENTITY" in body.upper():
            raise ValueError("Unsafe XML declaration")
        root = ET.fromstring(body)
        links = []
        for item in root.iter():
            if item.tag.rsplit("}", 1)[-1] not in {"item", "entry"}:
                continue
            values = {child.tag.rsplit("}", 1)[-1]: child for child in item}
            title = "".join(values["title"].itertext()) if "title" in values else ""
            published = next(((values[key].text or "") for key in ("pubDate", "published", "date") if key in values), "")
            for child in item:
                tag = child.tag.rsplit("}", 1)[-1]
                url = (child.get("href") or child.text or "") if tag == "link" else (child.get("url") or "") if tag == "enclosure" else ""
                if url.strip() and (tag != "link" or child.get("rel", "alternate") in {"alternate", "enclosure"}):
                    links.append(DiscoveredLink(urljoin(base_url, url.strip()), title, published,
                                                f"feed publication: {published}" if published else ""))
        return links
