from xml.etree import ElementTree as ET
from stock_multi_agent.services.source_adapters.base import DiscoveredLink


class SitemapAdapter:
    def discover(self, body, base_url):
        if "<!DOCTYPE" in body.upper() or "<!ENTITY" in body.upper():
            raise ValueError("Unsafe XML declaration")
        root = ET.fromstring(body)
        links = []
        for item in root:
            values = {child.tag.rsplit("}", 1)[-1]: child.text for child in item}
            if values.get("loc"):
                # lastmod describes modification, not publication.
                kind = "sitemap" if item.tag.rsplit("}", 1)[-1] == "sitemap" else "listing"
                links.append(DiscoveredLink(values["loc"], kind=kind))
        return links
