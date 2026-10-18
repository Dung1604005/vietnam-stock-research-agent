"""Inspect a bounded public listing using exactly the crawler's access policy."""
import argparse
import json
import re
import time

from stock_multi_agent.config import load_workspace_env
from stock_multi_agent.services.research_crawl_transport import CrawlTransport
from stock_multi_agent.services.research_discovery import ResearchDiscovery
from stock_multi_agent.services.source_registry import SourceRegistry


def main():
    load_workspace_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    args = parser.parse_args()
    source = SourceRegistry().by_id(args.source)
    if not source or not source.entrypoints:
        parser.error("Unknown source or no entrypoints")
    transport = CrawlTransport(source, deadline=time.monotonic() + 30)
    response = transport.fetch(str(source.entrypoints[0].url))
    links = ResearchDiscovery.adapters[source.entrypoints[0].kind].discover(response.body, str(response.final_url))
    print(json.dumps({"source": source.source_id, "status": response.status_code, "link_count": len(links),
        "document_links": [vars(link) for link in links if ".pdf" in link.url.lower()][:5],
        "pdf_html_context": [response.body[max(0, match.start()-180):match.end()+180]
            for match in list(re.finditer(r"\.pdf", response.body, re.IGNORECASE))[:3]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
