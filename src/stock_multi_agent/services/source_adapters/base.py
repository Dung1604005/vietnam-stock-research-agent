from dataclasses import dataclass
from typing import Protocol


@dataclass
class DiscoveredLink:
    url: str
    title: str = ""
    published: str = ""
    date_evidence: str = ""
    kind: str = "listing"


class SourceAdapter(Protocol):
    def discover(self, body: str, base_url: str) -> list[DiscoveredLink]: ...
