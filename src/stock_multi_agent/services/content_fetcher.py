from __future__ import annotations

import ipaddress
import socket
import ssl
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from stock_multi_agent.models.research import FetchedDocument


class ContentFetchError(RuntimeError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class SafeContentFetcher:
    """Fetch small public HTML documents with redirect and SSRF protections."""

    allowed_content_types = frozenset({"text/html", "application/xhtml+xml"})

    def __init__(
        self,
        *,
        timeout_seconds: int = 12,
        max_bytes: int = 1_500_000,
        max_redirects: int = 3,
        resolver: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.resolver = resolver or self._resolve_public_addresses

    def fetch(self, source_url: str, *, timeout_seconds: int | None = None, max_bytes: int | None = None) -> FetchedDocument:
        timeout = timeout_seconds or self.timeout_seconds
        byte_limit = max_bytes or self.max_bytes
        current_url = self._canonical_request_url(source_url)
        requested_url = current_url

        for _ in range(self.max_redirects + 1):
            self._validate_public_https_url(current_url)
            status, headers, payload = self._open_once(current_url, timeout, byte_limit)
            if status in {301, 302, 303, 307, 308}:
                location = headers.get("Location")
                if not location:
                    raise ContentFetchError("Redirect response has no Location header.")
                from urllib.parse import urljoin

                current_url = self._canonical_request_url(urljoin(current_url, location))
                continue
            if not 200 <= status < 300:
                raise ContentFetchError(f"Source returned HTTP {status}.")
            content_type = headers.get_content_type().lower()
            if content_type not in self.allowed_content_types:
                raise ContentFetchError(f"Unsupported content type: {content_type or 'unknown'}.")
            charset = headers.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="replace")
            if not body.strip():
                raise ContentFetchError("Source returned an empty document.")
            return FetchedDocument(
                requested_url=requested_url,
                final_url=current_url,
                content_type=content_type,
                status_code=status,
                retrieved_at=datetime.now(timezone.utc),
                body=body,
            )
        raise ContentFetchError(f"Source exceeded the redirect limit ({self.max_redirects}).")

    def _open_once(self, source_url: str, timeout: int, byte_limit: int) -> tuple[int, Any, bytes]:
        request = Request(source_url, headers={"User-Agent": "StockResearchAssistant/0.1 (research retrieval)"})
        opener = build_opener(_NoRedirect(), HTTPSHandler(context=self._ssl_context()))
        try:
            with opener.open(request, timeout=timeout) as response:
                payload = response.read(byte_limit + 1)
                if len(payload) > byte_limit:
                    raise ContentFetchError(f"Source exceeds the {byte_limit} byte safety limit.")
                return response.status, response.headers, payload
        except HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308}:
                return exc.code, exc.headers, b""
            raise ContentFetchError(f"Source returned HTTP {exc.code}.") from exc
        except (OSError, URLError) as exc:
            raise ContentFetchError("Unable to retrieve the source.") from exc

    @staticmethod
    def _ssl_context() -> ssl.SSLContext:
        try:
            import certifi
        except ImportError:
            return ssl.create_default_context()
        return ssl.create_default_context(cafile=certifi.where())

    def _validate_public_https_url(self, source_url: str) -> None:
        parsed = urlsplit(source_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ContentFetchError("Only public HTTPS URLs may be retrieved.")
        if parsed.username or parsed.password:
            raise ContentFetchError("URLs with embedded credentials are not allowed.")
        hostname = parsed.hostname.lower().rstrip(".")
        if hostname == "localhost" or hostname.endswith(".local"):
            raise ContentFetchError("Local network URLs are not allowed.")
        try:
            addresses = self.resolver(hostname)
        except OSError as exc:
            raise ContentFetchError("Could not resolve source hostname.") from exc
        if not addresses:
            raise ContentFetchError("Source hostname resolved to no addresses.")
        for address in addresses:
            try:
                parsed_address = ipaddress.ip_address(address)
            except ValueError as exc:
                raise ContentFetchError("Source hostname resolved to an invalid address.") from exc
            if not parsed_address.is_global:
                raise ContentFetchError("Source hostname resolves to a non-public address.")

    @staticmethod
    def _resolve_public_addresses(hostname: str) -> list[str]:
        records = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        return list(dict.fromkeys(record[4][0] for record in records))

    @staticmethod
    def _canonical_request_url(source_url: str) -> str:
        parsed = urlsplit(source_url)
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))
