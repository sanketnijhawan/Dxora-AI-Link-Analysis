"""Download one public HTML document through a DNS-pinned connection."""
import ipaddress
import re
import socket
from dataclasses import dataclass

import aiohttp
from aiohttp.abc import AbstractResolver
from bs4 import BeautifulSoup

from app.utils.validators import UnsafeURL, check_url, resolve_public_addresses

MAX_BYTES = 750_000
MAX_TEXT_CHARS = 12_000


class WebsiteFetchError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


@dataclass
class ExtractedSite:
    url: str
    title: str | None
    meta_description: str | None
    headings: list[str]
    content: str
    structured_data: list[str]
    word_count: int


class PinnedPublicResolver(AbstractResolver):
    """Avoid DNS rebinding: transport connects to already-checked IPs."""

    def __init__(self, hostname: str, ips: list[str]):
        self.hostname = hostname.lower().rstrip(".")
        self.ips = ips

    async def resolve(self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET):
        if host.lower().rstrip(".") != self.hostname:
            raise OSError("Unexpected hostname resolution")
        return [
            {
                "hostname": host,
                "host": ip,
                "port": port,
                "family": socket.AF_INET6 if ipaddress.ip_address(ip).version == 6 else socket.AF_INET,
                "proto": 0,
                "flags": socket.AI_NUMERICHOST,
            }
            for ip in self.ips
        ]

    async def close(self):
        pass


def extract_html(html: str, url: str) -> ExtractedSite:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    meta = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    meta_description = meta.get("content", "").strip() if meta else None
    headings = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])][:24]
    structured = [
        tag.get_text(" ", strip=True)[:2500]
        for tag in soup.find_all("script", attrs={"type": "application/ld+json"})[:3]
    ]
    for el in soup(["script", "style", "noscript", "nav", "footer", "header", "svg", "form", "aside"]):
        el.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    raw = root.get_text(" ", strip=True)
    clean = re.sub(r"\s+", " ", raw).strip()
    if not clean:
        raise WebsiteFetchError("Website had no readable text", 422)
    return ExtractedSite(url, title, meta_description, headings, clean[:MAX_TEXT_CHARS], structured, len(clean.split()))


async def fetch_website(url: str) -> ExtractedSite:
    """Resolve -> validate all IPs -> pin validated DNS -> fetch without redirects."""
    try:
        hostname = check_url(url)
        addresses = await resolve_public_addresses(hostname)
    except UnsafeURL as exc:
        raise WebsiteFetchError(str(exc), 422) from exc

    resolver = PinnedPublicResolver(hostname, addresses)
    connector = aiohttp.TCPConnector(resolver=resolver, use_dns_cache=False, limit=2)
    timeout = aiohttp.ClientTimeout(total=13, sock_connect=5, sock_read=8)
    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False) as session:
            async with session.get(url, allow_redirects=False, headers={
                "User-Agent": "DxoraLinkAnalysis/1.0 (+single-page research)",
                "Accept": "text/html,application/xhtml+xml",
            }) as response:
                if 300 <= response.status < 400:
                    raise WebsiteFetchError("Site redirected; submit the destination URL directly", 422)
                if response.status >= 400:
                    raise WebsiteFetchError(f"Website returned HTTP {response.status}", 502)
                mime = response.headers.get("Content-Type", "").lower()
                if not (mime.startswith("text/html") or mime.startswith("application/xhtml+xml")):
                    raise WebsiteFetchError("URL did not return an HTML page", 422)
                if response.content_length and response.content_length > MAX_BYTES:
                    raise WebsiteFetchError("Website HTML is too large to analyze", 422)
                chunks: list[bytes] = []
                received = 0
                async for chunk in response.content.iter_chunked(16_384):
                    received += len(chunk)
                    if received > MAX_BYTES:
                        raise WebsiteFetchError("Website HTML is too large to analyze", 422)
                    chunks.append(chunk)
                charset = response.charset or "utf-8"
                html = b"".join(chunks).decode(charset, errors="replace")
                return extract_html(html, url)
    except WebsiteFetchError:
        raise
    except (aiohttp.ClientError, TimeoutError, UnicodeError, LookupError, OSError) as exc:
        raise WebsiteFetchError("Website could not be fetched securely", 502) from exc
