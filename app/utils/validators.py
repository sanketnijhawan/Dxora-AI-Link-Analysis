"""Defend the website fetcher against common SSRF entry points."""
import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit


class UnsafeURL(ValueError):
    pass


def check_url(url: str) -> str:
    """Validate URL shape and return hostname; network resolution is separate."""
    try:
        parts = urlsplit(url)
        hostname = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise UnsafeURL("Malformed website URL") from exc
    if parts.scheme not in {"http", "https"} or not hostname:
        raise UnsafeURL("A valid http(s) website URL is required")
    if parts.username is not None or parts.password is not None:
        raise UnsafeURL("URLs with credentials are not permitted")
    if port is not None and port != (443 if parts.scheme == "https" else 80):
        raise UnsafeURL("Only standard HTTP/HTTPS ports are permitted")
    host = hostname.rstrip(".").lower()
    if not host or host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".test")):
        raise UnsafeURL("Local/internal hostnames are not permitted")
    return host


def ensure_global_ips(ips: list[str]) -> list[str]:
    if not ips:
        raise UnsafeURL("Hostname did not resolve")
    for raw in ips:
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError as exc:
            raise UnsafeURL("Invalid DNS address") from exc
        if not ip.is_global:
            raise UnsafeURL("Private, local or reserved network addresses are blocked")
    return sorted(set(ips))


async def resolve_public_addresses(hostname: str) -> list[str]:
    try:
        direct_ip = ipaddress.ip_address(hostname)
    except ValueError:
        direct_ip = None
    if direct_ip is not None:
        return ensure_global_ips([str(direct_ip)])
    try:
        answers = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(hostname, None, type=socket.SOCK_STREAM),
            timeout=5.0,
        )
    except (socket.gaierror, asyncio.TimeoutError, OSError) as exc:
        raise UnsafeURL("Website hostname could not be resolved") from exc
    return ensure_global_ips([record[4][0] for record in answers])
