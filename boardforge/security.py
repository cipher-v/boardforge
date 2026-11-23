from __future__ import annotations

import ipaddress
import socket
from urllib.parse import parse_qsl, urlparse


SENSITIVE_QUERY_TERMS = {"access_token", "api_key", "apikey", "auth", "key", "session", "signature", "token"}


class UnsafeUrlError(ValueError):
    pass


def _is_public_ip(value: str) -> bool:
    ip = ipaddress.ip_address(value)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def has_sensitive_query(url: str) -> bool:
    query_names = {name.casefold() for name, _ in parse_qsl(urlparse(url).query, keep_blank_values=True)}
    return bool(query_names & SENSITIVE_QUERY_TERMS)


def validate_public_url(url: str, *, resolve_dns: bool = True) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeUrlError("only HTTP(S) URLs are allowed")
    if not parsed.hostname:
        raise UnsafeUrlError("URL must contain a hostname")
    if parsed.username or parsed.password:
        raise UnsafeUrlError("URLs containing credentials are not allowed")

    hostname = parsed.hostname.rstrip(".").casefold()
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
        raise UnsafeUrlError("local hostnames are not allowed")

    try:
        literal_ip = ipaddress.ip_address(hostname)
    except ValueError:
        literal_ip = None
    if literal_ip is not None and not _is_public_ip(str(literal_ip)):
        raise UnsafeUrlError("private or non-routable addresses are not allowed")

    if resolve_dns:
        try:
            resolved = {item[4][0] for item in socket.getaddrinfo(hostname, parsed.port or 443)}
        except socket.gaierror as exc:
            raise UnsafeUrlError(f"hostname could not be resolved: {hostname}") from exc
        if not resolved or any(not _is_public_ip(address) for address in resolved):
            raise UnsafeUrlError("hostname resolves to a private or non-routable address")
    return url
