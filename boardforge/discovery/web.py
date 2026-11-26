from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

import httpx

from boardforge.connectors.detect import detect_source
from boardforge.database import StoreProtocol
from boardforge.models import CandidateProfile, Preferences, SourceRef
from boardforge.sync import sync_url


SUPPORTED_SEARCH_HOSTS = (
    "job-boards.greenhouse.io",
    "boards.greenhouse.io",
    "jobs.lever.co",
    "jobs.ashbyhq.com",
)

TAVILY_DOMAINS = (
    "job-boards.greenhouse.io",
    "boards.greenhouse.io",
    "jobs.lever.co",
    "jobs.ashbyhq.com",
)

ROLE_TERMS: dict[str, tuple[str, ...]] = {
    "AI/ML": ("AI engineer", "machine learning engineer", "LLM engineer"),
    "Computer Vision": ("computer vision engineer", "perception engineer"),
    "Backend": ("backend engineer", "platform engineer"),
    "Software Engineering": ("software engineer",),
    "Data Engineering": ("data engineer",),
    "Data Science": ("data scientist",),
}


class SearchProviderError(RuntimeError):
    retryable = False

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class SearchProviderAuthError(SearchProviderError):
    retryable = False


class SearchProviderRateLimitError(SearchProviderError):
    retryable = True


class SearchProviderTransientError(SearchProviderError):
    retryable = True


class SearchProviderRequestError(SearchProviderError):
    retryable = False


def _provider_http_error(provider: str, response: httpx.Response) -> None:
    status = response.status_code
    if status in {401, 403}:
        raise SearchProviderAuthError(
            f"{provider} authentication failed ({status})", status_code=status
        )
    if status == 429:
        raise SearchProviderRateLimitError(
            f"{provider} rate limit reached ({status})", status_code=status
        )
    if 500 <= status:
        raise SearchProviderTransientError(
            f"{provider} server error ({status})", status_code=status
        )
    if 400 <= status:
        raise SearchProviderRequestError(
            f"{provider} request failed ({status})", status_code=status
        )


class SearchProvider(Protocol):
    enabled: bool
    name: str

    async def search_urls(self, query: str, *, count: int = 20) -> list[str]:
        ...


@dataclass
class DiscoveryReport:
    enabled: bool
    provider: str = ""
    searched_queries: int = 0
    discovered_boards: int = 0
    synced_boards: int = 0
    jobs_fetched: int = 0
    failures: int = 0
    reason: str = ""
    retryable: bool = False
    error_kind: str = ""

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "searched_queries": self.searched_queries,
            "discovered_boards": self.discovered_boards,
            "synced_boards": self.synced_boards,
            "jobs_fetched": self.jobs_fetched,
            "failures": self.failures,
            "reason": self.reason,
            "retryable": self.retryable,
            "error_kind": self.error_kind,
        }


class TavilyDiscoveryClient:
    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None, *, budget_store=None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.enabled = bool(self.api_key)
        self.budget_store = budget_store

    async def search_urls(self, query: str, *, count: int = 20) -> list[str]:
        if not self.enabled:
            return []
        payload = {
            "query": query,
            "search_depth": "basic",
            "topic": "general",
            "max_results": max(1, min(count, 20)),
            "include_domains": list(TAVILY_DOMAINS),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        reservation_id = None
        estimated_cost = float(os.getenv("BOARDFORGE_TAVILY_COST_PER_SEARCH_USD", "0"))
        if self.budget_store is not None:
            reservation_id = str(uuid.uuid4())
            admission = self.budget_store.reserve_llm_usage(
                "tavily", "basic-search", reservation_id, 1,
                tpm_limit=int(os.getenv("BOARDFORGE_TAVILY_TPM_LIMIT", "60")),
                tpd_limit=int(os.getenv("BOARDFORGE_TAVILY_DAILY_LIMIT", "900")),
                reserved_cost_usd=estimated_cost,
                cumulative_cost_limit_usd=float(os.getenv(
                    "BOARDFORGE_CUMULATIVE_SPEND_LIMIT_USD", "2.0"
                )),
            )
            if not admission["allowed"]:
                raise SearchProviderRequestError(
                    f"tavily budget denied: {admission['reason']}"
                )
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if reservation_id:
                self.budget_store.finalize_llm_usage(
                    reservation_id, status="failed", estimated_cost_usd=estimated_cost
                )
            raise SearchProviderTransientError(f"tavily network failure: {exc}") from exc
        if reservation_id:
            self.budget_store.finalize_llm_usage(
                reservation_id, status="completed", input_tokens=1, total_tokens=1,
                estimated_cost_usd=estimated_cost,
            )
        _provider_http_error(self.name, response)
        return [
            item.get("url", "")
            for item in response.json().get("results", [])
            if item.get("url")
        ]


class BraveDiscoveryClient:
    """Legacy/optional search provider retained for provider portability."""

    name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("BOARDFORGE_BRAVE_API_KEY", "")
        self.enabled = bool(self.api_key)

    async def search_urls(self, query: str, *, count: int = 20) -> list[str]:
        if not self.enabled:
            return []
        # Brave lacks Tavily's multi-domain request body, so restrict in-query.
        sites = " OR ".join(f"site:{domain}" for domain in TAVILY_DOMAINS)
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }
        params = {
            "q": f"({sites}) {query}",
            "count": max(1, min(count, 20)),
            "search_lang": "en",
            "safesearch": "moderate",
        }
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                response = await client.get(self.endpoint, headers=headers, params=params)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise SearchProviderTransientError(f"brave network failure: {exc}") from exc
        _provider_http_error(self.name, response)
        payload = response.json()
        return [
            item.get("url", "")
            for item in (payload.get("web") or {}).get("results", [])
            if item.get("url")
        ]


class DisabledSearchProvider:
    name = "disabled"
    enabled = False

    async def search_urls(self, query: str, *, count: int = 20) -> list[str]:
        return []


def create_search_provider(budget_store=None) -> SearchProvider:
    provider = os.getenv("BOARDFORGE_SEARCH_PROVIDER", "tavily").strip().casefold()
    if provider == "tavily":
        return TavilyDiscoveryClient(budget_store=budget_store)
    if provider == "brave":
        return BraveDiscoveryClient()
    if provider in {"none", "disabled", "off"}:
        return DisabledSearchProvider()
    raise ValueError(
        "unsupported BOARDFORGE_SEARCH_PROVIDER; expected tavily, brave, or none"
    )


def _role_phrases(profile: CandidateProfile, prefs: Preferences) -> list[str]:
    roles = prefs.roles or profile.target_roles or profile.role_signals
    phrases: list[str] = []
    for role in roles[:4]:
        phrases.extend(ROLE_TERMS.get(role, (role,)))
    return list(dict.fromkeys(phrases))[:6]


def build_discovery_query(
    profile: CandidateProfile,
    prefs: Preferences,
    query: str = "",
) -> str:
    intent = query.strip()
    if not intent:
        phrases = _role_phrases(profile, prefs)
        intent = " OR ".join(f'"{phrase}"' for phrase in phrases) or '"software engineer"'

    geo_parts = list(prefs.locations[:3]) + list(prefs.countries[:2])
    geo = " ".join(dict.fromkeys(part.strip() for part in geo_parts if part.strip()))
    if geo:
        intent = f"{intent} {geo}"
    return intent


def canonical_sources(urls: list[str]) -> list[SourceRef]:
    seen: set[tuple[str, str]] = set()
    sources: list[SourceRef] = []
    for url in urls:
        try:
            host = urlparse(url).netloc.lower()
            if not any(
                host == allowed or host.endswith("." + allowed)
                for allowed in SUPPORTED_SEARCH_HOSTS
            ):
                continue
            source = detect_source(url)
            if source.kind not in {"greenhouse", "lever", "ashby"} or not source.key:
                continue
            marker = (source.kind, source.key.casefold())
            if marker in seen:
                continue
            seen.add(marker)
            sources.append(source)
        except Exception:
            continue
    return sources


async def discover_and_sync(
    store: StoreProtocol,
    profile: CandidateProfile,
    prefs: Preferences,
    *,
    query: str = "",
    max_boards: int | None = None,
    client: SearchProvider | None = None,
) -> DiscoveryReport:
    client = client or create_search_provider(store)
    provider_name = getattr(client, "name", type(client).__name__)
    if not client.enabled:
        return DiscoveryReport(
            enabled=False,
            provider=provider_name,
            reason=(
                "web discovery is disabled or its API key is missing; "
                "search uses the existing indexed jobs"
            ),
        )

    configured_boards = (
        max_boards
        if max_boards is not None
        else int(os.getenv("BOARDFORGE_DISCOVERY_MAX_BOARDS", "6"))
    )
    bounded_boards = max(1, min(8, configured_boards))
    bounded_results = max(
        1, min(20, int(os.getenv("BOARDFORGE_DISCOVERY_MAX_RESULTS", "15")))
    )
    sync_concurrency = max(
        1, min(4, int(os.getenv("BOARDFORGE_DISCOVERY_SYNC_CONCURRENCY", "3")))
    )

    discovery_query = build_discovery_query(profile, prefs, query)
    report = DiscoveryReport(
        enabled=True,
        provider=provider_name,
        searched_queries=1,
    )

    try:
        urls = await client.search_urls(discovery_query, count=bounded_results)
    except SearchProviderError as exc:
        report.failures += 1
        report.retryable = exc.retryable
        report.error_kind = type(exc).__name__
        report.reason = str(exc)
        urls = []
    except Exception as exc:
        report.failures += 1
        report.retryable = True
        report.error_kind = type(exc).__name__
        report.reason = str(exc)
        urls = []

    sources = canonical_sources(urls)[:bounded_boards]
    report.discovered_boards = len(sources)
    provenance = (
        "manual:"
        + hashlib.sha256(
            " ".join(query.casefold().split()).encode("utf-8")
        ).hexdigest()[:12]
        if query
        else "profile-derived"
    )
    semaphore = asyncio.Semaphore(sync_concurrency)

    async def sync_source(source: SourceRef) -> None:
        async with semaphore:
            store.record_discovered_source(
                source.kind, source.key, source.board_url, provenance
            )
            try:
                summary = await sync_url(source.board_url, store)
                report.synced_boards += 1
                report.jobs_fetched += summary.fetched
                store.record_discovered_source(
                    source.kind,
                    source.key,
                    source.board_url,
                    provenance,
                    synced=True,
                    status="synced",
                )
            except Exception:
                report.failures += 1
                store.record_discovered_source(
                    source.kind,
                    source.key,
                    source.board_url,
                    provenance,
                    status="failed",
                )

    await asyncio.gather(*(sync_source(source) for source in sources))
    return report
