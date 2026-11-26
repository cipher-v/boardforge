from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

import httpx

from boardforge.confidence import (
    LEADERSHIP_TITLE_RE,
    SENIORITY_TITLE_RE,
    occupational_query_text,
    parse_query_intent,
)
from boardforge.connectors.detect import detect_source
from boardforge.database import StoreProtocol
from boardforge.enrich import INDIA_CITIES, extract_job_experience, infer_country
from boardforge.matching import ROLE_PATTERNS, rank_jobs
from boardforge.models import CandidateProfile, Preferences, SourceRef
from boardforge.preferences import (
    effective_search_preferences,
    is_entry_level_target,
    is_explicit_senior_query,
    strip_orchestration_instructions,
)
from boardforge.quarantine import is_placeholder_job_title, is_quarantined_source
from boardforge.resume import apply_candidate_context
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
    "AI/ML": ("machine learning engineer", "AI engineer", "LLM engineer"),
    "Computer Vision": ("computer vision engineer", "perception engineer"),
    "Backend": ("backend engineer", "platform engineer"),
    "Software Engineering": ("software engineer", "software developer"),
    "Data Engineering": ("data engineer", "analytics engineer"),
    "Data Science": ("data scientist",),
    "QA": ("QA automation engineer", "SDET", "test engineer"),
}

_INDIA_HUB_QUERY_TERMS = (
    "India",
    "Bengaluru",
    "Bangalore",
    "Hyderabad",
    "Pune",
    "Noida",
    "Gurugram",
)
_EXTRA_INDIA_CITIES = {
    "kolkata",
    "coimbatore",
    "trivandrum",
    "thiruvananthapuram",
    "chandigarh",
    "mysuru",
    "mysore",
}
_NON_TECH_TITLE_RE = re.compile(
    r"\b(?:account\s+executive|sales\s+development|business\s+development|"
    r"recruiter|talent\s+acquisition|human\s+resources|\bhr\b|accountant|"
    r"accounting|payroll|audit|legal\s+counsel|public\s+relations|"
    r"event\s+marketing|partner\s+marketing|executive\s+assistant)\b",
    re.I,
)
_NON_VACANCY_HIT_RE = re.compile(
    r"\b(?:submit\s+(?:a|that)\s+referral|referral\s+here|talent\s+(?:pool|community)|"
    r"general\s+application|not\s+currently\s+posted|no\s+(?:current\s+)?open\s+(?:positions?|roles?))\b",
    re.I,
)
_CRM_SUPPORT_HIT_TITLE_RE = re.compile(
    r"\b(?:salesforce|agentforce|servicenow|workday|netsuite|amazon\s+connect|"
    r"contact\s+center|support\s+engineer|technical\s+support|customer\s+support|"
    r"it\s+support|help\s*desk|qa\s+consultant|solutions?\s+consultant|"
    r"engagement\s+manager|delivery\s+manager)\b",
    re.I,
)
_OFF_TARGET_AI_HIT_TITLE_RE = re.compile(
    r"\b(?:devops|sre|site\s+reliability|security\s+engineer|cybersecurity|"
    r"solutions?\s+engineer|forward[- ]deployed|sales\s+engineer|customer\s+engineer)\b",
    re.I,
)
_VALID_BOARD_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,79}$", re.I)
_RESERVED_BOARD_SLUGS = {
    "embed", "jobs", "apply", "search", "login", "signin", "signup",
    "privacy", "terms", "api", "boards", "careers", "openings",
}
_JUNIOR_SIGNAL_RE = re.compile(
    r"\b(?:entry[- ]level|early[- ]career|junior|jr\.?|freshers?|new\s+grads?|"
    r"recent\s+graduates?|intern(?:ship)?|associate\s+(?:software|backend|ml|ai|data|qa)|"
    r"(?:software|backend|ml|ai|vision|data)\s+engineer\s+i\b|sde[- ]?(?:1|i)\b)\b",
    re.I,
)
_FOREIGN_GEO_MARKERS: dict[str, tuple[str, ...]] = {
    "usa": (
        "united states", "u.s.", "usa", "remote (us)", "remote - us", "us only",
        "new york", "san francisco", "san jose", "san mateo", "seattle", "austin",
        "boston", "chicago", "los angeles", "dallas", "atlanta", "denver",
        "california", "texas", "washington", "massachusetts",
    ),
    "uk": ("united kingdom", "london", "greater london", "remote - uk", "uk only"),
    "canada": ("canada", "toronto", "vancouver", "kitchener", "montreal", "ottawa"),
    "germany": ("germany", "berlin", "munich", "dach"),
    "europe": ("emea", "europe", "france", "paris", "spain", "madrid", "ireland", "dublin", "amsterdam"),
    "latam": ("latam", "mexico", "brazil", "colombia", "argentina"),
    "apac_other": ("singapore", "tokyo", "japan", "australia", "sydney", "melbourne"),
}

MIN_METADATA_DISCOVERY_SCORE = 4.0


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


@dataclass(frozen=True)
class SearchHit:
    url: str
    title: str = ""
    snippet: str = ""
    provider_score: float = 0.0


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
    candidate_results: int = 0
    discovered_boards: int = 0
    skipped_existing_boards: int = 0
    skipped_low_relevance_boards: int = 0
    synced_boards: int = 0
    jobs_fetched: int = 0
    jobs_inserted: int = 0
    jobs_updated: int = 0
    new_matching_jobs: int = 0
    new_recommended_jobs: int = 0
    failed_boards: int = 0
    failures: int = 0
    reason: str = ""
    retryable: bool = False
    error_kind: str = ""
    estimated_cost_usd: float = 0.0
    latency_ms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "searched_queries": self.searched_queries,
            "candidate_results": self.candidate_results,
            "discovered_boards": self.discovered_boards,
            "skipped_existing_boards": self.skipped_existing_boards,
            "skipped_low_relevance_boards": self.skipped_low_relevance_boards,
            "synced_boards": self.synced_boards,
            "failed_boards": self.failed_boards,
            "jobs_fetched": self.jobs_fetched,
            "jobs_inserted": self.jobs_inserted,
            "jobs_updated": self.jobs_updated,
            "new_matching_jobs": self.new_matching_jobs,
            "new_recommended_jobs": self.new_recommended_jobs,
            "failures": self.failures,
            "reason": self.reason,
            "retryable": self.retryable,
            "error_kind": self.error_kind,
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "latency_ms": dict(self.latency_ms),
        }


class TavilyDiscoveryClient:
    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None, *, budget_store=None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.enabled = bool(self.api_key)
        self.budget_store = budget_store

    async def search_results(self, query: str, *, count: int = 20) -> list[SearchHit]:
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
        hits: list[SearchHit] = []
        for item in response.json().get("results", []):
            url = (item.get("url") or "").strip()
            if not url:
                continue
            raw_score = item.get("score")
            try:
                score_val = float(raw_score) if raw_score is not None else 0.0
            except (TypeError, ValueError):
                score_val = 0.0
            hits.append(SearchHit(
                url=url,
                title=str(item.get("title") or ""),
                snippet=str(item.get("content") or item.get("snippet") or ""),
                provider_score=score_val,
            ))
        return hits

    async def search_urls(self, query: str, *, count: int = 20) -> list[str]:
        return [hit.url for hit in await self.search_results(query, count=count)]


class BraveDiscoveryClient:
    """Legacy/optional search provider retained for provider portability."""

    name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("BOARDFORGE_BRAVE_API_KEY", "")
        self.enabled = bool(self.api_key)

    async def search_results(self, query: str, *, count: int = 20) -> list[SearchHit]:
        if not self.enabled:
            return []
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
            SearchHit(
                url=str(item.get("url")),
                title=str(item.get("title") or ""),
                snippet=str(item.get("description") or ""),
            )
            for item in (payload.get("web") or {}).get("results", [])
            if item.get("url")
        ]

    async def search_urls(self, query: str, *, count: int = 20) -> list[str]:
        return [hit.url for hit in await self.search_results(query, count=count)]


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


def _effective_discovery_preferences(
    profile: CandidateProfile,
    prefs: Preferences,
    query: str = "",
) -> Preferences:
    effective = effective_search_preferences(prefs, query) if query.strip() else prefs.model_copy(deep=True)
    if not effective.roles and not query.strip() and profile.target_roles:
        effective.roles = list(profile.target_roles[:4])
    return apply_candidate_context(effective, profile)


def _role_phrases(
    profile: CandidateProfile,
    prefs: Preferences,
    query: str = "",
) -> list[str]:
    roles = prefs.roles or profile.target_roles or profile.role_signals
    low_q = query.casefold()
    phrases: list[str] = []
    for role in roles[:4]:
        candidates = ROLE_TERMS.get(role, (role,))
        if low_q:
            matched = [
                phrase for phrase in candidates
                if any(
                    tok in low_q
                    for tok in phrase.casefold().split()
                    if tok not in {"engineer", "developer", "scientist"}
                )
            ]
            phrases.extend(matched or candidates[:2])
        else:
            phrases.extend(candidates)
    return list(dict.fromkeys(phrases))[:6]


def build_discovery_query(
    profile: CandidateProfile,
    prefs: Preferences,
    query: str = "",
) -> str:
    """Build a focused job-posting search query aligned with domain, geography, and seniority."""
    raw_query = strip_orchestration_instructions(query).strip()
    effective = effective_search_preferences(prefs, raw_query) if raw_query else prefs

    if not raw_query:
        phrases = _role_phrases(profile, effective)
        role_clause = " OR ".join(f'"{phrase}"' for phrase in phrases) or '"software engineer"'
        geo_parts = list(effective.locations[:3]) + list(effective.countries[:2])
        geo = " ".join(dict.fromkeys(part.strip() for part in geo_parts if part.strip()))
        return f"{role_clause} {geo}".strip()

    phrases = _role_phrases(profile, effective, raw_query)
    intent_info = parse_query_intent(effective, raw_query)
    specific_terms = sorted(intent_info.skill_terms | intent_info.domain_terms | intent_info.task_terms)[:4]

    if phrases:
        role_clause = (
            f'"{phrases[0]}"'
            if len(phrases) == 1
            else "(" + " OR ".join(f'"{phrase}"' for phrase in phrases[:4]) + ")"
        )
    else:
        cleaned = occupational_query_text(raw_query)
        for term in [*effective.countries, *effective.locations]:
            if term:
                cleaned = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", " ", cleaned, flags=re.I)
        cleaned = " ".join(cleaned.split())
        role_clause = cleaned or '"software engineer"'

    parts: list[str] = [role_clause]
    if specific_terms:
        existing_low = role_clause.casefold()
        extra_terms = [t for t in specific_terms if t.casefold() not in existing_low]
        if extra_terms:
            parts.append(" ".join(extra_terms[:3]))

    # Geography clause: expand India with major tech hubs so city-only ATS postings match too.
    loc_terms: list[str] = []
    for loc in effective.locations[:3]:
        loc_clean = loc.strip()
        if not loc_clean:
            continue
        loc_terms.append(loc_clean)
        if loc_clean.casefold() == "bengaluru":
            loc_terms.append("Bangalore")
        elif loc_clean.casefold() == "gurugram":
            loc_terms.append("Gurgaon")

    countries_low = {c.casefold() for c in effective.countries}
    if loc_terms:
        dedup_locs = list(dict.fromkeys(loc_terms))
        loc_expr = (
            f'"{dedup_locs[0]}"'
            if len(dedup_locs) == 1
            else "(" + " OR ".join(f'"{loc}"' for loc in dedup_locs) + ")"
        )
        country_suffix = " ".join(
            f'"{c}"' for c in effective.countries[:2] if c.casefold() not in {x.casefold() for x in dedup_locs}
        )
        parts.append(f"{loc_expr} {country_suffix}".strip())
    elif "india" in countries_low:
        hubs = " OR ".join(f'"{term}"' for term in _INDIA_HUB_QUERY_TERMS)
        parts.append(f"({hubs})")
    elif effective.countries:
        parts.append(" ".join(f'"{c}"' for c in effective.countries[:2]))

    # Seniority clause targeting individual job postings rather than generic career portals.
    if is_entry_level_target(effective, raw_query):
        parts.append('("entry level" OR "junior" OR "0-2 years" OR "graduate")')
    elif is_explicit_senior_query(raw_query):
        for token in ("principal", "staff", "senior", "lead"):
            if re.search(rf"\b{token}\b", raw_query, re.I):
                parts.append(f'"{token}"')
                break

    return " ".join(part for part in parts if part).strip()


def _valid_supported_ats_source(url: str) -> SourceRef | None:
    try:
        host = urlparse(url).netloc.lower()
        if not any(
            host == allowed or host.endswith("." + allowed)
            for allowed in SUPPORTED_SEARCH_HOSTS
        ):
            return None
        source = detect_source(url)
        if source.kind not in {"greenhouse", "lever", "ashby"} or not source.key:
            return None
        slug = source.key.strip()
        if slug.casefold() in _RESERVED_BOARD_SLUGS or not _VALID_BOARD_SLUG_RE.match(slug):
            return None
        return source
    except Exception:
        return None


def is_individual_posting_url(url: str, source: SourceRef | None = None) -> bool:
    """Return True when the URL points to a specific job posting rather than a board root."""
    resolved = source or _valid_supported_ats_source(url)
    if resolved is None:
        return False
    parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    if resolved.kind == "greenhouse":
        return len(parts) >= 3 and parts[1].casefold() == "jobs"
    if resolved.kind in {"lever", "ashby"}:
        return len(parts) >= 2 and parts[1].casefold() not in {"jobs", "openings", "careers"}
    return False


def canonical_sources(urls: list[str]) -> list[SourceRef]:
    seen: set[tuple[str, str]] = set()
    sources: list[SourceRef] = []
    for url in urls:
        source = _valid_supported_ats_source(url)
        if source is None:
            continue
        marker = (source.kind, source.key.casefold())
        if marker in seen:
            continue
        seen.add(marker)
        sources.append(source)
    return sources


def _matches_india_text(low_text: str) -> bool:
    if re.search(r"\bindia\b", low_text):
        return True
    return any(
        re.search(rf"\b{re.escape(city)}\b", low_text)
        for city in (INDIA_CITIES | _EXTRA_INDIA_CITIES)
    )


def _has_foreign_geo_conflict(low_text: str, allowed_countries: set[str]) -> bool:
    for region_key, markers in _FOREIGN_GEO_MARKERS.items():
        if region_key == "usa" and ("usa" in allowed_countries or "united states" in allowed_countries):
            continue
        if region_key == "uk" and ("united kingdom" in allowed_countries or "uk" in allowed_countries):
            continue
        if region_key == "canada" and "canada" in allowed_countries:
            continue
        if region_key == "germany" and "germany" in allowed_countries:
            continue
        if any(re.search(rf"(?<!\w){re.escape(m)}(?!\w)", low_text) for m in markers):
            return True
    return False


def _matches_requested_geography(
    hit: SearchHit,
    effective: Preferences,
) -> tuple[bool, float]:
    if not effective.locations and not effective.countries:
        return True, 0.0

    combined = f"{hit.title} {hit.snippet}"
    low = combined.casefold()
    allowed_countries = {c.casefold() for c in effective.countries}
    if effective.locations:
        for loc in effective.locations:
            inferred = infer_country(loc)
            if inferred:
                allowed_countries.add(inferred.casefold())

    loc_matched = False
    for loc in effective.locations:
        aliases = {loc.casefold()}
        if loc.casefold() == "bengaluru":
            aliases.add("bangalore")
        elif loc.casefold() == "gurugram":
            aliases.add("gurgaon")
        if any(re.search(rf"\b{re.escape(alias)}\b", low) for alias in aliases):
            loc_matched = True
            break

    country_matched = False
    if "india" in allowed_countries and _matches_india_text(low):
        country_matched = True
    for country in allowed_countries:
        if country != "india" and re.search(rf"(?<!\w){re.escape(country)}(?!\w)", low):
            country_matched = True
    if not country_matched:
        inferred_title = infer_country(hit.title).casefold()
        if inferred_title and inferred_title in allowed_countries:
            country_matched = True

    if effective.locations:
        if loc_matched:
            return True, 3.5
        if effective.remote_ok and country_matched and re.search(r"\bremote\b", low):
            return True, 2.5
        return False, -4.0

    # Country constraint without city constraint:
    if country_matched:
        return True, 3.0
    if _has_foreign_geo_conflict(low, allowed_countries):
        return False, -4.0
    return False, -2.5


def _matches_occupational_intent(
    hit: SearchHit,
    effective: Preferences,
    query: str,
) -> tuple[bool, float]:
    title_low = hit.title.casefold()
    snippet_low = hit.snippet.casefold()
    combined_low = f"{title_low} {snippet_low}"
    clean_q = strip_orchestration_instructions(query).casefold()

    if _NON_VACANCY_HIT_RE.search(combined_low) or (hit.title.strip() and is_placeholder_job_title(hit.title)):
        return False, -5.0

    tech_roles_requested = bool(
        set(effective.roles)
        & {"AI/ML", "Computer Vision", "Backend", "Software Engineering", "Data Engineering", "Data Science", "QA"}
    )
    if tech_roles_requested and _NON_TECH_TITLE_RE.search(title_low):
        return False, -4.0

    core_eng_requested = bool(
        set(effective.roles)
        & {"AI/ML", "Computer Vision", "Backend", "Software Engineering", "Data Engineering", "Data Science"}
    )
    if (
        core_eng_requested
        and _CRM_SUPPORT_HIT_TITLE_RE.search(title_low)
        and not _CRM_SUPPORT_HIT_TITLE_RE.search(clean_q)
    ):
        return False, -4.0

    # If the user specifically requested Computer Vision and/or Machine Learning (not generic AI/Agentforce),
    # do not let incidental company boilerplate mentioning "AI" or Salesforce/DevOps/Security titles qualify a board.
    ml_cv_specifically_requested = bool(
        re.search(r"\b(?:computer\s+vision|machine\s+learning|deep\s+learning|\bml\b)\b", clean_q)
        and not re.search(r"\b(?:ai\s+agents?|agentic|genai|generative\s+ai|llm|rag)\b", clean_q)
    )
    if ml_cv_specifically_requested:
        if _OFF_TARGET_AI_HIT_TITLE_RE.search(title_low) and not _OFF_TARGET_AI_HIT_TITLE_RE.search(clean_q):
            return False, -3.5
        if not re.search(
            r"\b(?:computer\s+vision|vision\s+engineer|perception|image\s+processing|opencv|"
            r"machine\s+learning|ml\s+engineer|deep\s+learning|pytorch|tensorflow|neural)\b",
            combined_low,
        ):
            return False, -3.5

    score = 0.0
    matched_role = False
    for role in effective.roles:
        if role in {"AI/ML", "Computer Vision"} and _OFF_TARGET_AI_HIT_TITLE_RE.search(title_low) and not _OFF_TARGET_AI_HIT_TITLE_RE.search(clean_q):
            continue
        patterns = ROLE_PATTERNS.get(role, ())
        if any(re.search(p, title_low, re.I) for p in patterns):
            score += 3.0
            matched_role = True
        elif any(re.search(p, snippet_low, re.I) for p in patterns):
            score += 1.5
            matched_role = True
        else:
            for phrase in ROLE_TERMS.get(role, ()):
                if phrase.casefold() in combined_low:
                    score += 2.0 if phrase.casefold() in title_low else 1.2
                    matched_role = True
                    break

    intent = parse_query_intent(effective, clean_q) if clean_q.strip() else None
    specific = set(intent.specific_terms) if intent else set()
    matched_specific = [
        term for term in specific
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", combined_low)
    ]
    if matched_specific:
        score += min(2.5, 0.9 * len(matched_specific))

    if (effective.roles or specific) and not matched_role and not matched_specific:
        # Allow generic "Software Engineer" title only when technical engineering work is evident in snippet.
        if (
            tech_roles_requested
            and re.search(r"\bsoftware\s+(?:engineer|developer)\b", title_low)
            and re.search(r"\b(?:python|java|golang|c\+\+|pytorch|tensorflow|backend|apis?|distributed|models?)\b", snippet_low)
        ):
            score += 1.0
        else:
            return False, -3.0

    return True, score


def score_discovery_hit(
    hit: SearchHit,
    source: SourceRef,
    effective: Preferences,
    query: str = "",
) -> tuple[bool, float]:
    """Deterministically evaluate a single search hit before canonicalizing to its board."""
    posting_url = is_individual_posting_url(hit.url, source)
    has_metadata = bool(hit.title.strip() or hit.snippet.strip())

    if not has_metadata:
        # URL-only fallback (e.g., minimal test doubles implementing search_urls only):
        # accept individual job posting URLs, reject bare career-portal roots when constrained.
        if posting_url:
            return True, 2.5
        if effective.countries or effective.locations or effective.roles:
            return False, 0.0
        return True, 1.0

    geo_ok, geo_score = _matches_requested_geography(hit, effective)
    if not geo_ok:
        return False, geo_score

    occ_ok, occ_score = _matches_occupational_intent(hit, effective, query)
    if not occ_ok:
        return False, occ_score

    entry_target = is_entry_level_target(effective, query)
    if entry_target and (
        SENIORITY_TITLE_RE.search(hit.title) or LEADERSHIP_TITLE_RE.search(hit.title)
    ):
        return False, -3.0

    min_exp, _ = extract_job_experience(hit.title, hit.snippet)
    if (
        effective.max_required_experience_years is not None
        and min_exp is not None
        and min_exp > effective.max_required_experience_years
    ):
        return False, -3.0

    score = geo_score + occ_score + (2.0 if posting_url else -1.0)
    if entry_target:
        if _JUNIOR_SIGNAL_RE.search(f"{hit.title} {hit.snippet}") or (
            min_exp is not None and min_exp <= 1.0
        ):
            score += 2.0

    score += min(0.5, max(0.0, float(hit.provider_score)) * 0.5)
    min_threshold = MIN_METADATA_DISCOVERY_SCORE if (
        effective.countries or effective.locations or effective.roles
    ) else 1.5
    if not posting_url and score < (min_threshold + 1.5):
        return False, score
    return score >= min_threshold, round(score, 4)


def _indexed_source_timestamps(store: StoreProtocol) -> dict[tuple[str, str], datetime | None]:
    """Collect already-indexed boards and their latest sync/seen timestamps from the store."""
    indexed: dict[tuple[str, str], datetime | None] = {}
    for job in store.active_jobs():
        marker = (job.source_kind, job.source_key.casefold())
        seen_at = job.last_seen_at
        if seen_at is not None and seen_at.tzinfo is None:
            seen_at = seen_at.replace(tzinfo=timezone.utc)
        prev = indexed.get(marker)
        if marker not in indexed or (seen_at is not None and (prev is None or seen_at > prev)):
            indexed[marker] = seen_at

    for row in store.discovered_sources():
        kind = str(row.get("source_kind") or "")
        key = str(row.get("source_key") or "").casefold()
        if not kind or not key:
            continue

        def _parse_ts(raw_val) -> datetime | None:
            if isinstance(raw_val, datetime):
                return raw_val if raw_val.tzinfo else raw_val.replace(tzinfo=timezone.utc)
            if isinstance(raw_val, str) and raw_val.strip():
                try:
                    parsed = datetime.fromisoformat(raw_val)
                    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
                except ValueError:
                    return None
            return None

        synced_at = _parse_ts(row.get("last_synced"))
        status = str(row.get("last_status") or "")
        if synced_at is not None or status == "synced":
            prev = indexed.get((kind, key))
            if (kind, key) not in indexed or (synced_at is not None and (prev is None or synced_at > prev)):
                indexed[(kind, key)] = synced_at
        elif status == "failed":
            # Dead/unavailable boards (e.g. 404 ATS slugs) use last_discovered for cooldown
            # so repeated discoveries do not keep wasting slots on the same broken board.
            failed_at = _parse_ts(row.get("last_discovered"))
            if failed_at is not None and (kind, key) not in indexed:
                indexed[(kind, key)] = failed_at
    return indexed


def _is_source_explicitly_requested(source: SourceRef, query: str) -> bool:
    if not query.strip():
        return False
    low = query.casefold()
    if source.board_url.casefold() in low:
        return True
    key = source.key.casefold()
    return len(key) >= 3 and bool(re.search(rf"(?<!\w){re.escape(key)}(?!\w)", low))


def prioritize_discovery_sources(
    hits: list[SearchHit],
    *,
    store: StoreProtocol | None,
    profile: CandidateProfile,
    prefs: Preferences,
    query: str = "",
    max_boards: int = 6,
    fresh_source_seconds: int | None = None,
) -> tuple[list[SourceRef], int, int]:
    """Rank and filter candidate ATS boards using hit metadata and store freshness.

    Returns (selected_sources, skipped_existing_boards, skipped_low_relevance_boards).
    """
    effective = _effective_discovery_preferences(profile, prefs, query)
    indexed = _indexed_source_timestamps(store) if store is not None else {}
    fresh_ttl = (
        fresh_source_seconds
        if fresh_source_seconds is not None
        else max(300, int(os.getenv("BOARDFORGE_DISCOVERY_FRESH_SOURCE_SECONDS", "86400")))
    )
    now = datetime.now(timezone.utc)

    board_best: dict[tuple[str, str], tuple[SourceRef, float, int, int]] = {}
    board_rejected_low: set[tuple[str, str]] = set()
    skipped_existing: set[tuple[str, str]] = set()

    for order_idx, hit in enumerate(hits):
        source = _valid_supported_ats_source(hit.url)
        if source is None:
            continue
        marker = (source.kind, source.key.casefold())
        if is_quarantined_source(source.kind, source.key):
            board_rejected_low.add(marker)
            continue
        explicit_board = _is_source_explicitly_requested(source, query)

        is_indexed = marker in indexed
        if is_indexed and not explicit_board:
            last_ts = indexed[marker]
            is_recent = last_ts is None or (now - last_ts).total_seconds() < fresh_ttl
            if is_recent:
                skipped_existing.add(marker)
                continue

        relevant, hit_score = score_discovery_hit(hit, source, effective, query)
        if not relevant and not explicit_board:
            board_rejected_low.add(marker)
            continue

        # Prefer previously unindexed boards over stale already-indexed boards.
        unindexed_bonus = 2.5 if not is_indexed else 0.0
        total_score = hit_score + unindexed_bonus
        existing_entry = board_best.get(marker)
        if existing_entry is None:
            board_best[marker] = (source, total_score, 1, order_idx)
        else:
            prev_source, prev_score, count, first_idx = existing_entry
            board_best[marker] = (
                prev_source,
                max(prev_score, total_score),
                count + 1,
                first_idx,
            )

    skipped_low_count = len(board_rejected_low - set(board_best) - skipped_existing)
    ranked = sorted(
        board_best.values(),
        key=lambda item: (
            -(item[1] + min(1.5, 0.5 * (item[2] - 1))),
            item[3],
        ),
    )
    selected = [item[0] for item in ranked[: max(1, max_boards)]]
    return selected, len(skipped_existing), skipped_low_count


async def discover_and_sync(
    store: StoreProtocol,
    profile: CandidateProfile,
    prefs: Preferences,
    *,
    query: str = "",
    profile_id: str = "",
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

    total_started = time.perf_counter()
    clean_query = strip_orchestration_instructions(query)
    effective_prefs = _effective_discovery_preferences(profile, prefs, clean_query)
    rejected_ids = store.rejected_job_ids(profile_id) if profile_id else set()
    before_matches = rank_jobs(
        store.active_jobs(),
        effective_prefs,
        query=clean_query,
        top_k=50,
        rejected_ids=rejected_ids,
    )
    before_match_ids = {m.job.stable_id for m in before_matches}
    before_recommended_ids = {
        m.job.stable_id for m in before_matches if m.status == "recommended"
    }

    discovery_query = build_discovery_query(profile, prefs, clean_query)
    report = DiscoveryReport(
        enabled=True,
        provider=provider_name,
        searched_queries=1,
        estimated_cost_usd=(
            float(os.getenv("BOARDFORGE_TAVILY_COST_PER_SEARCH_USD", "0"))
            if provider_name == "tavily"
            else 0.0
        ),
    )

    search_started = time.perf_counter()
    hits: list[SearchHit] = []
    try:
        if hasattr(client, "search_results"):
            hits = await client.search_results(discovery_query, count=bounded_results)
        else:
            urls = await client.search_urls(discovery_query, count=bounded_results)
            hits = [SearchHit(url=url) for url in urls if url]
    except SearchProviderError as exc:
        report.failures += 1
        report.retryable = exc.retryable
        report.error_kind = type(exc).__name__
        report.reason = str(exc)
    except Exception as exc:
        report.failures += 1
        report.retryable = True
        report.error_kind = type(exc).__name__
        report.reason = str(exc)
    search_ms = (time.perf_counter() - search_started) * 1000

    report.candidate_results = len(hits)
    # Keep up to bounded_boards + 2 strictly eligible candidates so a dead 404 board
    # can fall through to the next eligible board from the same search without touching low-relevance boards.
    candidate_pool_limit = min(10, bounded_boards + 2)
    eligible_sources, skipped_existing, skipped_low = prioritize_discovery_sources(
        hits,
        store=store,
        profile=profile,
        prefs=prefs,
        query=clean_query,
        max_boards=candidate_pool_limit,
    )
    report.discovered_boards = min(len(eligible_sources), bounded_boards)
    report.skipped_existing_boards = skipped_existing
    report.skipped_low_relevance_boards = skipped_low

    if not eligible_sources and not report.failures:
        if skipped_existing and skipped_low:
            report.reason = (
                "no new matching job boards found; skipped already-indexed and "
                "non-matching international or senior sources"
            )
        elif skipped_existing:
            report.reason = "matching job boards are already indexed and recently synchronized"
        elif skipped_low:
            report.reason = "search results did not match requested geography, role, or seniority"
        else:
            report.reason = "no supported official job boards were returned"
        report.latency_ms = {
            "search_ms": round(search_ms, 2),
            "sync_ms": 0.0,
            "ranking_ms": 0.0,
            "total_ms": round((time.perf_counter() - total_started) * 1000, 2),
        }
        return report

    provenance = (
        "manual:"
        + hashlib.sha256(
            " ".join(clean_query.casefold().split()).encode("utf-8")
        ).hexdigest()[:12]
        if clean_query
        else "profile-derived"
    )
    semaphore = asyncio.Semaphore(sync_concurrency)

    async def sync_source(source: SourceRef) -> bool:
        async with semaphore:
            store.record_discovered_source(
                source.kind, source.key, source.board_url, provenance
            )
            try:
                summary = await sync_url(source.board_url, store)
                report.synced_boards += 1
                report.jobs_fetched += summary.fetched
                report.jobs_inserted += summary.inserted
                report.jobs_updated += summary.updated
                store.record_discovered_source(
                    source.kind,
                    source.key,
                    source.board_url,
                    provenance,
                    synced=True,
                    status="synced",
                )
                return True
            except Exception:
                report.failed_boards += 1
                report.failures += 1
                store.record_discovered_source(
                    source.kind,
                    source.key,
                    source.board_url,
                    provenance,
                    status="failed",
                )
                return False

    sync_started = time.perf_counter()
    primary_batch = eligible_sources[:bounded_boards]
    backup_batch = eligible_sources[bounded_boards:]
    if primary_batch:
        await asyncio.gather(*(sync_source(source) for source in primary_batch))
    for backup_source in backup_batch:
        if report.synced_boards >= bounded_boards:
            break
        await sync_source(backup_source)
    report.discovered_boards = report.synced_boards + report.failed_boards
    sync_ms = (time.perf_counter() - sync_started) * 1000

    ranking_started = time.perf_counter()
    if report.synced_boards > 0:
        after_matches = rank_jobs(
            store.active_jobs(),
            effective_prefs,
            query=clean_query,
            top_k=50,
            rejected_ids=rejected_ids,
        )
        report.new_matching_jobs = sum(
            1 for m in after_matches if m.job.stable_id not in before_match_ids
        )
        report.new_recommended_jobs = sum(
            1
            for m in after_matches
            if m.status == "recommended" and m.job.stable_id not in before_recommended_ids
        )
        if report.new_matching_jobs == 0 and not report.reason:
            report.reason = (
                "synchronized board(s) had no active postings matching the requested "
                "role, location, and experience constraints"
            )
    elif report.failed_boards > 0 and not report.reason:
        report.reason = (
            f"{report.failed_boards} candidate job board(s) failed synchronization "
            "(unavailable or delisted ATS board)"
        )
    ranking_ms = (time.perf_counter() - ranking_started) * 1000
    report.latency_ms = {
        "search_ms": round(search_ms, 2),
        "sync_ms": round(sync_ms, 2),
        "ranking_ms": round(ranking_ms, 2),
        "total_ms": round((time.perf_counter() - total_started) * 1000, 2),
    }

    return report
