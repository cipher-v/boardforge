from __future__ import annotations

import re

import pycountry

from boardforge.models import Job, RemotePolicy


INDIA_CITIES = {
    "ahmedabad", "bangalore", "bengaluru", "chennai", "delhi", "gurgaon",
    "gurugram", "hyderabad", "indore", "jaipur", "kochi", "mumbai",
    "new delhi", "noida", "pune",
}
US_CITIES = {
    "austin", "boston", "new york", "pittsburgh", "san francisco",
    "san mateo", "seattle",
}
US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI",
    "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
    "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
    "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}
US_STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming", "district of columbia",
}
COUNTRY_ALIASES = {
    "south korea": "Korea, Republic of",
    "north korea": "Korea, Democratic People's Republic of",
    "russia": "Russian Federation",
    "vietnam": "Viet Nam",
    "taiwan": "Taiwan, Province of China",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "uae": "United Arab Emirates",
}


def _explicit_countries(location: str) -> list[str]:
    normalized = " ".join(location.casefold().split())
    found: list[str] = []

    for alias, canonical in COUNTRY_ALIASES.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized):
            found.append(canonical)

    for country in pycountry.countries:
        names = {
            country.name,
            getattr(country, "official_name", ""),
            getattr(country, "common_name", ""),
        }
        for name in names:
            if not name:
                continue
            if re.search(
                rf"(?<!\w){re.escape(name.casefold())}(?!\w)",
                normalized,
            ):
                found.append(country.name)
                break

    return list(dict.fromkeys(found))


def infer_country(location: str) -> str:
    """Infer one country conservatively from the structured location only.

    Multi-country locations intentionally remain unknown because Job.country is
    currently a single-valued field.
    """
    normalized = " ".join(location.lower().split())
    if not normalized:
        return ""
    if re.search(r"\bindia\b", normalized) or any(
        re.search(rf"\b{re.escape(city)}\b", normalized) for city in INDIA_CITIES
    ):
        return "India"
    if re.search(r"\b(usa|u\.s\.a\.?|united states|us)\b", normalized) or any(
        re.search(rf"\b{re.escape(city)}\b", normalized) for city in US_CITIES
    ):
        return "USA"
    for code in US_STATE_CODES:
        if re.search(rf",\s*{code}(?:\s|,|$)", location, re.I):
            return "USA"
    if any(
        re.search(rf"\b{re.escape(state)}\b", normalized)
        for state in US_STATE_NAMES
    ):
        return "USA"

    explicit = _explicit_countries(location)
    if len(explicit) != 1:
        return ""
    country = explicit[0]
    if country == "United States":
        return "USA"
    return country


def _parse_salary(job: Job, text: str) -> None:
    if job.salary_min is not None:
        return
    if any(phrase in text for phrase in ["unpaid", "no stipend", "without stipend"]):
        job.salary_currency = job.salary_currency or ("INR" if job.country == "India" else "")
        job.salary_min = 0
        job.salary_max = 0
        job.salary_period = "year"
        job.salary_text = job.salary_text or "Unpaid"
        return
    lpa = re.search(
        r"(?:₹|inr\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:-|–|to)\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:lpa|lakhs?(?:\s+per\s+year)?)",
        text,
    )
    if lpa:
        job.salary_currency = "INR"
        job.salary_min = float(lpa.group(1)) * 100_000
        job.salary_max = float(lpa.group(2)) * 100_000
        job.salary_period = "year"
        job.salary_text = job.salary_text or f"{lpa.group(1)}-{lpa.group(2)} LPA"
        return
    single_lpa = re.search(
        r"(?:₹|inr\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:lpa|lakhs?(?:\s+per\s+year)?)",
        text,
    )
    if single_lpa:
        job.salary_currency = "INR"
        job.salary_min = float(single_lpa.group(1)) * 100_000
        job.salary_period = "year"
        job.salary_text = job.salary_text or f"{single_lpa.group(1)} LPA"
        return
    monthly = re.search(
        r"(?:₹|inr\s*)([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:/|per\s+)?(?:month|monthly|mo\b)",
        text,
    )
    if monthly:
        value = float(monthly.group(1).replace(",", ""))
        job.salary_currency = "INR"
        job.salary_min = value
        job.salary_period = "month"
        job.salary_text = job.salary_text or f"INR {value:g}/month"


def salary_min_lpa(job: Job) -> float | None:
    if job.salary_currency.upper() != "INR" or job.salary_min is None:
        return None
    if job.salary_period == "year":
        return job.salary_min / 100_000
    if job.salary_period == "month":
        return job.salary_min * 12 / 100_000
    return None


_NEGATED_REMOTE_RE = re.compile(
    r"\b(?:not\s+(?:a\s+)?remote|non[- ]remote|no\s+remote|cannot\s+be\s+remote|"
    r"unable\s+to\s+offer\s+remote)\b",
    re.I,
)
_TITLE_HYBRID_TECH_RE = re.compile(
    r"\bhybrid\s+(?:cloud|search|retrieval|rag|model|models|ranking|index|systems?)\b",
    re.I,
)


def infer_remote_policy(location: str, title: str, description: str) -> RemotePolicy:
    """Infer workplace policy conservatively, prioritizing structured location and title.

    Company boilerplate frequently mentions 'remote-first culture', 'remote teams',
    or 'remote code execution'. Those incidental mentions must never override a
    structured Hybrid/Onsite location or turn an unspecified office role into remote.
    """
    loc = " ".join(location.lower().split())
    ttl = " ".join(title.lower().split())
    ttl_workplace = _TITLE_HYBRID_TECH_RE.sub(" ", ttl)

    # 1. Structured location and title take strict precedence over description prose.
    if re.search(r"\bhybrid\b", loc) or re.search(r"\bhybrid\b", ttl_workplace):
        return "hybrid"
    if (
        re.search(r"\b(?:on[- ]?site|in[- ]office|work from office|wfo)\b", f"{loc} {ttl_workplace}")
        or (re.search(r"\boffice\b", loc) and not re.search(r"\bremote\b", loc))
    ):
        return "onsite"
    if (
        (re.search(r"\bremote\b", loc) and not _NEGATED_REMOTE_RE.search(loc))
        or (re.search(r"\bremote\b", ttl_workplace) and not _NEGATED_REMOTE_RE.search(ttl_workplace))
    ):
        return "remote"

    # 2. Fall back to explicit role-level workplace statements in the description.
    desc = " ".join(description.lower().split())
    if not desc:
        return "unknown"
    desc_unnegated = _NEGATED_REMOTE_RE.sub(" ", desc)
    if re.search(
        r"\b(?:hybrid\s+(?:work(?:place|ing)?|role|position|schedule|model|environment)|"
        r"workplace\s*(?:type)?\s*:\s*hybrid|\d+\s+days?\s+(?:a\s+week\s+|per\s+week\s+)?in\s+(?:the\s+)?office)\b",
        desc_unnegated,
    ):
        return "hybrid"
    if re.search(
        r"\b(?:on[- ]?site\s+(?:only|role|position|work)|in[- ]office\s+(?:only|role|position)|"
        r"work\s+from\s+office|wfo|workplace\s*(?:type)?\s*:\s*on[- ]?site)\b",
        desc_unnegated,
    ):
        return "onsite"
    if re.search(
        r"\b(?:remote\s+(?:role|position|job|opportunity|eligible|work\s+eligible)|"
        r"fully\s+remote|100%\s+remote|work\s+from\s+anywhere|"
        r"workplace\s*(?:type)?\s*:\s*remote|location\s*:\s*remote)\b",
        desc_unnegated,
    ):
        return "remote"
    return "unknown"


def enrich_job(job: Job) -> Job:
    j = job.model_copy(deep=True)
    text = f"{j.location} {j.title} {j.description}".lower()
    if not j.country:
        j.country = infer_country(j.location)
    if j.remote_policy == "unknown":
        j.remote_policy = infer_remote_policy(j.location, j.title, j.description)
    if j.min_experience_years is None:
        patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
            r"(\d+(?:\.\d+)?)\+\s*(?:years?|yrs?)",
            r"(?:minimum|min\.?|at least)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
        ]
        for index, pattern in enumerate(patterns):
            match = re.search(pattern, text)
            if not match:
                continue
            j.min_experience_years = float(match.group(1))
            if index == 0:
                j.max_experience_years = float(match.group(2))
            break
    _parse_salary(j, text)
    return j


def reenrich_job_metadata(job: Job) -> tuple[Job, bool, str]:
    """Recompute derived country and workplace metadata safely for an existing indexed job.

    Greenhouse, Lever, and Generic JSON-LD connectors never populate remote_policy
    directly, so their stored remote_policy is 100% derived and can be recomputed
    offline. Ashby may set remote_policy='remote' from upstream isRemote; for Ashby,
    explicit structured hybrid/onsite markers in location/title override stale values
    while unconfirmed 'remote' values are retained and reported as ambiguous until
    the next live Ashby board sync.
    """
    candidate = job.model_copy(deep=True)
    inferred_policy = infer_remote_policy(candidate.location, candidate.title, candidate.description)
    note = "unchanged"

    if candidate.source_kind in {"greenhouse", "lever", "generic"}:
        candidate.remote_policy = "unknown"
        updated = enrich_job(candidate)
        if updated.remote_policy != job.remote_policy or updated.country != job.country:
            note = f"{job.remote_policy}->{updated.remote_policy}"
            return updated, True, note
        return updated, False, note

    # Ashby / learned / custom sources: structured hybrid/onsite overrides stale remote;
    # otherwise preserve explicit upstream remote while reporting ambiguity when inferred='unknown'.
    if inferred_policy in {"hybrid", "onsite"} and candidate.remote_policy != inferred_policy:
        candidate.remote_policy = inferred_policy
        updated = enrich_job(candidate)
        return updated, True, f"{job.remote_policy}->{updated.remote_policy}"
    if candidate.remote_policy == "unknown" and inferred_policy != "unknown":
        candidate.remote_policy = inferred_policy
        updated = enrich_job(candidate)
        return updated, True, f"unknown->{updated.remote_policy}"
    updated = enrich_job(candidate)
    if candidate.source_kind == "ashby" and updated.remote_policy == "remote" and inferred_policy == "unknown":
        note = "ashby_upstream_or_legacy_remote"
    changed = (updated.remote_policy != job.remote_policy) or (updated.country != job.country)
    return updated, changed, note


def reenrich_store_jobs(store, *, apply: bool = False) -> dict:
    """Inspect or update stored active job payloads in place without touching embeddings or history.

    Because job_text(job) excludes remote_policy and country, updating derived metadata
    in jobs.payload preserves search_document and embedding_hash, keeping all cached
    pgvector embeddings valid. Defaults to dry-run (apply=False).
    """
    import hashlib

    transitions: dict[str, int] = {}
    before_policies = {"remote": 0, "hybrid": 0, "onsite": 0, "unknown": 0}
    after_policies = {"remote": 0, "hybrid": 0, "onsite": 0, "unknown": 0}
    ashby_ambiguous = 0
    changed_rows: list[tuple[str, Job]] = []

    jobs = store.active_jobs()
    for job in jobs:
        before_policies[job.remote_policy] = before_policies.get(job.remote_policy, 0) + 1
        stripped = job.model_copy(update={"first_seen_at": None, "last_seen_at": None})
        updated, changed, note = reenrich_job_metadata(stripped)
        after_policies[updated.remote_policy] = after_policies.get(updated.remote_policy, 0) + 1
        if note == "ashby_upstream_or_legacy_remote":
            ashby_ambiguous += 1
        if changed:
            transitions[note] = transitions.get(note, 0) + 1
            changed_rows.append((job.stable_id, updated))

    if apply and changed_rows:
        store_name = type(store).__name__
        if store_name == "PostgresStore":
            from psycopg.types.json import Jsonb
            with store.connect() as connection:
                for stable_id, updated in changed_rows:
                    payload = updated.model_dump(mode="json", exclude_none=True)
                    digest = hashlib.sha256(
                        updated.model_dump_json(exclude_none=True).encode("utf-8")
                    ).hexdigest()
                    connection.execute(
                        "UPDATE jobs SET payload=%s, content_hash=%s WHERE stable_id=%s",
                        (Jsonb(payload), digest, stable_id),
                    )
        elif store_name == "Store":
            with store.connect() as connection:
                for stable_id, updated in changed_rows:
                    payload = updated.model_dump_json(exclude_none=True)
                    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                    connection.execute(
                        "UPDATE jobs SET payload=?, content_hash=? WHERE stable_id=?",
                        (payload, digest, stable_id),
                    )
        else:
            raise TypeError(f"Unsupported store for in-place re-enrichment: {store_name}")

    return {
        "mode": "applied" if apply else "dry_run",
        "active_jobs_scanned": len(jobs),
        "jobs_changed": len(changed_rows),
        "before_remote_policy": before_policies,
        "after_remote_policy": after_policies,
        "transitions": transitions,
        "ashby_retained_for_next_sync": ashby_ambiguous,
    }
