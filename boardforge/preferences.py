from __future__ import annotations

import json
import os
import re
import httpx
from boardforge.enrich import _explicit_countries
from boardforge.models import Preferences


ROLE_MAP = {
    "ai": "AI/ML",
    "ai engineer": "AI/ML",
    "ai engineers": "AI/ML",
    "machine learning": "AI/ML",
    "deep learning": "AI/ML",
    "ml": "AI/ML",
    "ml engineer": "AI/ML",
    "ml engineers": "AI/ML",
    "llm": "AI/ML",
    "rag": "AI/ML",
    "genai": "AI/ML",
    "generative ai": "AI/ML",
    "agentic ai": "AI/ML",
    "ai agent": "AI/ML",
    "ai agents": "AI/ML",
    "computer vision": "Computer Vision",
    "vision engineer": "Computer Vision",
    "vision engineers": "Computer Vision",
    "backend": "Backend",
    "back-end": "Backend",
    "back end": "Backend",
    "software engineer": "Software Engineering",
    "software engineers": "Software Engineering",
    "software engineering": "Software Engineering",
    "software developer": "Software Engineering",
    "software developers": "Software Engineering",
    "sde": "Software Engineering",
    "data engineer": "Data Engineering",
    "data engineers": "Data Engineering",
    "data engineering": "Data Engineering",
    "analytics engineer": "Data Engineering",
    "analytics engineering": "Data Engineering",
    "etl engineer": "Data Engineering",
    "data scientist": "Data Science",
    "data scientists": "Data Science",
    "data science": "Data Science",
    "qa": "QA",
    "quality assurance": "QA",
    "sdet": "QA",
    "test engineer": "QA",
    "test engineers": "QA",
    "automation testing": "QA",
    "test automation": "QA",
}

CITIES = ["bengaluru","bangalore","pune","hyderabad","gurugram","gurgaon","mumbai","noida","chennai","delhi"]
NEGATION_RE = re.compile(
    r"\b(?:do not|don't|dont|not|avoid|exclude|excluding|without|remove|stop showing|not interested in|no)\b"
)


def _is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 60):start]
    # Only consider the current clause so "not QA, but AI" still adds AI.
    clause = re.split(r"[.;]|\bbut\b|\bhowever\b", prefix)[-1]
    return bool(NEGATION_RE.search(clause))


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _reconcile(preferences: Preferences) -> Preferences:
    avoided = {term.casefold() for term in preferences.avoid_terms}
    excluded_roles = {
        canonical
        for raw, canonical in ROLE_MAP.items()
        if raw.casefold() in avoided or canonical.casefold() in avoided
    }
    preferences.roles = [role for role in preferences.roles if role not in excluded_roles]
    return preferences


_NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                 "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_NUMBER = r"(?:\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")"
DIRECT_EXPERIENCE_CAP_RE = re.compile(
    r"\b(?:up to|at most|maximum(?: of)?|max(?:imum)?(?: required)?(?: experience)?\s*:?|"
    r"no more than|not more than|no greater than|not over|under|at or below)"
    r"\s*(" + _NUMBER + r")\s*(?:years?|yrs?)\b(?:\s+of experience)?", re.I
)
EXCEEDING_EXPERIENCE_RE = re.compile(
    r"\b(?:more than|over|above|greater than)"
    r"\s*(" + _NUMBER + r")\s*(?:years?|yrs?)\b(?:\s+of experience)?", re.I
)
EXPERIENCE_FLOOR_RE = re.compile(
    r"(?:\b(?:at least|minimum(?: of)?|min\.?)\s*(" + _NUMBER + r")\s*\+?\s*(?:years?|yrs?)\b(?:\s+of experience)?|"
    r"\b(" + _NUMBER + r")\+\s*(?:years?|yrs?)\b(?:\s+of experience)?)",
    re.I,
)
# Combined pattern for stripping experience framing from occupational text.
EXPERIENCE_CAP_RE = re.compile(
    r"(?:" + DIRECT_EXPERIENCE_CAP_RE.pattern + r"|"
    + EXCEEDING_EXPERIENCE_RE.pattern + r"|"
    + EXPERIENCE_FLOOR_RE.pattern + r")",
    re.I,
)
NO_EXPERIENCE_RE = re.compile(
    r"\b(?:no|zero|without)\s+(?:prior\s+|formal\s+)?(?:employment\s+|work\s+|professional\s+|industry\s+)?experience\b",
    re.I,
)
ENTRY_LEVEL_CUE_RE = re.compile(
    r"\b(?:entry[- ]level|early[- ]career|freshers?|new grads?|recent graduates?)\b",
    re.I,
)


def _parse_number(raw: str) -> float:
    key = raw.casefold()
    return float(_NUMBER_WORDS[key]) if key in _NUMBER_WORDS else float(key)


def extract_experience_cap(text: str) -> float | None:
    """Extract an explicit numeric maximum-experience cap.

    Phrases like 'more than N years' or 'over N years' only define an upper cap
    when negated in the same clause (e.g. 'Avoid roles requiring more than 2 years').
    Unnegated 'more than N years' is a minimum-experience phrase, never a cap.
    """
    low = text.lower()
    values: list[float] = []
    for match in DIRECT_EXPERIENCE_CAP_RE.finditer(low):
        values.append(_parse_number(match.group(1)))
    for match in EXCEEDING_EXPERIENCE_RE.finditer(low):
        if _is_negated(low, match.start()):
            values.append(_parse_number(match.group(1)))
    return min(values) if values else None


def extract_experience_floor(text: str) -> float | None:
    """Detect unnegated minimum-experience phrases (e.g. 'more than 5 years', '5+ years')."""
    low = text.lower()
    floors: list[float] = []
    for match in EXCEEDING_EXPERIENCE_RE.finditer(low):
        if not _is_negated(low, match.start()):
            floors.append(_parse_number(match.group(1)))
    for match in EXPERIENCE_FLOOR_RE.finditer(low):
        if not _is_negated(low, match.start()):
            raw = match.group(1) or match.group(2)
            if raw:
                floors.append(_parse_number(raw))
    return max(floors) if floors else None


def infer_entry_level_cap(text: str) -> float | None:
    low = text.lower()
    if extract_experience_floor(low) is not None:
        return None
    if NO_EXPERIENCE_RE.search(low):
        return 0.0
    if ENTRY_LEVEL_CUE_RE.search(low):
        return 1.0
    return None


def is_entry_level_query(text: str) -> bool:
    low = text.lower()
    return bool(NO_EXPERIENCE_RE.search(low) or ENTRY_LEVEL_CUE_RE.search(low))


def _extract_explicit_in_countries(low: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"\bin\s+([a-z][a-z .'-]{1,35}?)(?=\s+(?:requiring|with|up\s+to|at\s+most|avoid|excluding|without|for|and|or)\b|[,.;!?]|$)", low):
        span = match.group(1).strip()
        if any(city in span for city in CITIES):
            continue
        if re.search(r"\b(?:usa|u\.s\.a\.?|united states|us)\b", span):
            _append_unique(found, "USA")
            continue
        for country in _explicit_countries(span):
            _append_unique(found, "USA" if country == "United States" else country)
    return found


def preserve_search_constraints(saved: Preferences, original: str, refined: Preferences) -> Preferences:
    """Do not let agent rewrites relax the original turn's country or experience limits."""
    base = effective_search_preferences(saved, original)
    result = refined.model_copy(deep=True)
    if base.countries:
        result.countries = list(base.countries)
    if base.max_required_experience_years is not None:
        cap = result.max_required_experience_years
        result.max_required_experience_years = (
            base.max_required_experience_years if cap is None else
            min(base.max_required_experience_years, cap)
        )
    return result


def deterministic_extract(text: str, current: Preferences | None = None) -> Preferences:
    low = text.lower()
    if re.search(r"\b(?:reset|clear)\s+(?:all\s+)?(?:my\s+)?preferences?\b", low):
        return Preferences()

    p = (current or Preferences()).model_copy(deep=True)
    negated_in_turn: set[str] = set()
    positive_canonical_in_turn: set[str] = set()

    for raw, canonical in ROLE_MAP.items():
        for match in re.finditer(rf"\b{re.escape(raw)}\b", low):
            if _is_negated(low, match.start()):
                p.roles = [role for role in p.roles if role != canonical]
                _append_unique(p.avoid_terms, raw)
                negated_in_turn.add(canonical)
            else:
                _append_unique(p.roles, canonical)
                positive_canonical_in_turn.add(canonical)

    # If the user now positively requests a role that was avoided in an earlier
    # turn (and did not negate it in this turn), lift the stale role avoidance.
    rehabilitated = positive_canonical_in_turn - negated_in_turn
    if rehabilitated:
        stale_aliases = {
            raw.casefold()
            for raw, canonical in ROLE_MAP.items()
            if canonical in rehabilitated
        } | {c.casefold() for c in rehabilitated}
        p.avoid_terms = [t for t in p.avoid_terms if t.casefold() not in stale_aliases]

    for term in ["sales", "hardware", "embedded", "firmware", "unpaid", "internship", "internships"]:
        match = re.search(rf"\b{re.escape(term)}\b", low)
        if match and _is_negated(low, match.start()):
            _append_unique(p.avoid_terms, term)

    for city in CITIES:
        if city in low:
            canonical = "Bengaluru" if city in {"bangalore","bengaluru"} else "Gurugram" if city in {"gurgaon","gurugram"} else city.title()
            _append_unique(p.locations, canonical)

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?|lakh)", low)
    if m:
        p.min_salary_lpa = float(m.group(1))

    # Explicit numeric experience caps take precedence over softer entry-level/fresher cues.
    caps = extract_experience_cap(low)
    if caps is not None:
        p.max_required_experience_years = caps
    else:
        inferred_cap = infer_entry_level_cap(low)
        if inferred_cap is not None:
            p.max_required_experience_years = inferred_cap

    if any(phrase in low for phrase in [
        "any experience", "no experience cap", "no experience limit",
        "experience doesn't matter", "experience does not matter",
    ]):
        p.max_required_experience_years = None

    if (
        any(x in low for x in ["no unpaid", "paid only", "only paid", "unpaid internship", "unpaid internships"])
        or ("avoid" in low and "unpaid" in low)
        or ("excluding" in low and "unpaid" in low)
        or ("without" in low and "unpaid" in low)
    ):
        p.paid_only = True

    if re.search(r"\bremote\b", low):
        p.remote_ok = True
    if re.search(r"\bon[- ]?site\b|\bhybrid\b", low):
        p.onsite_ok = True
    if (
        re.search(r"\bon[- ]?site[- ]only\b", low)
        or re.search(r"\bonly\s+(?:show\s+)?on[- ]?site\b", low)
        or any(phrase in low for phrase in ["not remote", "no remote", "avoid remote"])
    ):
        p.remote_ok = False
        p.onsite_ok = True
    if (
        re.search(r"\bremote[- ]only\b", low)
        or re.search(r"\bonly\s+(?:show\s+)?remote\b", low)
        or any(phrase in low for phrase in ["not onsite", "not on-site", "avoid onsite", "avoid on-site"])
    ):
        p.remote_ok = True
        p.onsite_ok = False

    if any(phrase in low for phrase in ["salary matters less", "salary doesn't matter", "salary does not matter", "no minimum salary"]):
        p.min_salary_lpa = None

    if "india" in low and "outside india" not in low:
        p.countries = ["India"]
    else:
        explicit_countries = _extract_explicit_in_countries(low)
        if explicit_countries and "outside " not in low:
            p.countries = explicit_countries
    if "worldwide" in low or "any country" in low:
        p.countries = []

    return _reconcile(p)


def effective_search_preferences(current: Preferences, text: str) -> Preferences:
    """Overlay explicit one-off query constraints without persisting them."""
    transient = deterministic_extract(text, Preferences())
    effective = current.model_copy(deep=True)
    low = text.casefold()

    if transient.roles:
        effective.roles = transient.roles
    if transient.locations:
        effective.locations = transient.locations
    if transient.countries or any(term in low for term in ["india", "worldwide", "any country"]):
        effective.countries = transient.countries
    if transient.min_salary_lpa is not None:
        effective.min_salary_lpa = transient.min_salary_lpa
    if transient.max_required_experience_years is not None:
        effective.max_required_experience_years = transient.max_required_experience_years
    if transient.paid_only:
        effective.paid_only = True
    if any(term in low for term in ["remote", "onsite", "on-site"]):
        effective.remote_ok = transient.remote_ok
        effective.onsite_ok = transient.onsite_ok
    for term in transient.avoid_terms:
        _append_unique(effective.avoid_terms, term)
    return _reconcile(effective)


class LLMPreferenceExtractor:
    """OpenAI-compatible structured extraction, with deterministic fallback."""

    def __init__(self):
        self.key = os.getenv("OPENAI_API_KEY")
        self.enabled = (
            os.getenv("BOARDFORGE_ENABLE_LLM_PREFERENCES", "0") == "1"
            and bool(self.key)
        )
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    async def extract(self, text: str, current: Preferences | None = None) -> Preferences:
        if not self.enabled:
            return deterministic_extract(text, current)

        schema = Preferences.model_json_schema()
        prompt = (
            "Extract durable job-search preferences. Preserve existing preferences unless the user clearly changes them. "
            "Distinguish hard constraints from soft preferences via the available fields. Return JSON only.\n"
            f"CURRENT={json.dumps((current or Preferences()).model_dump())}\nUSER={text}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role":"user","content":prompt}],
            "response_format": {
                "type":"json_schema",
                "json_schema":{"name":"preferences","strict":True,"schema":schema},
            },
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{self.base}/chat/completions",
                headers={"Authorization":f"Bearer {self.key}","Content-Type":"application/json"},
                json=payload,
            )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return _reconcile(Preferences.model_validate_json(content))
