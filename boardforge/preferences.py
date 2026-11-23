from __future__ import annotations

import json
import os
import re
import httpx
from boardforge.models import Preferences


ROLE_MAP = {
    "ai": "AI/ML",
    "machine learning": "AI/ML",
    "ml": "AI/ML",
    "computer vision": "Computer Vision",
    "backend": "Backend",
    "software engineer": "Software Engineering",
    "sde": "Software Engineering",
    "data engineer": "Data Engineering",
    "data scientist": "Data Science",
    "qa": "QA",
}

CITIES = ["bengaluru","bangalore","pune","hyderabad","gurugram","gurgaon","mumbai","noida","chennai","delhi"]
NEGATION_RE = re.compile(r"\b(?:do not|don't|dont|avoid|exclude|remove|stop showing|not interested in|no)\b")


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


def deterministic_extract(text: str, current: Preferences | None = None) -> Preferences:
    p = (current or Preferences()).model_copy(deep=True)
    low = text.lower()

    for raw, canonical in ROLE_MAP.items():
        for match in re.finditer(rf"\b{re.escape(raw)}\b", low):
            if _is_negated(low, match.start()):
                p.roles = [role for role in p.roles if role != canonical]
                _append_unique(p.avoid_terms, raw)
            else:
                _append_unique(p.roles, canonical)

    for term in ["sales", "hardware", "embedded", "unpaid"]:
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

    m = re.search(r"(?:more than|over|above|greater than)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", low)
    if m:
        p.max_required_experience_years = float(m.group(1))

    if (
        any(x in low for x in ["no unpaid", "paid only", "only paid", "unpaid internship", "unpaid internships"])
        or ("avoid" in low and "unpaid" in low)
    ):
        p.paid_only = True

    if "remote" in low:
        p.remote_ok = True
    if "onsite only" in low or "on-site only" in low:
        p.remote_ok = False
        p.onsite_ok = True
    if "remote only" in low:
        p.remote_ok = True
        p.onsite_ok = False

    if "india" in low and "outside india" not in low:
        p.countries = ["India"]
    if "worldwide" in low or "any country" in low:
        p.countries = []

    return _reconcile(p)


class LLMPreferenceExtractor:
    """OpenAI-compatible structured extraction, with deterministic fallback."""

    def __init__(self):
        self.key = os.getenv("OPENAI_API_KEY")
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    async def extract(self, text: str, current: Preferences | None = None) -> Preferences:
        if not self.key:
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
