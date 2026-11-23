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


def deterministic_extract(text: str, current: Preferences | None = None) -> Preferences:
    p = (current or Preferences()).model_copy(deep=True)
    low = text.lower()

    if any(x in low for x in ["don't show", "do not show", "avoid", "not interested"]):
        for term in ["qa","sales","hardware","embedded","unpaid"]:
            if term in low and term not in p.avoid_terms:
                p.avoid_terms.append(term)

    for raw, canonical in ROLE_MAP.items():
        if re.search(rf"\b{re.escape(raw)}\b", low) and canonical not in p.roles:
            p.roles.append(canonical)

    for city in CITIES:
        if city in low:
            canonical = "Bengaluru" if city in {"bangalore","bengaluru"} else "Gurugram" if city in {"gurgaon","gurugram"} else city.title()
            if canonical not in p.locations:
                p.locations.append(canonical)

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

    return p


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
        return Preferences.model_validate_json(content)
