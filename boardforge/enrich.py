from __future__ import annotations

import re
from boardforge.models import Job


INDIA_CITIES = {
    "bengaluru","bangalore","pune","hyderabad","gurugram","gurgaon","mumbai",
    "noida","chennai","delhi","new delhi","ahmedabad","indore","jaipur","kochi",
}
US_HINTS = {
    "san mateo","pittsburgh","new york","seattle","austin","boston","california",
    "ca","ny","tx","wa","pa","usa","united states",
}


def enrich_job(job: Job) -> Job:
    j = job.model_copy(deep=True)
    text = f"{j.location} {j.title} {j.description}".lower()

    if not j.country:
        if "india" in text or any(city in text for city in INDIA_CITIES):
            j.country = "India"
        elif any(hint in text for hint in US_HINTS):
            j.country = "USA"

    if j.remote_policy == "unknown":
        if re.search(r"\bremote\b", text):
            j.remote_policy = "remote"
        elif re.search(r"\bhybrid\b", text):
            j.remote_policy = "hybrid"
        elif re.search(r"\b(on[- ]?site|in office|work from office|wfo)\b", text):
            j.remote_policy = "onsite"

    if j.min_experience_years is None:
        patterns = [
            r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
            r"(\d+(?:\.\d+)?)\+\s*(?:years?|yrs?)",
            r"(?:minimum|min\.?|at least)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
        ]
        for i, pat in enumerate(patterns):
            m = re.search(pat, text)
            if not m:
                continue
            j.min_experience_years = float(m.group(1))
            if i == 0 and len(m.groups()) > 1:
                j.max_experience_years = float(m.group(2))
            break

    if not j.salary_text:
        m = re.search(r"(?:₹|inr\s*)?([0-9]+(?:\.[0-9]+)?)\s*[-–]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:lpa|lakh)", text)
        if m:
            j.salary_text = f"{m.group(1)}-{m.group(2)} LPA"

    return j
