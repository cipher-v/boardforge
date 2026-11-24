from __future__ import annotations

import re

from boardforge.models import Job


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


def infer_country(location: str) -> str:
    """Infer country conservatively from the structured location only."""
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
    return ""


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


def enrich_job(job: Job) -> Job:
    j = job.model_copy(deep=True)
    text = f"{j.location} {j.title} {j.description}".lower()
    if not j.country:
        j.country = infer_country(j.location)
    if j.remote_policy == "unknown":
        if re.search(r"\bremote\b", text):
            j.remote_policy = "remote"
        elif re.search(r"\bhybrid\b", text):
            j.remote_policy = "hybrid"
        elif re.search(r"\b(on[- ]?site|in office|work from office|wfo)\b", text):
            j.remote_policy = "onsite"
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
