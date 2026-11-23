from __future__ import annotations

import math
import re
from collections import Counter
from boardforge.models import Job, Match, Preferences


WORD_RE = re.compile(r"[a-zA-Z0-9+#.]{2,}")


def tokens(text: str) -> list[str]:
    return [x.lower() for x in WORD_RE.findall(text)]


def job_text(job: Job) -> str:
    return " ".join([job.title, job.company, job.location, job.description, " ".join(job.tags)])


def query_text(p: Preferences, query: str = "") -> str:
    return " ".join(p.roles + p.locations + p.prefer_terms + [query])


def lexical_score(job: Job, p: Preferences, query: str = "") -> float:
    q = Counter(tokens(query_text(p, query)))
    d = Counter(tokens(job_text(job)))
    if not q or not d:
        return 0.0
    overlap = sum(min(q[t], d[t]) for t in q)
    return overlap / math.sqrt(sum(q.values()) * sum(d.values()))


def hard_violations(job: Job, p: Preferences) -> list[str]:
    text = job_text(job).lower()
    out: list[str] = []

    if p.countries:
        allowed = {x.lower() for x in p.countries}
        if job.country:
            if job.country.lower() not in allowed:
                out.append("country")
        elif p.locations:
            # Unknown country is unsafe when the candidate explicitly constrained geography.
            if not any(loc.lower() in job.location.lower() for loc in p.locations):
                out.append("country_unknown")
        elif "india" in allowed and job.location:
            india_loc = any(x in job.location.lower() for x in [
                "india","bengaluru","bangalore","pune","hyderabad","gurugram","gurgaon",
                "mumbai","noida","chennai","delhi","ahmedabad","indore","jaipur","kochi"
            ])
            if not india_loc:
                out.append("country_unknown")

    if p.paid_only and any(x in text for x in ["unpaid","no stipend"]):
        out.append("unpaid")

    if p.max_required_experience_years is not None and job.min_experience_years is not None:
        if job.min_experience_years > p.max_required_experience_years:
            out.append("experience")

    if not p.remote_ok and job.remote_policy == "remote":
        out.append("remote_policy")
    if not p.onsite_ok and job.remote_policy == "onsite":
        out.append("onsite_policy")

    for term in p.avoid_terms:
        if term.lower() in text:
            out.append(f"avoid:{term}")

    return out


def _role_fit(job: Job, role: str) -> bool:
    title = job.title.lower()
    desc = job.description.lower()
    role_low = role.lower()

    if role_low == "computer vision":
        return "computer vision" in title or any(x in title for x in ["vision engineer","perception engineer"])
    if role_low == "backend":
        return any(x in title for x in ["backend","back-end","software engineer","full stack","full-stack"])
    if role_low == "ai/ml":
        return any(x in title for x in ["ai","ml","machine learning","deep learning","research engineer","research scientist"])
    if role_low == "software engineering":
        return any(x in title for x in ["software engineer","developer","full stack","full-stack"])
    if role_low == "data engineering":
        return "data engineer" in title
    if role_low == "data science":
        return any(x in title for x in ["data scientist","data science"])
    if role_low == "qa":
        return any(x in title for x in ["qa","test engineer","quality assurance"])

    return role_low in title or role_low in desc


def evidence(job: Job, p: Preferences) -> tuple[list[str], list[str], float]:
    text = job_text(job).lower()
    reasons: list[str] = []
    risks: list[str] = []
    bonus = 0.0

    for role in p.roles:
        if _role_fit(job, role):
            reasons.append(f"role aligns with {role}")
            bonus += 0.12

    if p.locations and job.location:
        if any(loc.lower() in job.location.lower() for loc in p.locations):
            reasons.append(f"location matches {job.location}")
            bonus += 0.10

    if p.remote_ok and job.remote_policy == "remote":
        reasons.append("remote option matches preference")
        bonus += 0.05

    for term in p.prefer_terms:
        if term.lower() in text:
            reasons.append(f"contains preferred term: {term}")
            bonus += 0.03

    if job.min_experience_years is not None and p.max_required_experience_years is not None:
        if job.min_experience_years <= p.max_required_experience_years:
            reasons.append("experience floor is within preference")
        else:
            risks.append(f"requires at least {job.min_experience_years:g} years")

    if not job.location:
        risks.append("location metadata missing")
    if job.remote_policy == "unknown":
        risks.append("remote policy not verified")
    if not job.salary_text:
        risks.append("salary not disclosed")

    return reasons, risks, bonus


def rank_jobs(
    jobs: list[Job],
    p: Preferences,
    query: str = "",
    top_k: int = 10,
    rejected_ids: set[str] | None = None,
) -> list[Match]:
    rejected_ids = rejected_ids or set()
    matches: list[Match] = []
    for job in jobs:
        if job.stable_id in rejected_ids:
            continue
        violations = hard_violations(job, p)
        reasons, risks, bonus = evidence(job, p)
        score = lexical_score(job, p, query) + bonus - 0.35 * len(violations)
        # Hard constraints are exclusion rules, not merely ranking penalties.
        if violations:
            continue
        matches.append(Match(job=job, score=round(score,4), reasons=reasons, risks=risks, hard_violations=[]))

    matches.sort(key=lambda m: (-m.score, m.job.title.lower()))
    return matches[:top_k]
