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

    if p.countries and job.country:
        if job.country.lower() not in {x.lower() for x in p.countries}:
            out.append("country")

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


def evidence(job: Job, p: Preferences) -> tuple[list[str], list[str], float]:
    text = job_text(job).lower()
    reasons: list[str] = []
    risks: list[str] = []
    bonus = 0.0

    for role in p.roles:
        role_tokens = [t for t in tokens(role) if len(t) > 2]
        if any(t in text for t in role_tokens):
            reasons.append(f"role aligns with {role}")
            bonus += 0.10

    if p.locations and job.location:
        if any(loc.lower() in job.location.lower() for loc in p.locations):
            reasons.append(f"location matches {job.location}")
            bonus += 0.08

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
        matches.append(Match(job=job, score=round(score,4), reasons=reasons, risks=risks, hard_violations=violations))

    # hard constraints always lose to clean candidates.
    matches.sort(key=lambda m: (bool(m.hard_violations), -m.score, m.job.title.lower()))
    return matches[:top_k]
