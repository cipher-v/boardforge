from __future__ import annotations

import math
import re
from collections import Counter

from boardforge.enrich import salary_min_lpa
from boardforge.models import Job, Match, MatchEvidence, Preferences


WORD_RE = re.compile(r"[a-zA-Z0-9+#.]{2,}")
ROLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "AI/ML": (
        r"\bai engineer\b", r"\bml engineer\b", r"\bmachine learning\b",
        r"\bdeep learning\b", r"\bapplied ai\b", r"\bllm engineer\b",
        r"\brag engineer\b", r"\bretrieval engineer\b",
        r"\bresearch (?:engineer|scientist)\b",
        r"\b(?:software|platform|systems?|applied|product)?\s*"
        r"(?:engineer|developer|scientist)\b.*\b"
        r"(?:ai|genai|generative ai|llm|rag|agentic)\b",
        r"\b(?:ai|genai|generative ai|llm|rag|agentic)\b.*\b"
        r"(?:engineer|developer|scientist)\b",
    ),
    "Computer Vision": (
        r"\bcomputer vision\b", r"\bvision engineer\b",
        r"\bperception engineer\b", r"\bimage processing engineer\b",
    ),
    "Backend": (
        r"\bback[- ]?end\b", r"\bplatform engineer\b", r"\bapi engineer\b",
        r"\binfrastructure engineer\b", r"\bdistributed systems?\b",
        r"\bserver[- ]side\b", r"\bfull[- ]stack\b",
    ),
    "Software Engineering": (
        r"\bsoftware engineer\b", r"\bsoftware developer\b",
        r"\bapplication developer\b", r"\bfull[- ]stack\b",
    ),
    "Data Engineering": (r"\bdata engineer\b", r"\banalytics engineer\b", r"\betl engineer\b"),
    "Data Science": (r"\bdata scientist\b", r"\bdata science\b", r"\bdecision scientist\b"),
    "QA": (r"\bqa\b", r"\btest engineer\b", r"\bquality assurance\b", r"\bsdet\b"),
}


def tokens(text: str) -> list[str]:
    return [value.lower() for value in WORD_RE.findall(text)]


def job_text(job: Job) -> str:
    return " ".join([job.title, job.company, job.location, job.description, " ".join(job.tags)])


def query_text(preferences: Preferences, query: str = "") -> str:
    return " ".join(preferences.roles + preferences.locations + preferences.prefer_terms + [query])


def _cosine_overlap(query: str, document: str) -> float:
    q = Counter(tokens(query))
    d = Counter(tokens(document))
    if not q or not d:
        return 0.0
    overlap = sum(min(q[token], d[token]) for token in q)
    return overlap / math.sqrt(sum(q.values()) * sum(d.values()))


def lexical_score(job: Job, preferences: Preferences, query: str = "") -> float:
    q = query_text(preferences, query)
    title_and_tags = " ".join([job.title, " ".join(job.tags)])
    return (0.7 * _cosine_overlap(q, title_and_tags)) + (0.3 * _cosine_overlap(q, job_text(job)))


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)", text.lower()))


def role_evidence(job: Job, role: str) -> MatchEvidence | None:
    title = " ".join(job.title.lower().split())
    patterns = ROLE_PATTERNS.get(role)
    if patterns:
        for pattern in patterns:
            match = re.search(pattern, title, re.I)
            if match:
                return MatchEvidence(
                    kind="role",
                    value=role,
                    detail=f"title matched '{match.group(0)}'",
                    weight=0.15,
                )
        return None
    if _contains_phrase(title, role):
        return MatchEvidence(kind="role", value=role, detail=f"title matched '{role}'", weight=0.12)
    return None


def hard_violations(job: Job, preferences: Preferences) -> list[str]:
    text = job_text(job).lower()
    violations: list[str] = []

    if preferences.countries:
        allowed = {country.casefold() for country in preferences.countries}
        if not job.country:
            violations.append("country_unknown")
        elif job.country.casefold() not in allowed:
            violations.append("country")

    if preferences.locations:
        location_match = bool(job.location) and any(
            location.casefold() in job.location.casefold() for location in preferences.locations
        )
        remote_match = preferences.remote_ok and job.remote_policy == "remote"
        if not location_match and not remote_match:
            violations.append("location_unknown" if not job.location else "location")

    if preferences.paid_only and any(phrase in text for phrase in ["unpaid", "no stipend", "without stipend"]):
        violations.append("unpaid")

    if preferences.min_salary_lpa is not None:
        known_lpa = salary_min_lpa(job)
        explicitly_unpaid = any(phrase in text for phrase in ["unpaid", "no stipend", "without stipend"])
        if explicitly_unpaid and preferences.min_salary_lpa > 0:
            violations.append("salary")
        elif known_lpa is not None and known_lpa < preferences.min_salary_lpa:
            violations.append("salary")

    if preferences.max_required_experience_years is not None and job.min_experience_years is not None:
        if job.min_experience_years > preferences.max_required_experience_years:
            violations.append("experience")

    if not preferences.remote_ok and job.remote_policy == "remote":
        violations.append("remote_policy")
    if not preferences.onsite_ok:
        if job.remote_policy == "onsite":
            violations.append("onsite_policy")
        elif job.remote_policy == "unknown":
            violations.append("remote_policy_unknown")

    for term in preferences.avoid_terms:
        if _contains_phrase(text, term):
            violations.append(f"avoid:{term}")
    return violations


def evidence(job: Job, preferences: Preferences) -> tuple[list[str], list[str], float, list[MatchEvidence]]:
    text = job_text(job).lower()
    items: list[MatchEvidence] = []
    risks: list[str] = []

    for role in preferences.roles:
        item = role_evidence(job, role)
        if item:
            items.append(item)

    if preferences.locations and job.location:
        for location in preferences.locations:
            if location.casefold() in job.location.casefold():
                items.append(MatchEvidence(
                    kind="location", value=location,
                    detail=f"location matched '{job.location}'", weight=0.10,
                ))
                break

    if preferences.remote_ok and job.remote_policy == "remote":
        items.append(MatchEvidence(
            kind="remote_policy", value="remote",
            detail="job is explicitly marked remote", weight=0.05,
        ))

    for term in preferences.prefer_terms:
        if _contains_phrase(text, term):
            items.append(MatchEvidence(
                kind="preferred_term", value=term,
                detail=f"job contains preferred term '{term}'", weight=0.03,
            ))

    known_lpa = salary_min_lpa(job)
    if preferences.min_salary_lpa is not None:
        if known_lpa is None:
            risks.append("salary not disclosed or not comparable")
        elif known_lpa >= preferences.min_salary_lpa:
            items.append(MatchEvidence(
                kind="salary", value=f"{known_lpa:g} LPA",
                detail=f"salary meets minimum {preferences.min_salary_lpa:g} LPA", weight=0.08,
            ))
    elif not job.salary_text:
        risks.append("salary not disclosed")

    if preferences.max_required_experience_years is not None:
        if job.min_experience_years is None:
            risks.append("experience requirement not verified")
        elif job.min_experience_years <= preferences.max_required_experience_years:
            items.append(MatchEvidence(
                kind="experience", value=f"{job.min_experience_years:g} years",
                detail="experience floor is within preference", weight=0.04,
            ))

    if job.remote_policy == "unknown":
        risks.append("remote policy not verified")
    reasons = [item.detail for item in items]
    return reasons, risks, sum(item.weight for item in items), items


def rank_jobs(
    jobs: list[Job],
    preferences: Preferences,
    query: str = "",
    top_k: int = 10,
    rejected_ids: set[str] | None = None,
) -> list[Match]:
    rejected_ids = rejected_ids or set()
    matches: list[Match] = []
    for job in jobs:
        if job.stable_id in rejected_ids:
            continue
        violations = hard_violations(job, preferences)
        if violations:
            continue
        reasons, risks, bonus, structured_evidence = evidence(job, preferences)
        if preferences.roles and not any(item.kind == "role" for item in structured_evidence):
            continue
        score = lexical_score(job, preferences, query) + bonus
        status = "recommended"

        # Unknown metadata should not outrank candidates that actually satisfy a
        # stated numeric constraint. Keep them visible, but demote for verification.
        if preferences.min_salary_lpa is not None and salary_min_lpa(job) is None:
            score -= 0.12
            status = "needs_verification"
        if preferences.max_required_experience_years is not None and job.min_experience_years is None:
            score -= 0.08
            status = "needs_verification"

        matches.append(Match(
            job=job,
            score=round(score, 4),
            status=status,
            reasons=reasons,
            risks=risks,
            hard_violations=[],
            evidence=structured_evidence,
        ))
    matches.sort(
        key=lambda match: (
            match.status == "needs_verification",
            -match.score,
            match.job.title.casefold(),
        )
    )
    return matches[:top_k]
