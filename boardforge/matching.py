from __future__ import annotations

import math
import re
from collections import Counter

from boardforge.confidence import (
    LEADERSHIP_TITLE_RE,
    SENIORITY_TITLE_RE,
    assess_query_relevance,
    enforce_recommendation_confidence,
    occupational_query_text,
    parse_query_intent,
    query_evidence_items,
)
from boardforge.enrich import (
    extract_fresher_eligibility_evidence,
    infer_country,
    salary_min_lpa,
)
from boardforge.models import Job, Match, MatchEvidence, Preferences
from boardforge.preferences import (
    extract_experience_floor,
    is_entry_level_target,
    strip_orchestration_instructions,
)


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
    "Data Engineering": (
        r"\bdata engineer\b", r"\bdata engineering\b",
        r"\banalytics engineer\b", r"\betl engineer\b",
    ),
    "Data Science": (r"\bdata scientist\b", r"\bdata science\b", r"\bdecision scientist\b"),
    "QA": (r"\bqa\b", r"\btest engineer\b", r"\bquality assurance\b", r"\bsdet\b"),
}

_QA_TITLE_RE = re.compile(
    r"\b(?:sdet|qa|quality\s+assurance|test(?:ing)?\s+engineer|engineer\s+in\s+test|quality\s+engineer)\b",
    re.I,
)
_FRONTEND_FOCUSED_TITLE_RE = re.compile(
    r"\b(?:front[- ]?end|ui|ux|mobile|ios|android)(?:[- ]+(?:focused|first|only|specialist))?\b",
    re.I,
)
_SOLUTIONS_CUSTOMER_TITLE_RE = re.compile(
    r"\b(?:ai\s+|ml\s+|technical\s+)?(?:solutions?|forward[- ]deployed|implementation|"
    r"deployment|customer|partner|field|pre[- ]?sales|sales)\s+"
    r"(?:ai\s+|ml\s+)?(?:engineer|developer|architect|specialist)\b",
    re.I,
)
_DEVOPS_INFRA_TITLE_RE = re.compile(
    r"\b(?:devops|sre|site\s+reliability|cloud\s+infrastructure|release\s+engineer|build\s+engineer)\b",
    re.I,
)
_SECURITY_TITLE_RE = re.compile(
    r"\b(?:security|cybersecurity|appsec|infosec|soc|pentest)\s+(?:ai\s+)?(?:engineer|analyst|specialist)\b",
    re.I,
)
_CRM_SUPPORT_TITLE_RE = re.compile(
    r"\b(?:salesforce|agentforce|servicenow|workday|netsuite|amazon\s+connect|"
    r"contact\s+center|technical\s+support|customer\s+support|it\s+support|help\s*desk|"
    r"support\s+engineer)\b",
    re.I,
)
_NON_VACANCY_TITLE_RE = re.compile(
    r"\b(?:submit\s+(?:a|that)\s+referral|talent\s+(?:pool|community)|"
    r"general\s+application|not\s+currently\s+posted)\b",
    re.I,
)
_MLOPS_INFRA_RESPONSIBILITY_RE = re.compile(
    r"\b(?:mlops|model\s+(?:serving|deployment|training|inference)|"
    r"inference\s+(?:pipeline|pipelines|server|engine)|kubeflow|sagemaker|triton|vllm)\b",
    re.I,
)
_ADJACENT_AI_SOLUTIONS_DESC_RE = re.compile(
    r"\b(?:system\s+prompts?|prompt\s+engineering|voice\s+agents?|llm|rag|"
    r"agent\s+flows?|conversational\s+ai|generative\s+ai)\b",
    re.I,
)


def tokens(text: str) -> list[str]:
    return [value.lower() for value in WORD_RE.findall(text)]


def job_text(job: Job) -> str:
    return " ".join([job.title, job.company, job.location, job.description, " ".join(job.tags)])


def query_text(preferences: Preferences, query: str = "") -> str:
    cleaned_query = strip_orchestration_instructions(query) if query else ""
    return " ".join(preferences.roles + preferences.locations + preferences.prefer_terms + [cleaned_query])


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


def _generic_engineering_title(title: str) -> bool:
    return bool(re.search(
        r"\b(?:software|application|applications|platform)\s+"
        r"(?:engineer|developer)\b",
        title,
        re.I,
    ))


def _description_role_evidence(job: Job, role: str) -> MatchEvidence | None:
    """Allow specialized work hidden behind generic engineering titles.

    This intentionally requires a generic hands-on engineering title plus
    substantive role-specific language in the JD. It does not let managers or
    architects pass merely because their descriptions mention AI or data.
    """
    title = " ".join(job.title.lower().split())
    if not _generic_engineering_title(title):
        return None

    description = job.description.lower()
    patterns_by_role: dict[str, tuple[str, ...]] = {
        "AI/ML": (
            r"\bmachine learning systems?\b",
            r"\bllm orchestration\b",
            r"\bembedding models?\b",
            r"\bvector stores?\b",
            r"\bautonomous agents?\b",
            r"\bagentic ai systems?\b",
            r"\brag pipelines?\b",
            r"\bretrieval strategies\b",
            r"\btransformer models?\b",
            r"\bcomputer vision\b",
            r"\bdeep learning\b",
        ),
        "Backend": (
            r"\bbackend (?:services?|systems?)\b",
            r"\bmicroservices?\b",
            r"\bservice[- ]oriented architecture\b",
            r"\bapi development\b",
            r"\bdistributed systems?\b",
        ),
        "Data Engineering": (
            r"\bdata platform\b",
            r"\bdata pipelines?\b",
            r"\bdata lake\b",
            r"\bdata services?\b",
            r"\betl\b",
            r"\bairflow\b",
        ),
    }
    for pattern in patterns_by_role.get(role, ()):
        match = re.search(pattern, description, re.I)
        if match:
            return MatchEvidence(
                kind="role",
                value=role,
                detail=f"description substantiated {role} work via '{match.group(0)}'",
                weight=0.10,
            )
    return None


def query_topic_terms(preferences: Preferences, query: str) -> set[str]:
    return set(parse_query_intent(preferences, query).specific_terms)


def query_topic_evidence(
    job: Job,
    preferences: Preferences,
    query: str,
) -> MatchEvidence | None:
    """Require substantial overlap on meaningful query terms.

    Geography, seniority wording, and generic role words are intentionally
    ignored so they cannot by themselves make a semantic neighbor look
    confidently relevant.
    """
    assessment = assess_query_relevance(job, preferences, query)
    terms = assessment.intent.specific_terms
    if not terms or not assessment.confident_topic:
        return None
    matched = sorted(assessment.coherent_terms)
    coverage = len(assessment.matched_terms) / len(terms)
    weight = min(0.12, 0.04 + (0.08 * coverage))
    return MatchEvidence(
        kind="query_topic",
        value=", ".join(matched),
        detail=(
            f"query topic coherently matched {len(matched)}/{len(terms)} meaningful terms: "
            + ", ".join(matched)
        ),
        weight=weight,
    )


def role_evidence(job: Job, role: str, *, query: str = "") -> MatchEvidence | None:
    title = " ".join(job.title.lower().split())
    low_q = query.casefold()
    if LEADERSHIP_TITLE_RE.search(title) or _NON_VACANCY_TITLE_RE.search(title):
        return None
    if role == "AI/ML" and re.search(r"\b(?:hardware|embedded|firmware)\b", title, re.I):
        return None

    # Do not let CRM/Salesforce/support/security titles with incidental "AI" qualify as direct core engineering.
    if role in {"AI/ML", "Computer Vision", "Backend", "Software Engineering", "Data Engineering"}:
        if _CRM_SUPPORT_TITLE_RE.search(title) and not _CRM_SUPPORT_TITLE_RE.search(low_q):
            return None
    if role in {"AI/ML", "Computer Vision"}:
        if _SECURITY_TITLE_RE.search(title) and "security" not in low_q:
            return None
        if _DEVOPS_INFRA_TITLE_RE.search(title) and not _DEVOPS_INFRA_TITLE_RE.search(low_q):
            if role == "AI/ML" and _MLOPS_INFRA_RESPONSIBILITY_RE.search(job.description):
                return MatchEvidence(
                    kind="adjacent_role",
                    value=role,
                    detail=f"adjacent MLOps/AI infrastructure role ('{job.title}')",
                    weight=0.09,
                )
            return None

    if _SOLUTIONS_CUSTOMER_TITLE_RE.search(title) and not _SOLUTIONS_CUSTOMER_TITLE_RE.search(low_q):
        if role == "AI/ML" and (
            re.search(r"\b(?:ai|ml|llm|voice)\b", title, re.I)
            or _ADJACENT_AI_SOLUTIONS_DESC_RE.search(job.description)
        ):
            return MatchEvidence(
                kind="adjacent_role",
                value=role,
                detail=f"adjacent solutions/customer-facing AI role ('{job.title}')",
                weight=0.09,
            )
        if role in {"Software Engineering", "Backend"} and re.search(
            r"\b(?:api|apis|backend|python|integrations?|sdk)\b", job.description, re.I
        ):
            return MatchEvidence(
                kind="adjacent_role",
                value=role,
                detail=f"adjacent solutions/integration engineering role ('{job.title}')",
                weight=0.08,
            )
        return None

    if role == "Backend":
        if _QA_TITLE_RE.search(title):
            return None
        if not re.search(r"\bback[- ]?end\b", title, re.I) and _FRONTEND_FOCUSED_TITLE_RE.search(title):
            return None
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
        return _description_role_evidence(job, role)
    if _contains_phrase(title, role):
        return MatchEvidence(kind="role", value=role, detail=f"title matched '{role}'", weight=0.12)
    return None


def hard_violations(job: Job, preferences: Preferences, query: str = "") -> list[str]:
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
        inferred_countries = {
            country.casefold()
            for country in (infer_country(loc) for loc in preferences.locations)
            if country
        }
        country_compatible = (
            not inferred_countries
            or (bool(job.country) and job.country.casefold() in inferred_countries)
        )
        remote_match = (
            preferences.remote_ok
            and job.remote_policy == "remote"
            and country_compatible
        )
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

    if is_entry_level_target(preferences, query) and (
        SENIORITY_TITLE_RE.search(job.title) or LEADERSHIP_TITLE_RE.search(job.title)
    ):
        violations.append("seniority_title")

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


def evidence(
    job: Job, preferences: Preferences, *, query: str = ""
) -> tuple[list[str], list[str], float, list[MatchEvidence]]:
    text = job_text(job).lower()
    items: list[MatchEvidence] = []
    risks: list[str] = []

    for role in preferences.roles:
        item = role_evidence(job, role, query=query)
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
            fresher_ev = extract_fresher_eligibility_evidence(job.title, job.description)
            if fresher_ev and fresher_ev[0] == "explicit":
                items.append(MatchEvidence(
                    kind="experience",
                    value="entry-level eligible",
                    detail=f"explicit fresher/entry-level eligibility ('{fresher_ev[1]}')",
                    weight=0.04,
                ))
            elif fresher_ev and fresher_ev[0] == "early_career_signal":
                items.append(MatchEvidence(
                    kind="experience_signal",
                    value="early-career signal",
                    detail=f"early-career/fresher cue in description ('{fresher_ev[1]}')",
                    weight=0.02,
                ))
                risks.append(
                    "numeric experience requirement not stated (early-career/fresher cues present)"
                )
            else:
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


def _has_query_overlap(
    job: Job,
    preferences: Preferences,
    query: str,
    structured_evidence: list[MatchEvidence],
    assessment,
) -> bool:
    """Return False when an explicit search query has zero lexical or role overlap with a job."""
    if not query.strip():
        return True
    if preferences.roles:
        return any(item.kind in {"role", "adjacent_role"} for item in structured_evidence)
    if assessment is not None:
        if assessment.matched_terms:
            return True
        if assessment.intent.leadership and LEADERSHIP_TITLE_RE.search(job.title):
            return True
        if (
            (assessment.intent.domain_terms or assessment.intent.skill_terms)
            and _generic_engineering_title(job.title)
        ):
            return True
        if assessment.intent.specific_terms:
            return False
    cleaned = occupational_query_text(query)
    for term in [*preferences.countries, *preferences.locations]:
        if term:
            cleaned = re.sub(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", " ", cleaned)
    cleaned = re.sub(r"\b(?:remote|remote-only|onsite|on-site|onsite-only|hybrid)\b", " ", cleaned)
    if not tokens(cleaned):
        return True
    return _cosine_overlap(cleaned, job_text(job)) > 0.0


def rank_jobs(
    jobs: list[Job],
    preferences: Preferences,
    query: str = "",
    top_k: int = 10,
    rejected_ids: set[str] | None = None,
    *,
    allow_semantic_candidates: bool = False,
) -> list[Match]:
    if top_k <= 0:
        return []
    rejected_ids = rejected_ids or set()
    entry_level_target = is_entry_level_target(preferences, query)
    min_floor = extract_experience_floor(query) if query.strip() else None
    low_query = query.casefold()
    explicit_remote_requested = (
        bool(re.search(r"\bremote\b", low_query))
        and not bool(re.search(r"\b(?:on[- ]?site|hybrid)\b", low_query))
    )

    matches: list[Match] = []
    for job in jobs:
        if job.stable_id in rejected_ids:
            continue
        violations = hard_violations(job, preferences, query=query)
        if violations:
            continue
        reasons, risks, bonus, structured_evidence = evidence(job, preferences, query=query)
        assessment = None
        if query.strip():
            assessment = assess_query_relevance(job, preferences, query)
            intent_items = query_evidence_items(assessment)
            structured_evidence.extend(intent_items)
            reasons.extend(item.detail for item in intent_items)
            bonus += sum(item.weight for item in intent_items)
        has_direct_role = any(item.kind == "role" for item in structured_evidence)
        has_adjacent_role = any(item.kind == "adjacent_role" for item in structured_evidence)
        if preferences.roles and not has_direct_role and not has_adjacent_role:
            continue
        if not allow_semantic_candidates and not _has_query_overlap(
            job, preferences, query, structured_evidence, assessment
        ):
            continue

        score = lexical_score(job, preferences, query) + bonus
        status = "recommended"
        if has_adjacent_role and not has_direct_role:
            status = "alternative"
            adj_risk = (
                "adjacent occupational match rather than direct "
                + ", ".join(preferences.roles)
                + " engineering"
            )
            if adj_risk not in risks:
                risks.append(adj_risk)

        # On SWE/Backend searches, flag DevOps-titled roles whose focus is cloud/SRE infrastructure.
        if (
            has_direct_role
            and _DEVOPS_INFRA_TITLE_RE.search(job.title)
            and query.strip()
            and not _DEVOPS_INFRA_TITLE_RE.search(query)
        ):
            status = "needs_verification"
            devops_risk = "title indicates DevOps/infrastructure focus; verify fit with target role"
            if devops_risk not in risks:
                risks.append(devops_risk)

        if explicit_remote_requested:
            if job.remote_policy == "remote":
                score += 0.08
            elif job.remote_policy == "onsite":
                score -= 0.12
                status = "needs_verification"
                if "job is onsite while query requested remote" not in risks:
                    risks.append("job is onsite while query requested remote")

        has_explicit_fresher = any(
            item.kind == "experience" and item.value == "entry-level eligible"
            for item in structured_evidence
        )
        has_early_career_signal = any(
            item.kind == "experience_signal" for item in structured_evidence
        )

        # Unknown metadata should not outrank candidates that actually satisfy a
        # stated numeric constraint. Keep them visible, but demote for verification.
        if preferences.min_salary_lpa is not None and salary_min_lpa(job) is None:
            score -= 0.12
            if status != "alternative":
                status = "needs_verification"
        if (
            preferences.max_required_experience_years is not None
            and job.min_experience_years is None
            and not has_explicit_fresher
        ):
            score -= 0.04 if has_early_career_signal else 0.08
            if status != "alternative":
                status = "needs_verification"
        if entry_level_target and SENIORITY_TITLE_RE.search(job.title):
            score -= 0.18
            status = "needs_verification"
        if min_floor is not None:
            if job.min_experience_years is None:
                score -= 0.08
                status = "needs_verification"
            elif job.min_experience_years < min_floor:
                score -= 0.15
                status = "needs_verification"

        match = Match(
            job=job,
            score=round(score, 4),
            status=status,
            reasons=reasons,
            risks=risks,
            hard_violations=[],
            evidence=structured_evidence,
        )
        matches.append(enforce_recommendation_confidence(match, preferences, query))
    status_order = {"recommended": 0, "alternative": 1, "needs_verification": 2}
    matches.sort(
        key=lambda match: (
            status_order.get(match.status, 2),
            (entry_level_target and bool(SENIORITY_TITLE_RE.search(match.job.title))),
            (
                preferences.max_required_experience_years is not None
                and match.job.min_experience_years is None
                and not any(
                    item.kind == "experience" and item.value == "entry-level eligible"
                    for item in match.evidence
                )
            ),
            -match.score,
            match.job.title.casefold(),
        )
    )
    return matches[:top_k]
