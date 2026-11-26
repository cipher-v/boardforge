from __future__ import annotations

import math
import re
from dataclasses import dataclass

from boardforge.models import Job, Match, MatchEvidence, Preferences
from boardforge.preferences import EXPERIENCE_CAP_RE, extract_experience_floor, is_entry_level_target


TOKEN_RE = re.compile(r"[a-zA-Z0-9+#.]{2,}")

# This is a reusable occupational vocabulary, not a benchmark rule table. It
# separates kinds of intent so that two incidental words in a long JD cannot
# stand in for role/task alignment.
GENERIC_QUERY_TERMS = {
    "a", "an", "and", "or", "the", "to", "for", "of", "in", "on", "with",
    "job", "jobs", "role", "roles", "position", "positions", "work", "working",
    "opening", "openings", "opportunity", "opportunities", "career", "careers",
    "candidate", "candidates", "india", "remote", "onsite", "hybrid",
    "worldwide", "anywhere", "global", "paid", "unpaid",
    "now", "then", "please", "me", "my", "changed", "mind",
    "only", "avoid", "exclude", "excluding", "without", "requiring", "require",
    "required", "minimum", "least", "more",
    "than", "over", "above", "greater", "at", "most", "max", "maximum", "under",
    "below", "up",
    "years", "year", "yrs", "experience", "senior", "staff", "principal",
}
OCCUPATION_TERMS = {
    "engineer", "engineering", "developer", "architect", "manager", "specialist",
    "scientist", "analyst", "consultant", "administrator", "lead", "technical",
}
ROLE_TERMS = {
    "AI/ML": {"ai", "ml", "machine", "learning", "llm", "genai", "generative"},
    "Computer Vision": {"computer", "vision", "perception", "image", "processing"},
    "Backend": {"backend", "back", "end", "server", "side"},
    "Software Engineering": {"software", "engineering", "developer"},
    "Data Engineering": {"data", "engineering", "etl"},
    "Data Science": {"data", "science", "scientist"},
    "QA": {"qa", "test", "testing", "quality", "assurance", "sdet"},
}
DOMAIN_TERMS = {
    "security", "fraud", "abuse", "biometric", "biometrics", "identity",
    "vision", "messaging", "telecom", "support", "customer", "sales",
    "finance", "compliance", "risk", "data", "platform", "infrastructure",
    "operations", "automation", "analytics", "conversation", "conversational",
    "chatbot", "storage", "streaming", "onboarding", "account", "gtm",
}
BROAD_CONTEXT_TERMS = {
    "platform", "systems", "system", "services", "service", "data", "customer",
    "engineering", "software", "operations", "process", "workflows", "scale",
}
SKILL_TERMS = {
    "python", "sql", "java", "golang", "kotlin", "javascript", "typescript",
    "pytorch", "tensorflow", "onnx", "tensorrt", "airflow", "kafka", "spark",
    "aws", "gcp", "azure", "vertex", "langgraph", "langsmith", "bedrock",
    "salesforce", "okta", "jira", "temporal", "n8n", "mcp", "hitl", "rag",
    "llm", "cdp", "a2p", "api", "apis", "etl", "mlops", "sagemaker",
}
LEADERSHIP_TITLE_RE = re.compile(
    r"\b(?:engineering|development|data|platform|software|product)\s+manager\b|"
    r"\b(?:director|head|vp|vice president)\b|"
    r"\bengineering\s+leader\b",
    re.I,
)
SENIORITY_TITLE_RE = re.compile(
    r"(?<!rising\s)\b(?:senior|sr\.?|staff|principal|distinguished|fellow|lead|leader|"
    r"director|head|vp|vice\s+president|chief|cto|cio|ciso|architect|"
    r"iii|iv|(?:sde|swe)[- ]?[3-9]|l[5-9]|p[5-9])\b|"
    r"(?<!associate\sproduct\s)\bmanager\b",
    re.I,
)
ACCOUNT_MANAGER_RE = re.compile(r"\b(?:technical\s+)?account manager\b", re.I)
GENERIC_ENGINEERING_TITLE_RE = re.compile(
    r"\b(?:software|application|applications|platform)?\s*"
    r"(?:engineer|developer)\b",
    re.I,
)


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text.replace("-", " "))]


@dataclass(frozen=True)
class QueryIntent:
    role_terms: frozenset[str]
    domain_terms: frozenset[str]
    skill_terms: frozenset[str]
    task_terms: frozenset[str]
    leadership: bool
    individual_contributor: bool

    @property
    def specific_terms(self) -> frozenset[str]:
        return self.domain_terms | self.skill_terms | self.task_terms


@dataclass(frozen=True)
class RelevanceAssessment:
    intent: QueryIntent
    matched_terms: frozenset[str]
    coherent_terms: frozenset[str]
    title_terms: frozenset[str]
    required_terms: int
    confident_topic: bool
    reasons: tuple[str, ...]


_NEGATED_CLAUSE_RE = re.compile(
    r"\b(?:avoid|exclude|excluding|without|do\s+not\s+(?:show|include|want)|"
    r"don't\s+(?:show|include|want)|dont\s+(?:show|include|want)|stop\s+showing|"
    r"no\s+unpaid)\b[^.,;!?]*",
    re.I,
)


def occupational_query_text(query: str) -> str:
    """Remove search framing, negated exclusions, and eligibility phrases, not occupational evidence."""
    cleaned = EXPERIENCE_CAP_RE.sub(" ", query.casefold())
    cleaned = _NEGATED_CLAUSE_RE.sub(" ", cleaned)
    for pattern in (
        r"\b(?:entry[- ]level|early[- ]career|freshers?|new grads?|recent graduates?)\b",
        r"\b(?:(?:i\s+have|i've\s+got)\s+)?(?:for\s+candidates?\s+(?:with|without)\s+)?"
        r"(?:no|zero|without)\s+(?:prior\s+|formal\s+)?(?:employment\s+|work\s+|professional\s+|industry\s+)?experience\b",
        r"\b(?:suitable for|matching|based on)\s+my\s+(?:profile|resume|cv|background)\b",
        r"\b(?:and\s+)?(?:clearly\s+)?flag\s+(?:any\s+)?(?:uncertain|unknown|unverified)\s+eligibility\b",
        r"\b(?:changed\s+my\s+mind|from\s+now\s+on|going\s+forward)\b",
        r"\b(?:find|search(?:\s+for)?|show(?:\s+me)?|include|explore|look\s+for|list)\b",
    ):
        cleaned = re.sub(pattern, " ", cleaned)
    # Sentence punctuation is not a technical query term.
    cleaned = re.sub(r"[.,;:!?](?=\s|$)", " ", cleaned)
    return " ".join(cleaned.split())


def parse_query_intent(preferences: Preferences, query: str) -> QueryIntent:
    low = query.casefold().replace("-", " ")
    leadership = bool(LEADERSHIP_TITLE_RE.search(low)) and not ACCOUNT_MANAGER_RE.search(low)
    individual_contributor = bool(
        re.search(r"\b(?:engineer|developer|scientist|specialist)\b", low)
    ) and not leadership

    cleaned = occupational_query_text(query).replace("-", " ")
    for term in [*preferences.countries, *preferences.locations]:
        if term:
            cleaned = re.sub(
                rf"(?<!\w){re.escape(term.casefold())}(?!\w)", " ", cleaned
            )
    cleaned = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:lpa|lakhs?|lakh|years?|yrs?)\b", " ", cleaned
    )
    raw = set(_tokens(cleaned)) - GENERIC_QUERY_TERMS
    role_terms = set().union(
        *(ROLE_TERMS.get(role, set()) for role in preferences.roles)
    ) if preferences.roles else set()
    avoided_tokens = {
        tok
        for term in preferences.avoid_terms
        for tok in _tokens(term)
    }
    remaining = raw - role_terms - OCCUPATION_TERMS - avoided_tokens
    domains = remaining & DOMAIN_TERMS
    skills = remaining & SKILL_TERMS
    tasks = remaining - domains - skills
    return QueryIntent(
        role_terms=frozenset(role_terms),
        domain_terms=frozenset(domains),
        skill_terms=frozenset(skills),
        task_terms=frozenset(tasks),
        leadership=leadership,
        individual_contributor=individual_contributor,
    )


def assess_query_relevance(
    job: Job,
    preferences: Preferences,
    query: str,
) -> RelevanceAssessment:
    intent = parse_query_intent(preferences, query)
    specific = set(intent.specific_terms)
    if not specific:
        return RelevanceAssessment(
            intent=intent,
            matched_terms=frozenset(),
            coherent_terms=frozenset(),
            title_terms=frozenset(),
            required_terms=0,
            confident_topic=True,
            reasons=(),
        )

    title_terms = specific & set(_tokens(" ".join([job.title, *job.tags])))
    description_tokens = _tokens(job.description)
    matched = specific & set(description_tokens) | title_terms

    # A short sliding window requires evidence to describe one coherent piece
    # of work. Scattered mentions across benefits, company boilerplate, and a
    # long requirements list no longer combine into confident relevance.
    best_window: set[str] = set()
    window_size = 48
    stride = 16
    for start in range(0, max(1, len(description_tokens)), stride):
        window = set(description_tokens[start:start + window_size]) & specific
        if len(window) > len(best_window):
            best_window = window
    coherent = title_terms | best_window
    salient = specific - BROAD_CONTEXT_TERMS

    count = len(specific)
    required = 1 if count == 1 else 2 if count <= 4 else max(3, math.ceil(count * 0.4))
    coverage = len(matched) / count
    coherent_coverage = len(coherent) / count
    confident = (
        len(coherent) >= required
        and coverage >= (0.5 if count <= 4 else 0.4)
        and coherent_coverage >= (0.5 if count <= 4 else 0.35)
        and (not salient or bool(coherent & salient))
    )

    reasons: list[str] = []
    if len(matched) < required:
        reasons.append(
            f"only {len(matched)}/{count} specific query terms were supported"
        )
    elif len(coherent) < required:
        reasons.append("specific terms appeared only as scattered description overlap")
    elif not confident:
        if salient and not (coherent & salient):
            reasons.append("only broad context matched; the distinctive intent was absent")
        else:
            reasons.append("query-specific evidence coverage was too weak")

    return RelevanceAssessment(
        intent=intent,
        matched_terms=frozenset(matched),
        coherent_terms=frozenset(coherent),
        title_terms=frozenset(title_terms),
        required_terms=required,
        confident_topic=confident,
        reasons=tuple(reasons),
    )


def query_evidence_items(assessment: RelevanceAssessment) -> list[MatchEvidence]:
    if not assessment.confident_topic:
        return []
    items: list[MatchEvidence] = []
    for kind, terms in (
        ("query_domain", assessment.intent.domain_terms),
        ("query_skill", assessment.intent.skill_terms),
        ("query_task", assessment.intent.task_terms),
    ):
        matched = sorted(set(terms) & set(assessment.coherent_terms))
        if matched:
            items.append(MatchEvidence(
                kind=kind,
                value=", ".join(matched),
                detail=f"coherent {kind.removeprefix('query_')} evidence: " + ", ".join(matched),
                weight=min(0.06, 0.02 + 0.01 * len(matched)),
            ))
    return items


def enforce_recommendation_confidence(
    match: Match,
    preferences: Preferences,
    query: str,
) -> Match:
    """Apply one deterministic recommendation policy to every ranking path.

    This function never removes a candidate. It only separates evidence-backed
    recommendations from plausible candidates that still need verification.
    """
    if not query.strip():
        return match

    assessment = assess_query_relevance(match.job, preferences, query)
    role_items = [item for item in match.evidence if item.kind == "role"]
    title_role = any(item.detail.startswith("title matched") for item in role_items)
    description_role = bool(role_items) and not title_role
    uncertain_reasons: list[str] = []

    job_leadership = bool(LEADERSHIP_TITLE_RE.search(match.job.title))
    if assessment.intent.leadership and not job_leadership:
        uncertain_reasons.append("leadership intent does not match an individual-contributor title")
    if assessment.intent.individual_contributor and job_leadership:
        uncertain_reasons.append("individual-contributor intent does not match a leadership title")

    if preferences.roles and not role_items:
        uncertain_reasons.append("requested role was not verified")
    elif preferences.roles and not assessment.intent.specific_terms and description_role:
        # For a broad role-only search, a generic title cannot become a verified
        # recommendation from one specialist phrase buried in the description.
        uncertain_reasons.append("role appears only in the description, not the job title")

    if assessment.intent.specific_terms and not assessment.confident_topic:
        uncertain_reasons.append("query-specific relevance not verified")
        uncertain_reasons.extend(assessment.reasons)

    if (
        assessment.intent.specific_terms
        and GENERIC_ENGINEERING_TITLE_RE.search(match.job.title)
        and not assessment.confident_topic
    ):
        uncertain_reasons.append("generic engineering title lacks coherent task evidence")

    entry_level_target = is_entry_level_target(preferences, query)
    if entry_level_target and SENIORITY_TITLE_RE.search(match.job.title):
        uncertain_reasons.append("seniority title conflicts with entry-level target")

    min_floor = extract_experience_floor(query)
    if min_floor is not None:
        if match.job.min_experience_years is None:
            uncertain_reasons.append(
                f"minimum experience target ({min_floor:g}+ years) not verified in job metadata"
            )
        elif match.job.min_experience_years < min_floor:
            uncertain_reasons.append(
                f"stated experience ({match.job.min_experience_years:g}+ years) is below requested {min_floor:g}+ years"
            )

    if uncertain_reasons:
        match.status = "needs_verification"
        for reason in uncertain_reasons:
            if reason not in match.risks:
                match.risks.append(reason)
    return match
