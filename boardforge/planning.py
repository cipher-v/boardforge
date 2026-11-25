from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


SEARCH_VERB_CUES = re.compile(
    r"\b(find|search|show|explore|look\s+for|list)\b",
    re.I,
)
SEARCH_CUES = re.compile(
    r"\b(find|search|show|explore|look\s+for|list|openings?|alternatives?|anything|"
    r"what matches|what are|any good|similar|jobs?|roles?|positions?|opportunities|careers?)\b",
    re.I,
)
DURABLE_PREFERENCE_CUES = re.compile(
    r"\b(from now on|going forward|in future searches|permanently|anymore|"
    r"save\s+(?:this\s+|my\s+)?preferences?|update\s+(?:my\s+)?preferences?|"
    r"remember(?:\s+that)?|changed my mind|salary matters|no minimum salary|"
    r"lower priority|focus on|(?:is|are)\s+(?:okay|fine|good)(?:\s+(?:now|too))?)\b",
    re.I,
)
FIRST_PERSON_PREFERENCE_CUES = re.compile(
    r"\bi\s+(?:only\s+|really\s+|also\s+)?(?:want|need|prefer|don't\s+want|dont\s+want|"
    r"do\s+not\s+want|am\s+okay\s+with|'m\s+okay\s+with|would\s+like)\b",
    re.I,
)
CLAUSE_PREFERENCE_CUES = re.compile(
    r"\b(do not|don't|dont|not|no\s+unpaid|avoid|exclude|remove|stop showing|"
    r"only|minimum|at least|worldwide|onsite\s+only|on-site\s+only|remote\s+only|"
    r"onsite-only|remote-only)\b",
    re.I,
)
STANDALONE_DIRECTIVE_START_RE = re.compile(
    r"^\s*(?:please\s+)?(?:do\s+not|don't|dont|stop\s+showing|avoid|exclude|remove|"
    r"not\b|no\s+unpaid\b|minimum\s+\d|at\s+least\s+\d)",
    re.I,
)
NO_EXPERIENCE_PHRASE_RE = re.compile(
    r"\b(?:(?:i\s+have|i've\s+got|with|having)\s+)?(?:no|zero|without)\s+"
    r"(?:prior\s+|formal\s+)?(?:employment\s+|work\s+|professional\s+|industry\s+)?experience\b",
    re.I,
)
NEGATED_SHOW_CUES = re.compile(r"\b(do not|don't|dont|stop)\s+show(?:ing)?\b", re.I)
FEEDBACK_CUES = re.compile(r"\b(reject|ignore|save|applied|interview|mark)\b", re.I)
JOB_ID_CUE = re.compile(r"\b[\w-]+:[\w-]+:[\w-]+\b")
PREFERENCE_THEN_SEARCH_SPLIT_RE = re.compile(
    r"(?:[.;]\s*|(?:,\s*(?:now\s+|then\s+|and\s+)?)|\s+\band\s+)"
    r"(?=(?:now\s+|then\s+|please\s+)?(?:find|search|show|explore|look\s+for|list)\b)",
    re.I,
)


@dataclass(frozen=True)
class PlannedTool:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


def _feedback_verdict(text: str) -> str:
    low = text.casefold()
    if "reject" in low:
        return "reject"
    if "ignore" in low:
        return "ignore"
    if "interview" in low:
        return "interview"
    if "applied" in low or re.search(r"\bmark\b.*\bapplied\b", low):
        return "applied"
    return "save"


def _is_explicit_preference_update(text: str) -> bool:
    """Return True only when the message explicitly states a durable preference change.

    Constraints inside a search request (e.g. 'Find onsite-only backend roles in
    Bengaluru', 'Software engineering jobs for candidates with no formal employment
    experience', 'Find AI roles, avoid hardware') are transient search overlays and
    must not mutate saved profile preferences.
    """
    cleaned = NO_EXPERIENCE_PHRASE_RE.sub(" ", text).strip()
    if not cleaned:
        return False
    if NEGATED_SHOW_CUES.search(cleaned) or DURABLE_PREFERENCE_CUES.search(cleaned):
        return True
    if FIRST_PERSON_PREFERENCE_CUES.search(cleaned):
        return True

    has_search_verb = bool(SEARCH_VERB_CUES.search(cleaned))
    if not has_search_verb and STANDALONE_DIRECTIVE_START_RE.search(cleaned):
        return True

    # Support explicit two-part turns such as "Only India roles. Show me AI jobs"
    # or "Minimum 10 LPA, show backend jobs", where a dedicated preference clause
    # precedes an explicit search command, without treating trailing search
    # modifiers ("Find AI roles, avoid hardware") as durable updates.
    parts = PREFERENCE_THEN_SEARCH_SPLIT_RE.split(cleaned, maxsplit=1)
    if len(parts) == 2:
        leading_clause = parts[0].strip()
        if (
            leading_clause
            and not SEARCH_VERB_CUES.search(leading_clause)
            and not (FEEDBACK_CUES.search(leading_clause) and JOB_ID_CUE.search(leading_clause))
            and CLAUSE_PREFERENCE_CUES.search(leading_clause)
        ):
            return True
    return False


def _is_search_request(text: str, *, preference_updated: bool, feedback_recorded: bool) -> bool:
    if NEGATED_SHOW_CUES.search(text):
        # Strip negated 'do not show ...' clauses and check whether a positive search remains.
        remaining = re.sub(
            r"\b(?:do not|don't|dont|stop)\s+show(?:ing)?\b[^.;,]*",
            " ",
            text,
            flags=re.I,
        )
        return bool(SEARCH_VERB_CUES.search(remaining) or re.search(
            r"\b(openings?|alternatives?|similar)\b", remaining, re.I
        ))
    if SEARCH_VERB_CUES.search(text):
        return True
    if preference_updated or feedback_recorded:
        # Pure preference or feedback statements (e.g. "Avoid hardware-heavy jobs",
        # "From now on I only want remote roles", "Reject x:y:z") should not trigger
        # an unrequested search unless they include a positive search cue beyond the
        # object of the preference/feedback directive.
        return bool(re.search(
            r"\b(openings?|alternatives?|anything|what matches|what are|any good|similar)\b",
            text,
            re.I,
        ))
    return bool(SEARCH_CUES.search(text))


def plan_tools(text: str, *, top_k: int = 10) -> list[PlannedTool]:
    """Plan deterministic fallback tools with executable arguments."""
    tools: list[PlannedTool] = []
    job_id_match = JOB_ID_CUE.search(text)
    feedback_recorded = bool(FEEDBACK_CUES.search(text) and job_id_match)
    if feedback_recorded and job_id_match is not None:
        tools.append(PlannedTool(
            "record_feedback",
            {"job_id": job_id_match.group(0), "verdict": _feedback_verdict(text), "note": ""},
        ))

    preference_updated = _is_explicit_preference_update(text)
    if preference_updated:
        tools.append(PlannedTool("update_preferences", {"text": text}))

    if _is_search_request(
        text,
        preference_updated=preference_updated,
        feedback_recorded=feedback_recorded,
    ):
        tools.append(PlannedTool("search_jobs", {"query": text, "top_k": top_k}))
    if not tools:
        tools.append(PlannedTool("search_jobs", {"query": text, "top_k": top_k}))
    return tools


def tool_names(text: str) -> list[str]:
    return [tool.name for tool in plan_tools(text)]
