from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


SEARCH_CUES = re.compile(
    r"\b(find|search|show|openings?|alternatives?|anything|what matches|what are|any good|similar)\b",
    re.I,
)
PREFERENCE_CUES = re.compile(
    r"\b(do not|don't|dont|not|no|avoid|exclude|remove|stop showing|only|from now on|minimum|at least|"
    r"changed my mind|salary matters|okay with|lower priority|focus on|want internships|want remote|"
    r"worldwide|onsite|hybrid)\b",
    re.I,
)
NEGATED_SHOW_CUES = re.compile(r"\b(do not|don't|dont|stop)\s+show(?:ing)?\b", re.I)
FEEDBACK_CUES = re.compile(r"\b(reject|ignore|save|applied|interview|mark)\b", re.I)
JOB_ID_CUE = re.compile(r"\b[\w-]+:[\w-]+:[\w-]+\b")


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


def plan_tools(text: str, *, top_k: int = 10) -> list[PlannedTool]:
    """Plan deterministic fallback tools with executable arguments."""
    tools: list[PlannedTool] = []
    job_id_match = JOB_ID_CUE.search(text)
    if FEEDBACK_CUES.search(text) and job_id_match:
        tools.append(PlannedTool(
            "record_feedback",
            {"job_id": job_id_match.group(0), "verdict": _feedback_verdict(text), "note": ""},
        ))
    preference_text = re.sub(
        r"\bi have\s+no\s+(?:formal\s+)?(?:employment\s+|work\s+|professional\s+)?experience\b",
        " ", text, flags=re.I,
    )
    if PREFERENCE_CUES.search(preference_text):
        tools.append(PlannedTool("update_preferences", {"text": text}))

    search_requested = bool(SEARCH_CUES.search(text))
    if NEGATED_SHOW_CUES.search(text):
        search_requested = bool(re.search(r"\b(find|search|show me|openings?|alternatives?|similar)\b", text, re.I))
    if search_requested:
        tools.append(PlannedTool("search_jobs", {"query": text, "top_k": top_k}))
    if not tools:
        tools.append(PlannedTool("search_jobs", {"query": text, "top_k": top_k}))
    return tools


def tool_names(text: str) -> list[str]:
    return [tool.name for tool in plan_tools(text)]
