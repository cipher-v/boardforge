"""Configurable source quarantine and conservative placeholder/test-vacancy detection."""
from __future__ import annotations

import os
import re

DEFAULT_QUARANTINED_SOURCES = "lever:jobtwine"


def quarantined_sources() -> set[tuple[str | None, str]]:
    """Parse BOARDFORGE_QUARANTINED_SOURCES into normalized (kind_or_none, key) pairs.

    Defaults to 'lever:jobtwine' when unset. Setting BOARDFORGE_QUARANTINED_SOURCES to
    '', 'none', or 'off' disables quarantine entries.
    """
    raw = os.getenv("BOARDFORGE_QUARANTINED_SOURCES")
    if raw is None:
        raw = DEFAULT_QUARANTINED_SOURCES
    cleaned = raw.strip()
    if not cleaned or cleaned.casefold() in {"none", "off", "0", "false"}:
        return set()

    entries: set[tuple[str | None, str]] = set()
    for token in re.split(r"[,;\s]+", cleaned):
        part = token.strip().casefold()
        if not part:
            continue
        if ":" in part:
            kind, key = part.split(":", 1)
            kind = kind.strip() or None
            key = key.strip()
            if key:
                entries.add((kind, key))
        else:
            entries.add((None, part))
    return entries


def quarantined_sources_list() -> list[str]:
    return sorted(
        f"{kind}:{key}" if kind else key
        for kind, key in quarantined_sources()
    )


def is_quarantined_source(source_kind: str, source_key: str) -> bool:
    kind_low = (source_kind or "").strip().casefold()
    key_low = (source_key or "").strip().casefold()
    if not key_low:
        return False
    configured = quarantined_sources()
    if not configured:
        return False
    return (kind_low, key_low) in configured or (None, key_low) in configured


_BARE_PLACEHOLDER_TITLE_RE = re.compile(
    r"^\s*(?:test|testing|dummy|sample|placeholder|mock|sandbox|demo|temp|tbd|"
    r"ignore|delete\s+me)(?:\s*[-#]?\s*\d+)?\s*$",
    re.I,
)
_EXPLICIT_PLACEHOLDER_PHRASE_RE = re.compile(
    r"(?:"
    r"\b(?:do\s+not\s+apply|don['’]?t\s+apply|internal\s+test\s+(?:only|job|role|posting|opening|requisition))\b|"
    r"\b(?:dummy|placeholder|fake|sandbox|sample)\s+(?:job|role|posting|opening|position|vacancy|requisition|listing)\b|"
    r"\b(?:add\s+)?test(?:ing)?\s+(?:job|posting|requisition|vacancy|listing|opening)\b|"
    r"\b(?:lever|greenhouse|ashby|kombo)\s+test(?:ing)?\b|"
    r"\btest(?:ing)?\s+(?:lever|greenhouse|ashby|kombo)\b|"
    r"\bmock\s+(?:job|role|posting|vacancy|test|listing)\b|"
    r"^\s*\d{1,2}(?:st|nd|rd|th)\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s*$|"
    r"^\s*(?:test|testing|dummy|mock|sample)\s+\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s*$|"
    r"\b\d{1,2}:\d{2}\s*$"
    r")",
    re.I,
)


def is_placeholder_job_title(title: str) -> bool:
    """Conservatively detect obvious internal ATS test/placeholder/mock postings.

    Preserves legitimate QA, Software Test Engineer, SDET, Penetration Tester,
    and Automation Testing roles.
    """
    normalized = " ".join((title or "").split())
    if not normalized:
        return True
    if _BARE_PLACEHOLDER_TITLE_RE.match(normalized):
        return True
    return bool(_EXPLICIT_PLACEHOLDER_PHRASE_RE.search(normalized))
