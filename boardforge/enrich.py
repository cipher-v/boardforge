from __future__ import annotations

import re

import pycountry

from boardforge.models import Job, RemotePolicy


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
US_STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming", "district of columbia",
}
COUNTRY_ALIASES = {
    "south korea": "Korea, Republic of",
    "north korea": "Korea, Democratic People's Republic of",
    "russia": "Russian Federation",
    "vietnam": "Viet Nam",
    "taiwan": "Taiwan, Province of China",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "uae": "United Arab Emirates",
}


def _explicit_countries(location: str) -> list[str]:
    normalized = " ".join(location.casefold().split())
    found: list[str] = []

    for alias, canonical in COUNTRY_ALIASES.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized):
            found.append(canonical)

    for country in pycountry.countries:
        names = {
            country.name,
            getattr(country, "official_name", ""),
            getattr(country, "common_name", ""),
        }
        for name in names:
            if not name:
                continue
            if re.search(
                rf"(?<!\w){re.escape(name.casefold())}(?!\w)",
                normalized,
            ):
                found.append(country.name)
                break

    return list(dict.fromkeys(found))


def infer_country(location: str) -> str:
    """Infer one country conservatively from the structured location only.

    Multi-country locations intentionally remain unknown because Job.country is
    currently a single-valued field.
    """
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
    if any(
        re.search(rf"\b{re.escape(state)}\b", normalized)
        for state in US_STATE_NAMES
    ):
        return "USA"

    explicit = _explicit_countries(location)
    if len(explicit) != 1:
        return ""
    country = explicit[0]
    if country == "United States":
        return "USA"
    return country


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


_NEGATED_REMOTE_RE = re.compile(
    r"\b(?:not\s+(?:a\s+)?remote|non[- ]remote|no\s+remote|cannot\s+be\s+remote|"
    r"unable\s+to\s+offer\s+remote)\b",
    re.I,
)
_TITLE_HYBRID_TECH_RE = re.compile(
    r"\bhybrid\s+(?:cloud|search|retrieval|rag|model|models|ranking|index|systems?)\b",
    re.I,
)


def infer_remote_policy(location: str, title: str, description: str) -> RemotePolicy:
    """Infer workplace policy conservatively, prioritizing structured location and title.

    Company boilerplate frequently mentions 'remote-first culture', 'remote teams',
    or 'remote code execution'. Those incidental mentions must never override a
    structured Hybrid/Onsite location or turn an unspecified office role into remote.
    """
    loc = " ".join(location.lower().split())
    ttl = " ".join(title.lower().split())
    ttl_workplace = _TITLE_HYBRID_TECH_RE.sub(" ", ttl)

    # 1. Structured location and title take strict precedence over description prose.
    if re.search(r"\bhybrid\b", loc) or re.search(r"\bhybrid\b", ttl_workplace):
        return "hybrid"
    if (
        re.search(r"\b(?:on[- ]?site|in[- ]office|work from office|wfo)\b", f"{loc} {ttl_workplace}")
        or (re.search(r"\boffice\b", loc) and not re.search(r"\bremote\b", loc))
    ):
        return "onsite"
    if (
        (re.search(r"\bremote\b", loc) and not _NEGATED_REMOTE_RE.search(loc))
        or (re.search(r"\bremote\b", ttl_workplace) and not _NEGATED_REMOTE_RE.search(ttl_workplace))
    ):
        return "remote"

    # 2. Fall back to explicit role-level workplace statements in the description.
    desc = " ".join(description.lower().split())
    if not desc:
        return "unknown"
    desc_unnegated = _NEGATED_REMOTE_RE.sub(" ", desc)
    if re.search(
        r"\b(?:hybrid\s+(?:work(?:place|ing)?|role|position|schedule|model|environment)|"
        r"workplace\s*(?:type)?\s*:\s*hybrid|\d+\s+days?\s+(?:a\s+week\s+|per\s+week\s+)?in\s+(?:the\s+)?office)\b",
        desc_unnegated,
    ):
        return "hybrid"
    if re.search(
        r"\b(?:on[- ]?site\s+(?:only|role|position|work)|in[- ]office\s+(?:only|role|position)|"
        r"work\s+from\s+office|wfo|workplace\s*(?:type)?\s*:\s*on[- ]?site)\b",
        desc_unnegated,
    ):
        return "onsite"
    if re.search(
        r"\b(?:remote\s+(?:role|position|job|opportunity|eligible|work\s+eligible)|"
        r"fully\s+remote|100%\s+remote|work\s+from\s+anywhere|"
        r"workplace\s*(?:type)?\s*:\s*remote|location\s*:\s*remote)\b",
        desc_unnegated,
    ):
        return "remote"
    return "unknown"


_EXP_NUMBER_WORDS = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "thirteen": 13.0,
    "fourteen": 14.0,
    "fifteen": 15.0,
    "sixteen": 16.0,
    "seventeen": 17.0,
    "eighteen": 18.0,
    "nineteen": 19.0,
    "twenty": 20.0,
}
_EXP_WORD_ALT = "|".join(sorted(_EXP_NUMBER_WORDS, key=len, reverse=True))
_EXP_NUM_TOKEN = (
    rf"(?:\d+(?:\.\d+)?|(?:{_EXP_WORD_ALT})(?:\s*\(\s*\d+(?:\.\d+)?\s*\))?)"
)

_INLINE_SECTION_BREAK_RE = re.compile(
    r"(?<!\n)(?=(?:\*+\s*)?\b(?:"
    r"required\s+skills\s+(?:&|and)\s+experience|"
    r"required\s+qualifications|minimum\s+qualifications|basic\s+qualifications|"
    r"mandatory\s+qualifications|specific\s+requirements|"
    r"what\s+we(?:'re|’re|\s+are)?\s+looking\s+for|"
    r"what\s+we\s+look\s+for(?:\s+in\s+you)?|"
    r"what\s+you(?:'ll|’ll|\s+will)?\s+(?:bring|need)|"
    r"preferred\s+(?:qualifications|skills|experience)|"
    r"desired\s+(?:qualifications|skills|experience)|"
    r"nice[- ]to[- ]haves?|good[- ]to[- ]haves?|"
    r"bonus\s+points(?:\s+for)?|"
    r"(?:required|desired|preferred|must[- ]haves?|requirements)\s*:|"
    r"(?<!preferred\s)(?<!desired\s)(?<!optional\s)(?<!additional\s)"
    r"(?<!required\s)(?<!minimum\s)(?<!basic\s)(?<!mandatory\s)qualifications\s*:"
    r"))",
    re.I,
)
_REQUIRED_HEADING_RE = re.compile(
    r"^\s*[\*\#\-\•\·\s]*(?:"
    r"requirements?|required(?:\s+skills(?:\s+(?:&|and)\s+experience)?)?(?:\s+qualifications)?|"
    r"minimum\s+qualifications|basic\s+qualifications|mandatory\s+qualifications|"
    r"specific\s+requirements|must[- ]haves?|"
    r"what\s+we(?:'re|’re|\s+are)?\s+looking\s+for|"
    r"what\s+we\s+look\s+for(?:\s+in\s+you)?(?:\s*\([^)]*\))?|"
    r"what\s+you(?:'ll|’ll|\s+will)?\s+(?:bring|need)|"
    r"who\s+you\s+are|skills\s*(?:&|and)\s*experience|"
    r"experience\s*(?:&|and)\s*qualifications|qualifications"
    r")(?:\s*:|\s*$)",
    re.I,
)
_PREFERRED_HEADING_RE = re.compile(
    r"^\s*[\*\#\-\•\·\s]*(?:"
    r"preferred(?:\s+qualifications|\s+skills|\s+experience)?|"
    r"nice[- ]to[- ]haves?|good[- ]to[- ]haves?|"
    r"desired(?:\s+qualifications|\s+skills|\s+experience)?|"
    r"bonus(?:\s+points(?:\s+for)?)?|pluses|optional(?:\s+qualifications)?|"
    r"additional\s+qualifications"
    r")(?:\s*:|\s*$)",
    re.I,
)
_IGNORED_HEADING_RE = re.compile(
    r"^\s*[\*\#\-\•\·\s]*(?:"
    r"about\s+(?:us|the\s+company|the\s+interview|[a-z0-9]+)|"
    r"why\s+(?:join\s+us|[a-z0-9]+)|benefits|compensation(?:\s*(?:&|and)\s*benefits)?|"
    r"what\s+you(?:'ll|’ll|\s+will)?\s+do|responsibilities|role\s+overview"
    r")(?:\s*:|\s*$)",
    re.I,
)
_OPTIONAL_PREFIX_RE = re.compile(
    r"(?:(?:\b(?:preferred|desired|optional|nice[- ]to[- ]have|good[- ]to[- ]have|bonus|plus)\s*[:\-–]\s*)|"
    r"(?:\b(?:ideally|preferably|typically)(?:\s+have|\s+with)?[,:\s]*))$",
    re.I,
)
_OPTIONAL_DIRECT_SUFFIX_RE = re.compile(
    r"^(?:\s+(?:of\s+)?(?:[a-z0-9/+,&'’-]+\s+){0,6}?(?:"
    r"is\s+(?:a\s+)?(?:plus|bonus|preferred|nice[- ]to[- ]have)|"
    r"preferred(?=\s*(?:[.,;)]|$))|"
    r"\(not\s+required[^)]*\)|not\s+required\b))",
    re.I,
)
_NEGATED_EXP_PREFIX_RE = re.compile(
    r"\b(?:don['’]?t\s+care\s+about|do\s+not\s+care\s+about|no\s+need\s+for|"
    r"not\s+requiring|without\s+requiring)\s*$",
    re.I,
)
_COMPANY_BOILERPLATE_PREFIX_RE = re.compile(
    r"\b(?:founded|established|for\s+over|over\s+the\s+(?:next|past|last)|"
    r"in\s+the\s+(?:next|past|last)|for\s+the\s+(?:next|past|last)|"
    r"within\s+(?:the\s+)?first|in\s+your\s+first|over\s+your\s+first|"
    r"by\s+end\s+of\s+year|we\s+have\s+over|our\s+team\s+brings|"
    r"doing\s+this\s+work\s+for|backed\s+by|serving\s+customers\s+for)\s*$",
    re.I,
)
_NON_EXPERIENCE_SUFFIX_RE = re.compile(
    r"^\s*(?:ago\b|old\b|of\s+age\b|vesting\b|cliff\b|warranty\b|degree\b|"
    r"bachelor|master|undergraduate|program\b|plan\b|contract\b|lease\b|per\s+year\b)",
    re.I,
)
_EXPERIENCE_CONTEXT_RE = re.compile(
    r"\b(?:experience|exp\.?|yoe|background|track\s+record|hands[- ]on|practitioner|"
    r"in\s+[a-z]|working\s+(?:with|in|on)|building|developing|designing|selling|managing|leading)\b",
    re.I,
)
_UPPER_ONLY_EXP_RE = re.compile(
    rf"\b(?:up\s+to|at\s+most|no\s+more\s+than|under|less\s+than)\s+"
    rf"(?P<max>{_EXP_NUM_TOKEN})\s*(?:years?|yrs?)\s+(?:of\s+)?"
    rf"(?:[a-z][a-z0-9/+,&'’-]*,?\s+){{0,5}}?(?:experience|exp\.?)\b",
    re.I,
)
_RANGE_EXP_RE = re.compile(
    rf"(?<![\w\-–—.])(?P<min>{_EXP_NUM_TOKEN})\s*(?:-|–|—|to)\s*"
    rf"(?P<max>{_EXP_NUM_TOKEN})\s*\+?\s*(?:years?|yrs?|yoe)\b",
    re.I,
)
_PLUS_EXP_RE = re.compile(
    rf"(?<![\w\-–—.])(?P<min>{_EXP_NUM_TOKEN})\s*(?:\+|\bplus\b)\s*(?:years?|yrs?|yoe)\b|"
    rf"(?<![\w\-–—.])(?P<min2>{_EXP_NUM_TOKEN})\s*(?:years?|yrs?)\+",
    re.I,
)
_FLOOR_WORD_EXP_RE = re.compile(
    rf"\b(?P<kw>(?:a\s+)?minimum(?:\s+of)?|min\.?(?:\s+of)?|at\s+least|more\s+than|over|above)\s+"
    rf"(?P<min>{_EXP_NUM_TOKEN})\s*(?!(?:-|–|—|to)\s*\d)\+?\s*(?:years?|yrs?|yoe)\b",
    re.I,
)
_LABELED_EXP_RE = re.compile(
    rf"\b(?:required\s+|minimum\s+|total\s+|relevant\s+|professional\s+|work\s+|overall\s+)?"
    rf"experience(?:\s+required)?\s*[:\-–]\s*(?P<min>{_EXP_NUM_TOKEN})"
    rf"(?:\s*(?:-|–|—|to)\s*(?P<max>{_EXP_NUM_TOKEN}))?\s*\+?\s*(?:years?|yrs?)?\b",
    re.I,
)
_PLAIN_YEARS_EXP_RE = re.compile(
    rf"(?<![\w\-–—.])(?P<min>{_EXP_NUM_TOKEN})\s*(?:years?|yrs?)\s+(?:of\s+)?"
    rf"(?:[a-z][a-z0-9/+,&'’-]*,?\s+){{0,6}}?(?:experience|exp\.?)\b",
    re.I,
)


def _parse_exp_number(raw: str) -> float:
    cleaned = re.sub(r"\(\s*\d+(?:\.\d+)?\s*\)", "", raw).strip().casefold()
    return _EXP_NUMBER_WORDS[cleaned] if cleaned in _EXP_NUMBER_WORDS else float(cleaned)


def _is_disqualified_exp_span(text: str, start: int, end: int) -> bool:
    prefix = text[max(0, start - 45):start]
    suffix = text[end:end + 80]
    return bool(
        _COMPANY_BOILERPLATE_PREFIX_RE.search(prefix)
        or _NEGATED_EXP_PREFIX_RE.search(prefix)
        or _OPTIONAL_PREFIX_RE.search(prefix)
        or _NON_EXPERIENCE_SUFFIX_RE.search(suffix)
        or _OPTIONAL_DIRECT_SUFFIX_RE.search(suffix)
    )


def extract_job_experience(
    title: str, description: str
) -> tuple[float | None, float | None]:
    """Deterministically extract required (min_years, max_years) from title and JD.

    Prioritizes explicit requirement sections over general prose, ignores preferred /
    nice-to-have / bonus sections and optional clauses, and filters out company age,
    roadmap ('next 3 years'), onboarding ('first year'), degree duration, and age/vesting.
    """
    segmented = _INLINE_SECTION_BREAK_RE.sub("\n", description or "")
    lines = [title or "", *segmented.splitlines()]
    section = "general"
    by_bucket: dict[str, list[tuple[float, float | None]]] = {
        "required": [],
        "general": [],
    }

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        if idx > 0:
            m_pref = _PREFERRED_HEADING_RE.match(line)
            if m_pref and (len(line) < 90 or ":" in m_pref.group(0)):
                section = "preferred"
                continue
            m_req = _REQUIRED_HEADING_RE.match(line)
            if m_req and (len(line) < 90 or ":" in m_req.group(0)):
                section = "required"
                line = line[m_req.end():].strip()
                if not line:
                    continue
            else:
                m_ign = _IGNORED_HEADING_RE.match(line)
                if m_ign and (len(line) < 90 or ":" in m_ign.group(0)):
                    section = "ignored"
                    line = line[m_ign.end():].strip()
                    if not line:
                        continue

        if section == "preferred":
            continue

        clauses = re.split(r"(?<=[.!?])\s+|\s+[•·▪◦]\s+", line)
        for clause in clauses:
            c = clause.strip()
            if not c:
                continue
            m_clause_pref = _PREFERRED_HEADING_RE.match(c)
            if m_clause_pref and (len(c) < 90 or ":" in m_clause_pref.group(0)):
                section = "preferred"
                break

            target_bucket = "required" if section == "required" else "general"
            spans_used: list[tuple[int, int]] = []
            clause_matches: list[tuple[int, int, float, float | None]] = []

            def _overlaps(s: int, e: int) -> bool:
                return any(not (e <= us or s >= ue) for us, ue in spans_used)

            for um in _UPPER_ONLY_EXP_RE.finditer(c):
                if _is_disqualified_exp_span(c, um.start(), um.end()):
                    continue
                max_v = _parse_exp_number(um.group("max"))
                if 0.0 <= max_v <= 25.0:
                    clause_matches.append((um.start(), um.end(), 0.0, max_v))
                    spans_used.append((um.start(), um.end()))

            for m in _RANGE_EXP_RE.finditer(c):
                if _overlaps(m.start(), m.end()) or _is_disqualified_exp_span(c, m.start(), m.end()):
                    continue
                lo, hi = _parse_exp_number(m.group("min")), _parse_exp_number(m.group("max"))
                if 0.0 <= lo <= hi <= 25.0:
                    if section == "ignored" and not _EXPERIENCE_CONTEXT_RE.search(c[m.start():m.end() + 55]):
                        continue
                    clause_matches.append((m.start(), m.end(), lo, hi))
                    spans_used.append((m.start(), m.end()))

            for m in _PLUS_EXP_RE.finditer(c):
                if _overlaps(m.start(), m.end()) or _is_disqualified_exp_span(c, m.start(), m.end()):
                    continue
                val = _parse_exp_number(m.group("min") or m.group("min2"))
                if 0.0 <= val <= 25.0:
                    if section == "ignored" and not _EXPERIENCE_CONTEXT_RE.search(c[m.start():m.end() + 55]):
                        continue
                    clause_matches.append((m.start(), m.end(), val, None))
                    spans_used.append((m.start(), m.end()))

            for m in _FLOOR_WORD_EXP_RE.finditer(c):
                if _overlaps(m.start(), m.end()) or _is_disqualified_exp_span(c, m.start(), m.end()):
                    continue
                prefix = c[max(0, m.start() - 35):m.start()]
                suffix = c[m.end():m.end() + 55]
                kw = m.group("kw").casefold()
                if kw in {"over", "more than", "above"}:
                    if re.search(r"\bfor\s*$", prefix, re.I):
                        continue
                    if section != "required" and not _EXPERIENCE_CONTEXT_RE.search(suffix):
                        continue
                val = _parse_exp_number(m.group("min"))
                if 0.0 <= val <= 25.0:
                    clause_matches.append((m.start(), m.end(), val, None))
                    spans_used.append((m.start(), m.end()))

            for lm in _LABELED_EXP_RE.finditer(c):
                if _overlaps(lm.start(), lm.end()) or _is_disqualified_exp_span(c, lm.start(), lm.end()):
                    continue
                lo = _parse_exp_number(lm.group("min"))
                hi = _parse_exp_number(lm.group("max")) if lm.group("max") else None
                if 0.0 <= lo <= 25.0 and (hi is None or lo <= hi <= 25.0):
                    clause_matches.append((lm.start(), lm.end(), lo, hi))
                    spans_used.append((lm.start(), lm.end()))

            for m in _PLAIN_YEARS_EXP_RE.finditer(c):
                if _overlaps(m.start(), m.end()) or _is_disqualified_exp_span(c, m.start(), m.end()):
                    continue
                prefix = c[max(0, m.start() - 35):m.start()]
                if re.search(r"\b(?:up\s+to|at\s+most|under|less\s+than|no\s+more\s+than)\s*$", prefix, re.I):
                    continue
                val = _parse_exp_number(m.group("min"))
                if 0.0 <= val <= 25.0 and section != "ignored":
                    clause_matches.append((m.start(), m.end(), val, None))
                    spans_used.append((m.start(), m.end()))

            if not clause_matches:
                continue
            clause_matches.sort(key=lambda item: item[0])
            if len(clause_matches) >= 2:
                between = c[clause_matches[0][1]:clause_matches[1][0]]
                if re.search(r"\bor\b", between, re.I) and not re.search(r"\b(?:and|with)\b", between, re.I):
                    low_min = min(item[2] for item in clause_matches)
                    low_maxes = [item[3] for item in clause_matches if item[2] == low_min and item[3] is not None]
                    by_bucket[target_bucket].append((low_min, max(low_maxes) if low_maxes else None))
                    continue
            for _, _, lo, hi in clause_matches:
                by_bucket[target_bucket].append((lo, hi))

    chosen = by_bucket["required"] or by_bucket["general"]
    if not chosen:
        return None, None
    best_min = max(item[0] for item in chosen)
    max_candidates = [item[1] for item in chosen if item[0] == best_min and item[1] is not None]
    best_max = max(max_candidates) if max_candidates else None
    return best_min, best_max


_EXPLICIT_FRESHER_TITLE_RE = re.compile(
    r"\b(?:new\s+grad(?:uate)?s?|recent\s+graduates?|graduate\s+engineer(?:\s+trainee)?|"
    r"campus\s+hire|fresher|entry[- ]level)\b",
    re.I,
)
_EXPLICIT_FRESHER_DESC_RE = re.compile(
    r"\b(?:"
    r"(?:no|zero)\s+(?:prior\s+|previous\s+|formal\s+|professional\s+|work\s+|industry\s+)?experience\s+(?:is\s+)?(?:required|needed|necessary)|"
    r"freshers?\s+(?:are\s+)?(?:welcome|encouraged(?:\s+to\s+apply)?|eligible|can\s+apply|may\s+apply|invited|accepted)|"
    r"open\s+to\s+(?:freshers?|recent\s+graduates?|new\s+grads?|entry[- ]level\s+(?:candidates?|engineers?|applicants?))|"
    r"(?:recent\s+graduates?|new\s+grads?)\s+(?:are\s+)?(?:welcome|encouraged\s+to\s+apply|eligible|can\s+apply|may\s+apply)"
    r")\b",
    re.I,
)
_EARLY_CAREER_SIGNAL_DESC_RE = re.compile(
    r"\b(?:"
    r"freshers?\s+who\s+joined|"
    r"(?:compensation|role|opportunity|suited|ideal|designed)\s+for\s+(?:early[- ]career|freshers?|recent\s+graduates?)|"
    r"early[- ]career\s+(?:profiles?|engineers?|candidates?|talent)"
    r")\b",
    re.I,
)
_FRESHER_NEGATION_PREFIX_RE = re.compile(
    r"\b(?:not|no|never|neither|cannot|can't|unable\s+to\s+consider|do\s+not\s+apply\s+if)\s+(?:[a-z]+\s+){0,3}$",
    re.I,
)


def extract_fresher_eligibility_evidence(
    title: str, description: str
) -> tuple[str, str] | None:
    """Identify explicit fresher/graduate/no-experience eligibility without fabricating numeric years.

    Returns ('explicit', matched_text) for direct eligibility statements (e.g. 'Freshers welcome',
    'No prior experience required', 'New Grad') or ('early_career_signal', matched_text) for
    contextual early-career cues (e.g. 'Freshers who joined...', 'early-career profiles').
    Returns None when experience is unstated or contradicts entry-level eligibility.
    """
    min_exp, _ = extract_job_experience(title, description)
    if min_exp is not None and min_exp > 1.0:
        return None

    t_norm = " ".join((title or "").split())
    m_title = _EXPLICIT_FRESHER_TITLE_RE.search(t_norm)
    if m_title and not _FRESHER_NEGATION_PREFIX_RE.search(t_norm[max(0, m_title.start() - 30):m_title.start()]):
        return ("explicit", m_title.group(0))

    d_norm = " ".join((description or "").split())
    if not d_norm:
        return None

    for m in _EXPLICIT_FRESHER_DESC_RE.finditer(d_norm):
        prefix = d_norm[max(0, m.start() - 35):m.start()]
        if _FRESHER_NEGATION_PREFIX_RE.search(prefix) or _OPTIONAL_PREFIX_RE.search(prefix):
            continue
        return ("explicit", m.group(0))

    for m in _EARLY_CAREER_SIGNAL_DESC_RE.finditer(d_norm):
        prefix = d_norm[max(0, m.start() - 35):m.start()]
        if _FRESHER_NEGATION_PREFIX_RE.search(prefix):
            continue
        return ("early_career_signal", m.group(0))

    return None


def enrich_job(job: Job) -> Job:
    j = job.model_copy(deep=True)
    text = f"{j.location} {j.title} {j.description}".lower()
    if not j.country:
        j.country = infer_country(j.location)
    if j.remote_policy == "unknown":
        j.remote_policy = infer_remote_policy(j.location, j.title, j.description)
    if j.min_experience_years is None:
        min_exp, max_exp = extract_job_experience(j.title, j.description)
        if min_exp is not None:
            j.min_experience_years = min_exp
            if j.max_experience_years is None:
                j.max_experience_years = max_exp
    _parse_salary(j, text)
    return j


def reenrich_job_metadata(job: Job) -> tuple[Job, bool, str]:
    """Recompute derived country, workplace, and experience metadata safely.

    Greenhouse, Lever, Ashby, and Generic JSON-LD connectors derive
    min_experience_years / max_experience_years exclusively via enrich_job.
    Greenhouse, Lever, and Generic also derive remote_policy exclusively via
    enrich_job, while Ashby may set remote_policy='remote' from upstream isRemote.
    """
    candidate = job.model_copy(deep=True)
    candidate.min_experience_years = None
    candidate.max_experience_years = None
    inferred_policy = infer_remote_policy(candidate.location, candidate.title, candidate.description)
    note = "unchanged"

    if candidate.source_kind in {"greenhouse", "lever", "generic"}:
        candidate.remote_policy = "unknown"
        updated = enrich_job(candidate)
    else:
        if inferred_policy in {"hybrid", "onsite"} and candidate.remote_policy != inferred_policy:
            candidate.remote_policy = inferred_policy
        elif candidate.remote_policy == "unknown" and inferred_policy != "unknown":
            candidate.remote_policy = inferred_policy
        updated = enrich_job(candidate)
        if candidate.source_kind == "ashby" and updated.remote_policy == "remote" and inferred_policy == "unknown":
            note = "ashby_upstream_or_legacy_remote"

    policy_changed = updated.remote_policy != job.remote_policy
    country_changed = updated.country != job.country
    exp_changed = (
        updated.min_experience_years != job.min_experience_years
        or updated.max_experience_years != job.max_experience_years
    )
    changed = policy_changed or country_changed or exp_changed
    if policy_changed:
        note = f"{job.remote_policy}->{updated.remote_policy}"
    elif exp_changed:
        note = f"exp:{job.min_experience_years}->{updated.min_experience_years}"
    elif country_changed:
        note = f"country:{job.country}->{updated.country}"
    return updated, changed, note


def reenrich_store_jobs(store, *, apply: bool = False) -> dict:
    """Inspect or update stored active job payloads in place without touching embeddings or history.

    Because job_text(job) excludes remote_policy and country, updating derived metadata
    in jobs.payload preserves search_document and embedding_hash, keeping all cached
    pgvector embeddings valid. Defaults to dry-run (apply=False).
    """
    import hashlib

    transitions: dict[str, int] = {}
    before_policies = {"remote": 0, "hybrid": 0, "onsite": 0, "unknown": 0}
    after_policies = {"remote": 0, "hybrid": 0, "onsite": 0, "unknown": 0}
    ashby_ambiguous = 0
    changed_rows: list[tuple[str, Job]] = []

    jobs = store.active_jobs()
    for job in jobs:
        before_policies[job.remote_policy] = before_policies.get(job.remote_policy, 0) + 1
        stripped = job.model_copy(update={"first_seen_at": None, "last_seen_at": None})
        updated, changed, note = reenrich_job_metadata(stripped)
        after_policies[updated.remote_policy] = after_policies.get(updated.remote_policy, 0) + 1
        if note == "ashby_upstream_or_legacy_remote":
            ashby_ambiguous += 1
        if changed:
            transitions[note] = transitions.get(note, 0) + 1
            changed_rows.append((job.stable_id, updated))

    if apply and changed_rows:
        store_name = type(store).__name__
        if store_name == "PostgresStore":
            from psycopg.types.json import Jsonb
            with store.connect() as connection:
                for stable_id, updated in changed_rows:
                    payload = updated.model_dump(mode="json", exclude_none=True)
                    digest = hashlib.sha256(
                        updated.model_dump_json(exclude_none=True).encode("utf-8")
                    ).hexdigest()
                    connection.execute(
                        "UPDATE jobs SET payload=%s, content_hash=%s WHERE stable_id=%s",
                        (Jsonb(payload), digest, stable_id),
                    )
        elif store_name == "Store":
            with store.connect() as connection:
                for stable_id, updated in changed_rows:
                    payload = updated.model_dump_json(exclude_none=True)
                    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                    connection.execute(
                        "UPDATE jobs SET payload=?, content_hash=? WHERE stable_id=?",
                        (payload, digest, stable_id),
                    )
        else:
            raise TypeError(f"Unsupported store for in-place re-enrichment: {store_name}")

    return {
        "mode": "applied" if apply else "dry_run",
        "active_jobs_scanned": len(jobs),
        "jobs_changed": len(changed_rows),
        "before_remote_policy": before_policies,
        "after_remote_policy": after_policies,
        "transitions": transitions,
        "ashby_retained_for_next_sync": ashby_ambiguous,
    }
