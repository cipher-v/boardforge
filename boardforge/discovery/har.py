from __future__ import annotations

import json
from typing import Any, Iterable

from boardforge.discovery.manifest import ConnectorManifest, FieldMap


TITLE_KEYS = ["title","name","jobTitle","positionTitle"]
ID_KEYS = ["id","jobId","job_id","postingId","requisitionId","reqId"]
LOCATION_KEYS = ["location","locationName","city","jobLocation"]
DESCRIPTION_KEYS = ["description","descriptionPlain","content","jobDescription"]
URL_KEYS = ["url","applyUrl","absolute_url","hostedUrl","jobUrl"]
TYPE_KEYS = ["employmentType","commitment","type"]


def _first_key(sample: dict, candidates: list[str]) -> str | None:
    for key in candidates:
        if key in sample:
            return key
    lower = {str(k).lower(): str(k) for k in sample}
    for key in candidates:
        if key.lower() in lower:
            return lower[key.lower()]
    return None


def _candidate_lists(value: Any, path: list[str | int] | None = None):
    path = path or []
    if isinstance(value, list):
        dicts = [x for x in value if isinstance(x, dict)]
        if len(dicts) >= 1:
            yield path, dicts
        for i, child in enumerate(value[:5]):
            yield from _candidate_lists(child, path + [i])
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _candidate_lists(child, path + [key])


def _score(sample: dict) -> tuple[float, FieldMap | None, list[str]]:
    title = _first_key(sample, TITLE_KEYS)
    ext_id = _first_key(sample, ID_KEYS)
    location = _first_key(sample, LOCATION_KEYS)
    description = _first_key(sample, DESCRIPTION_KEYS)
    url = _first_key(sample, URL_KEYS)
    emp = _first_key(sample, TYPE_KEYS)

    score = 0.0
    evidence = []
    if title:
        score += 0.35
        evidence.append(f"title field: {title}")
    if ext_id:
        score += 0.20
        evidence.append(f"id field: {ext_id}")
    if location:
        score += 0.10
        evidence.append(f"location field: {location}")
    if description:
        score += 0.15
        evidence.append(f"description field: {description}")
    if url:
        score += 0.15
        evidence.append(f"url field: {url}")
    if emp:
        score += 0.05
        evidence.append(f"employment field: {emp}")

    if not title or not (ext_id or url):
        return score, None, evidence

    return score, FieldMap(
        id=ext_id or url,
        title=title,
        location=location,
        description=description,
        apply_url=url,
        employment_type=emp,
    ), evidence


def discover_from_har(har: dict, min_confidence: float = 0.55) -> list[ConnectorManifest]:
    out: list[ConnectorManifest] = []
    entries = ((har.get("log") or {}).get("entries") or [])

    for entry in entries:
        request = entry.get("request") or {}
        response = entry.get("response") or {}
        content = response.get("content") or {}
        mime = str(content.get("mimeType") or "").lower()
        text = content.get("text")

        if not text or ("json" not in mime and not str(text).lstrip().startswith(("{","["))):
            continue

        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            continue

        for path, rows in _candidate_lists(payload):
            sample = rows[0]
            score, fields, evidence = _score(sample)
            if fields is None or score < min_confidence:
                continue
            endpoint = request.get("url") or ""
            method = (request.get("method") or "GET").upper()
            out.append(ConnectorManifest(
                name=f"learned-{len(out)+1}",
                endpoint=endpoint,
                method=method,
                response_path=path,
                fields=fields,
                confidence=round(score, 3),
                evidence=evidence + [f"observed {len(rows)} row(s) in captured response"],
            ))

    # Highest-confidence candidate per endpoint/path first.
    out.sort(key=lambda m: (-m.confidence, m.endpoint, str(m.response_path)))
    unique = []
    seen = set()
    for item in out:
        key = (item.endpoint, tuple(item.response_path))
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique
