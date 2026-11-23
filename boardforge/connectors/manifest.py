from __future__ import annotations

from typing import Any
import httpx

from boardforge.discovery.manifest import ConnectorManifest
from boardforge.enrich import enrich_job
from boardforge.models import Job


def _dig(value: Any, path: list[str | int]) -> Any:
    cur = value
    for part in path:
        cur = cur[part]
    return cur


def _value(row: dict, key: str | None) -> str:
    if not key:
        return ""
    value = row.get(key, "")
    if isinstance(value, dict):
        # Common nested location shapes.
        for candidate in ["name","text","city","value"]:
            if candidate in value:
                return str(value[candidate])
        return str(value)
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value or "")


async def fetch_manifest(manifest: ConnectorManifest) -> list[Job]:
    if manifest.method != "GET":
        raise ValueError("Learned connector execution currently supports public GET endpoints only")

    async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers={"User-Agent":"BoardForge/0.2"}) as client:
        r = await client.get(manifest.endpoint)
    r.raise_for_status()
    rows = _dig(r.json(), manifest.response_path)
    if not isinstance(rows, list):
        raise ValueError("Manifest response path did not resolve to a list")

    jobs = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        f = manifest.fields
        ext = _value(row, f.id)
        title = _value(row, f.title)
        if not ext or not title:
            continue
        jobs.append(enrich_job(Job(
            source_kind="learned",
            source_key=manifest.name,
            external_id=ext,
            title=title,
            company=manifest.name,
            location=_value(row, f.location),
            description=_value(row, f.description),
            apply_url=_value(row, f.apply_url),
            employment_type=_value(row, f.employment_type),
        )))
    return jobs
