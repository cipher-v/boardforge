from __future__ import annotations

import html
import re
import httpx
from boardforge.connectors.base import JobConnector, ConnectorError
from boardforge.models import Job, SourceRef


_TAGS = ["python", "sql", "llm", "rag", "computer vision", "backend", "machine learning", "docker", "kubernetes"]


def _strip_html(text: str) -> str:
    decoded = text or ""
    # Greenhouse content can contain nested/double-escaped HTML entities.
    for _ in range(3):
        newer = html.unescape(decoded)
        if newer == decoded:
            break
        decoded = newer
    cleaned = re.sub(r"<[^>]+>", " ", decoded)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


class GreenhouseConnector(JobConnector):
    kind = "greenhouse"

    async def fetch(self, source: SourceRef) -> list[Job]:
        endpoint = f"https://boards-api.greenhouse.io/v1/boards/{source.key}/jobs?content=true"
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            r = await client.get(endpoint)
        if r.status_code != 200:
            raise ConnectorError(f"Greenhouse fetch failed: {r.status_code}")
        payload = r.json()
        jobs = []
        for item in payload.get("jobs", []):
            description = _strip_html(item.get("content", ""))
            title = item.get("title", "")
            text = f"{title} {description}".lower()
            jobs.append(Job(
                source_kind=self.kind,
                source_key=source.key,
                external_id=str(item["id"]),
                title=title,
                company=source.key,
                location=(item.get("location") or {}).get("name", ""),
                description=description,
                apply_url=item.get("absolute_url", ""),
                tags=[t for t in _TAGS if t in text],
                updated_at=item.get("updated_at"),
            ))
        return jobs
