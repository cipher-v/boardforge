from __future__ import annotations

import re
import httpx
from boardforge.connectors.base import JobConnector, ConnectorError
from boardforge.models import Job, SourceRef


def _text(item: dict) -> str:
    parts = [
        item.get("descriptionPlain", ""),
        item.get("additionalPlain", ""),
        " ".join(x.get("text", "") for x in item.get("lists", [])),
    ]
    return " ".join(parts).strip()


class LeverConnector(JobConnector):
    kind = "lever"

    async def fetch(self, source: SourceRef) -> list[Job]:
        endpoint = f"https://api.lever.co/v0/postings/{source.key}?mode=json"
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            r = await client.get(endpoint)
        if r.status_code != 200:
            raise ConnectorError(f"Lever fetch failed: {r.status_code}")
        jobs = []
        for item in r.json():
            cats = item.get("categories") or {}
            description = _text(item)
            lowered = f"{item.get('text','')} {description}".lower()
            tags = [t for t in ["python","sql","llm","rag","backend","machine learning","computer vision"] if t in lowered]
            jobs.append(Job(
                source_kind=self.kind,
                source_key=source.key,
                external_id=item.get("id", ""),
                title=item.get("text", ""),
                company=source.key,
                location=cats.get("location", ""),
                description=description,
                apply_url=item.get("hostedUrl", ""),
                employment_type=cats.get("commitment", ""),
                tags=tags,
            ))
        return jobs
