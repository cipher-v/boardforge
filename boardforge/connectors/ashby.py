from __future__ import annotations

import httpx
from boardforge.connectors.base import JobConnector, ConnectorError
from boardforge.models import Job, SourceRef


class AshbyConnector(JobConnector):
    kind = "ashby"

    async def fetch(self, source: SourceRef) -> list[Job]:
        endpoint = f"https://api.ashbyhq.com/posting-api/job-board/{source.key}"
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            r = await client.get(endpoint)
        if r.status_code != 200:
            raise ConnectorError(f"Ashby fetch failed: {r.status_code}")
        payload = r.json()
        jobs = []
        for item in payload.get("jobs", []):
            desc = item.get("descriptionPlain", "") or ""
            lowered = f"{item.get('title','')} {desc}".lower()
            remote = "remote" if item.get("isRemote") else "unknown"
            jobs.append(Job(
                source_kind=self.kind,
                source_key=source.key,
                external_id=item.get("id", item.get("jobUrl", "")),
                title=item.get("title", ""),
                company=source.key,
                location=item.get("location", ""),
                remote_policy=remote,
                description=desc,
                apply_url=item.get("applyUrl") or item.get("jobUrl", ""),
                employment_type=item.get("employmentType", ""),
                tags=[t for t in ["python","sql","llm","rag","backend","machine learning","computer vision"] if t in lowered],
                published_at=item.get("publishedAt") if False else None,
            ))
        return jobs
