from __future__ import annotations

import html
import re
import httpx
from boardforge.connectors.base import JobConnector, ConnectorError
from boardforge.models import Job, SourceRef


def _strip_html(raw: str) -> str:
    decoded = raw or ""
    for _ in range(3):
        newer = html.unescape(decoded)
        if newer == decoded:
            break
        decoded = newer
    cleaned = re.sub(r"<\s*li\b[^>]*>", "\n- ", decoded, flags=re.I)
    cleaned = re.sub(r"<\s*(?:br|/p|/div|/ul|/ol|/h[1-6])\b[^>]*>", "\n", cleaned, flags=re.I)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    lines = [" ".join(line.split()) for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _text(item: dict) -> str:
    intro = (item.get("descriptionPlain") or "").strip() or _strip_html(item.get("description", ""))
    closing = (item.get("additionalPlain") or "").strip() or _strip_html(item.get("additional", ""))
    sections: list[str] = []
    for section in item.get("lists") or []:
        heading = (section.get("text") or "").strip()
        body = _strip_html(section.get("content", ""))
        if heading and body:
            sections.append(f"{heading}\n{body}")
        elif heading:
            sections.append(heading)
        elif body:
            sections.append(body)
    parts = [part for part in (intro, "\n\n".join(sections), closing) if part]
    return "\n\n".join(parts).strip()


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
