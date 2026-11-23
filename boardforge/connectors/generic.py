from __future__ import annotations

import json
import re
import httpx
from boardforge.connectors.base import JobConnector, ConnectorError
from boardforge.models import Job, SourceRef


SCRIPT_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.I | re.S,
)


def _walk(value):
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)


class GenericJsonLdConnector(JobConnector):
    kind = "generic"

    async def fetch(self, source: SourceRef) -> list[Job]:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False, headers={"User-Agent":"BoardForge/0.2"}) as client:
            r = await client.get(source.board_url)
        if r.is_redirect:
            raise ConnectorError("Generic connector redirects are not followed for SSRF safety")
        if r.status_code != 200:
            raise ConnectorError(f"Generic fetch failed: {r.status_code}")

        jobs: list[Job] = []
        idx = 0
        for block in SCRIPT_RE.findall(r.text):
            try:
                data = json.loads(block)
            except json.JSONDecodeError:
                continue
            for node in _walk(data):
                if node.get("@type") != "JobPosting":
                    continue
                idx += 1
                org = node.get("hiringOrganization") or {}
                loc = node.get("jobLocation") or {}
                if isinstance(loc, list):
                    loc = loc[0] if loc else {}
                addr = (loc.get("address") or {}) if isinstance(loc, dict) else {}
                location = ", ".join(str(addr.get(k,"")) for k in ["addressLocality","addressRegion","addressCountry"] if addr.get(k))
                desc = re.sub(r"<[^>]+>", " ", str(node.get("description",""))).strip()
                url = node.get("url") or source.board_url
                ext = str(node.get("identifier") or url or idx)
                jobs.append(Job(
                    source_kind=self.kind,
                    source_key=source.key,
                    external_id=ext,
                    title=node.get("title",""),
                    company=org.get("name", source.key) if isinstance(org, dict) else source.key,
                    location=location,
                    description=desc,
                    apply_url=url,
                    employment_type=str(node.get("employmentType","")),
                ))
        return jobs
