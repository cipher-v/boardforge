from __future__ import annotations

import re
from urllib.parse import urlparse

from boardforge.models import SourceRef


def detect_source(url: str) -> SourceRef:
    u = urlparse(url)
    host = u.netloc.lower()
    parts = [p for p in u.path.strip("/").split("/") if p]

    if "greenhouse.io" in host:
        # job-boards.greenhouse.io/company[/jobs/id] and boards.greenhouse.io/company/jobs/id
        key = parts[0] if parts else ""
        board_url = f"https://{host}/{key}" if key else url
        return SourceRef(kind="greenhouse", key=key, board_url=board_url)

    if "lever.co" in host:
        # jobs.lever.co/company/job-id
        key = parts[0] if parts else ""
        board_url = f"https://{host}/{key}" if key else url
        return SourceRef(kind="lever", key=key, board_url=board_url)

    if "ashbyhq.com" in host:
        # jobs.ashbyhq.com/company/job-id
        key = parts[0] if parts else ""
        board_url = f"https://{host}/{key}" if key else url
        return SourceRef(kind="ashby", key=key, board_url=board_url)

    safe_key = re.sub(r"[^a-z0-9]+", "-", host).strip("-")
    return SourceRef(kind="generic", key=safe_key, board_url=url)
