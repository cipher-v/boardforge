from __future__ import annotations

import re
from urllib.parse import urlparse
from boardforge.models import SourceRef


def detect_source(url: str) -> SourceRef:
    u = urlparse(url)
    host = u.netloc.lower()
    path = u.path.strip("/")

    if "greenhouse.io" in host:
        parts = [p for p in path.split("/") if p]
        token = parts[-1] if parts else ""
        return SourceRef(kind="greenhouse", key=token, board_url=url)

    if "lever.co" in host:
        parts = [p for p in path.split("/") if p]
        key = parts[0] if parts else ""
        return SourceRef(kind="lever", key=key, board_url=url)

    if "ashbyhq.com" in host:
        parts = [p for p in path.split("/") if p]
        key = parts[-1] if parts else ""
        return SourceRef(kind="ashby", key=key, board_url=url)

    # Common career URLs that embed a hosted ATS are handled by HTML discovery later.
    safe_key = re.sub(r"[^a-z0-9]+", "-", host).strip("-")
    return SourceRef(kind="generic", key=safe_key, board_url=url)
