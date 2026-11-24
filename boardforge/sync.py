from __future__ import annotations

import asyncio

from boardforge.connectors.detect import detect_source
from boardforge.connectors.registry import CONNECTORS
from boardforge.enrich import enrich_job
from boardforge.models import SyncSummary
from boardforge.security import validate_public_url
from boardforge.database import StoreProtocol


async def sync_url(url: str, store: StoreProtocol) -> SyncSummary:
    await asyncio.to_thread(validate_public_url, url)
    source = detect_source(url)
    connector = CONNECTORS[source.kind]
    raw_jobs = await connector.fetch(source)
    jobs = [enrich_job(job) for job in raw_jobs]
    return store.upsert_source_jobs(source.kind, source.key, jobs)
