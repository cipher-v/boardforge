from __future__ import annotations

from boardforge.connectors.detect import detect_source
from boardforge.connectors.registry import CONNECTORS
from boardforge.models import SyncSummary
from boardforge.store import Store


async def sync_url(url: str, store: Store) -> SyncSummary:
    source = detect_source(url)
    connector = CONNECTORS[source.kind]
    jobs = await connector.fetch(source)
    return store.upsert_source_jobs(source.kind, source.key, jobs)
