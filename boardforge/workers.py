from __future__ import annotations

import asyncio
import time

from boardforge.database import StoreProtocol
from boardforge.quarantine import is_quarantined_source
from boardforge.sync import sync_url


async def sync_due_sources(
    store: StoreProtocol,
    *,
    interval_seconds: int = 3600,
    limit: int = 25,
    concurrency: int = 4,
) -> dict:
    started = time.perf_counter()
    sources = [
        s for s in store.sources_due(interval_seconds=interval_seconds, limit=limit)
        if not is_quarantined_source(s.get("source_kind", ""), s.get("source_key", ""))
    ]
    semaphore = asyncio.Semaphore(max(1, min(concurrency, 16)))

    async def sync_one(source: dict) -> dict:
        one_started = time.perf_counter()
        async with semaphore:
            try:
                summary = await sync_url(source["board_url"], store)
                store.record_discovered_source(
                    source["source_kind"], source["source_key"], source["board_url"],
                    "scheduled-refresh", synced=True, status="synced",
                )
                return {
                    "source": source["board_url"],
                    "status": "synced",
                    "fetched": summary.fetched,
                    "inserted": summary.inserted,
                    "duration_ms": round((time.perf_counter() - one_started) * 1000, 2),
                }
            except Exception as exc:
                store.record_discovered_source(
                    source["source_kind"], source["source_key"], source["board_url"],
                    "scheduled-refresh", status="failed",
                )
                return {
                    "source": source["board_url"],
                    "status": "failed",
                    "error": str(exc),
                    "duration_ms": round((time.perf_counter() - one_started) * 1000, 2),
                }

    results = await asyncio.gather(*(sync_one(source) for source in sources))
    return {
        "due": len(sources),
        "synced": sum(item["status"] == "synced" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "jobs_fetched": sum(item.get("fetched", 0) for item in results),
        "jobs_inserted": sum(item.get("inserted", 0) for item in results),
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "results": results,
    }
