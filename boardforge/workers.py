from __future__ import annotations

import asyncio

from boardforge.database import StoreProtocol
from boardforge.sync import sync_url


async def sync_due_sources(
    store: StoreProtocol,
    *,
    interval_seconds: int = 3600,
    limit: int = 25,
    concurrency: int = 4,
) -> dict:
    sources = store.sources_due(interval_seconds=interval_seconds, limit=limit)
    semaphore = asyncio.Semaphore(max(1, min(concurrency, 16)))

    async def sync_one(source: dict) -> dict:
        async with semaphore:
            try:
                summary = await sync_url(source["board_url"], store)
                store.record_discovered_source(
                    source["source_kind"], source["source_key"], source["board_url"],
                    "scheduled-refresh", synced=True, status="synced",
                )
                return {"source": source["board_url"], "status": "synced", "fetched": summary.fetched}
            except Exception as exc:
                store.record_discovered_source(
                    source["source_kind"], source["source_key"], source["board_url"],
                    "scheduled-refresh", status="failed",
                )
                return {"source": source["board_url"], "status": "failed", "error": str(exc)}

    results = await asyncio.gather(*(sync_one(source) for source in sources))
    return {
        "due": len(sources),
        "synced": sum(item["status"] == "synced" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "jobs_fetched": sum(item.get("fetched", 0) for item in results),
        "results": results,
    }
