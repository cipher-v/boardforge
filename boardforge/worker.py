from __future__ import annotations

import asyncio
import json
import os

from boardforge.database import create_store
from boardforge.discovery import run_pending_discoveries
from boardforge.workers import sync_due_sources


async def run_worker() -> None:
    store = create_store()
    poll_seconds = max(5, int(os.getenv("BOARDFORGE_WORKER_POLL_SECONDS", "30")))
    sync_interval = max(60, int(os.getenv("BOARDFORGE_SOURCE_SYNC_SECONDS", "3600")))
    while True:
        discovery = await run_pending_discoveries(store, limit=25)
        sources = await sync_due_sources(
            store, interval_seconds=sync_interval, limit=25, concurrency=4,
        )
        print(json.dumps({"event": "worker_cycle", "discovery": discovery, "sources": sources}))
        await asyncio.sleep(poll_seconds)


if __name__ == "__main__":
    asyncio.run(run_worker())
