from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from boardforge.database import create_store
from boardforge.discovery import run_pending_discoveries
from boardforge.observability import observe_embedding_backlog
from boardforge.semantic import EmbeddingClient, warm_job_embeddings
from boardforge.workers import sync_due_sources


async def run_worker_cycle(store, embeddings: EmbeddingClient) -> dict:
    cycle_started = time.perf_counter()
    discovery_limit = max(
        1, min(10, int(os.getenv("BOARDFORGE_DISCOVERY_REQUESTS_PER_CYCLE", "5")))
    )
    disc_started = time.perf_counter()
    discovery = await run_pending_discoveries(store, limit=discovery_limit)
    disc_ms = (time.perf_counter() - disc_started) * 1000
    sync_interval = max(60, int(os.getenv("BOARDFORGE_SOURCE_SYNC_SECONDS", "3600")))
    sync_limit = max(
        1, min(25, int(os.getenv("BOARDFORGE_SOURCE_SYNC_LIMIT", "10")))
    )
    sync_concurrency = max(
        1, min(8, int(os.getenv("BOARDFORGE_SOURCE_SYNC_CONCURRENCY", "4")))
    )
    sync_started = time.perf_counter()
    sources = await sync_due_sources(
        store,
        interval_seconds=sync_interval,
        limit=sync_limit,
        concurrency=sync_concurrency,
    )
    sync_ms = (time.perf_counter() - sync_started) * 1000
    embedding_result: dict = {"status": "disabled"}
    emb_started = time.perf_counter()
    if embeddings.enabled:
        status = store.embedding_status(embeddings.model)
        observe_embedding_backlog(status["backlog"])
        try:
            result = await warm_job_embeddings(
                store,
                embeddings,
                batch_size=max(
                    1, min(64, int(os.getenv("BOARDFORGE_EMBEDDING_BATCH_SIZE", "32")))
                ),
                max_jobs=max(
                    1,
                    min(
                        250,
                        int(os.getenv("BOARDFORGE_EMBEDDING_JOBS_PER_CYCLE", "64")),
                    ),
                ),
            )
            embedding_result = {"status": "ok", **result}
        except Exception as exc:
            # Discovery/source refresh must remain available during a provider
            # outage. The next worker cycle retries only the still-missing jobs.
            embedding_result = {
                "status": "failed",
                "error": str(exc),
                "model": embeddings.model,
            }
    emb_ms = (time.perf_counter() - emb_started) * 1000
    return {
        "event": "worker_cycle",
        "discovery": discovery,
        "sources": sources,
        "embeddings": embedding_result,
        "latency_ms": {
            "discovery": round(disc_ms, 2),
            "source_sync": round(sync_ms, 2),
            "embedding_backfill": round(emb_ms, 2),
            "total": round((time.perf_counter() - cycle_started) * 1000, 2),
        },
    }


async def run_worker(*, once: bool = False) -> None:
    store = create_store()
    embeddings = EmbeddingClient(budget_store=store)
    poll_seconds = max(5, int(os.getenv("BOARDFORGE_WORKER_POLL_SECONDS", "30")))
    while True:
        print(json.dumps(await run_worker_cycle(store, embeddings)))
        if once:
            return
        await asyncio.sleep(poll_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one maintenance cycle and exit (for Cloud Run Jobs).",
    )
    args = parser.parse_args()
    asyncio.run(run_worker(once=args.once))
