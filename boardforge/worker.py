from __future__ import annotations

import argparse
import asyncio
import json
import os

from boardforge.database import create_store
from boardforge.discovery import run_pending_discoveries
from boardforge.observability import observe_embedding_backlog
from boardforge.semantic import EmbeddingClient, warm_job_embeddings
from boardforge.workers import sync_due_sources


async def run_worker_cycle(store, embeddings: EmbeddingClient) -> dict:
    discovery = await run_pending_discoveries(store, limit=25)
    sync_interval = max(60, int(os.getenv("BOARDFORGE_SOURCE_SYNC_SECONDS", "3600")))
    sources = await sync_due_sources(
        store, interval_seconds=sync_interval, limit=25, concurrency=4,
    )
    embedding_result: dict = {"status": "disabled"}
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
                        1000,
                        int(os.getenv("BOARDFORGE_EMBEDDING_JOBS_PER_CYCLE", "100")),
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
    return {
        "event": "worker_cycle",
        "discovery": discovery,
        "sources": sources,
        "embeddings": embedding_result,
    }


async def run_worker(*, once: bool = False) -> None:
    store = create_store()
    embeddings = EmbeddingClient()
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
