import asyncio

from boardforge.models import Job
from boardforge.semantic import EmbeddingClient
from boardforge.store import Store
from boardforge.worker import run_worker_cycle


class _FakeProvider:
    enabled = True
    model = "fake:worker:2"
    dimensions = 2

    async def embed(self, texts, *, task_type="RETRIEVAL_DOCUMENT"):
        return [[float(len(text)), 1.0] for text in texts]


def test_worker_backfills_only_missing_embeddings(tmp_path, monkeypatch):
    monkeypatch.setenv("BOARDFORGE_SEARCH_PROVIDER", "none")
    monkeypatch.setenv("BOARDFORGE_EMBEDDING_JOBS_PER_CYCLE", "10")
    store = Store(str(tmp_path / "worker.db"))
    store.upsert_source_jobs(
        "test",
        "worker",
        [
            Job(
                source_kind="test",
                source_key="worker",
                external_id="1",
                title="Backend Engineer",
                company="Acme",
                description="Python FastAPI",
            )
        ],
    )
    client = EmbeddingClient(_FakeProvider())

    first = asyncio.run(run_worker_cycle(store, client))
    second = asyncio.run(run_worker_cycle(store, client))

    assert first["embeddings"]["status"] == "ok"
    assert first["embeddings"]["processed"] == 1
    assert second["embeddings"]["processed"] == 0
    assert store.embedding_status(client.model)["coverage"] == 1.0
