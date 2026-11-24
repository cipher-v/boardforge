import asyncio
from pathlib import Path

from boardforge.models import Job, Preferences
from boardforge.semantic import hybrid_rank_jobs
from boardforge.store import Store


class FakeEmbeddings:
    enabled = True
    model = "fake-v1"

    def __init__(self):
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text) % 7 + 1), 1.0] for text in texts]


def make_job(external_id: str, description: str) -> Job:
    return Job(
        source_kind="benchmark",
        source_key="cache",
        external_id=external_id,
        title="Software Engineer",
        company="Demo",
        description=description,
    )


def run(jobs: list[Job], client: FakeEmbeddings, store: Store):
    return asyncio.run(hybrid_rank_jobs(
        jobs,
        Preferences(),
        query="Python systems",
        top_k=10,
        rejected_ids=set(),
        client=client,
        store=store,
    ))


def test_job_embeddings_are_reused_until_content_changes(tmp_path: Path):
    store = Store(str(tmp_path / "cache.db"))
    client = FakeEmbeddings()
    jobs = [make_job("1", "Python APIs"), make_job("2", "Distributed systems")]

    run(jobs, client, store)
    assert [len(call) for call in client.calls] == [1, 2]

    run(jobs, client, store)
    assert [len(call) for call in client.calls] == [1, 2, 1]

    changed = [make_job("1", "Python APIs and retrieval"), jobs[1]]
    run(changed, client, store)
    assert [len(call) for call in client.calls] == [1, 2, 1, 1, 1]
