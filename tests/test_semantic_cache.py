import asyncio
from pathlib import Path

from boardforge.models import Job, Preferences
from boardforge.semantic import hybrid_rank_jobs, semantic_query_text
from boardforge.store import Store


class FakeEmbeddings:
    enabled = True
    model = "fake-v1"

    def __init__(self):
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str], *, task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
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


def test_hybrid_keeps_verified_matches_ahead_of_needs_verification(tmp_path: Path):
    store = Store(str(tmp_path / "verify.db"))
    client = FakeEmbeddings()
    jobs = [
        Job(
            source_kind="benchmark",
            source_key="verify",
            external_id="known",
            title="AI Engineer",
            company="Demo",
            country="India",
            salary_currency="INR",
            salary_min=1400000,
            salary_period="year",
        ),
        Job(
            source_kind="benchmark",
            source_key="verify",
            external_id="unknown",
            title="AI Engineer",
            company="Demo",
            country="India",
            description="Python AI retrieval",
        ),
    ]
    matches = asyncio.run(hybrid_rank_jobs(
        jobs,
        Preferences(roles=["AI/ML"], countries=["India"], min_salary_lpa=12),
        query="Python AI retrieval",
        top_k=10,
        rejected_ids=set(),
        client=client,
        store=store,
    ))
    assert matches[0].job.external_id == "known"
    assert matches[1].status == "needs_verification"


def test_semantic_query_text_removes_hard_filter_tokens():
    prefs = Preferences(roles=["AI/ML"], countries=["India"], locations=["Bengaluru"])
    assert semantic_query_text(
        prefs, "machine learning engineer in Bengaluru India remote 12 LPA"
    ) == "AI/ML machine learning engineer in"


def test_semantic_only_match_is_not_marked_recommended(tmp_path: Path):
    store = Store(str(tmp_path / "semantic-only.db"))
    client = FakeEmbeddings()
    jobs = [
        make_job("backend", "Python FastAPI PostgreSQL"),
        Job(
            source_kind="benchmark",
            source_key="cache",
            external_id="sales",
            title="Sales Executive",
            company="Demo",
            description="Enterprise account management",
        ),
    ]

    matches = asyncio.run(hybrid_rank_jobs(
        jobs,
        Preferences(),
        query="Python",
        top_k=10,
        rejected_ids=set(),
        client=client,
        store=store,
    ))

    by_id = {match.job.external_id: match for match in matches}
    assert by_id["backend"].status == "recommended"
    assert by_id["sales"].status == "needs_verification"
    assert "semantic-only relevance; no lexical or role evidence" in by_id["sales"].risks
