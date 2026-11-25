import asyncio
from pathlib import Path
import time

from boardforge.models import Job, Preferences
from boardforge.semantic import hybrid_rank_jobs, retrieve_and_rank_jobs, semantic_query_text
from boardforge.store import Store


class FakeEmbeddings:
    enabled = True
    model = "fake-v1"
    dimensions = 2

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    async def embed(self, texts: list[str], *, task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        self.calls.append((task_type, list(texts)))
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


def test_store_backed_search_does_not_embed_missing_documents(tmp_path: Path):
    store = Store(str(tmp_path / "cache.db"))
    client = FakeEmbeddings()
    jobs = [make_job("1", "Python APIs"), make_job("2", "Distributed systems")]

    run(jobs, client, store)

    assert [task for task, _ in client.calls] == ["RETRIEVAL_QUERY"]
    assert [len(texts) for _, texts in client.calls] == [1]


def test_query_embedding_is_persisted_and_reused_for_normalized_query(tmp_path: Path):
    store = Store(str(tmp_path / "query-cache.db"))
    store.upsert_source_jobs("benchmark", "cache", [
        make_job("1", "Build Python APIs and distributed systems."),
    ])
    client = FakeEmbeddings()

    async def search(query):
        return await retrieve_and_rank_jobs(
            store, Preferences(), query=query, top_k=10, rejected_ids=set(),
            client=client, candidate_limit=100,
        )

    _, cold = asyncio.run(search("  Python   systems "))
    _, warm = asyncio.run(search("python systems"))

    assert len(client.calls) == 1
    assert cold["query_embedding_source"] == "provider"
    assert cold["query_embedding_cache_hit"] is False
    assert warm["query_embedding_source"] == "persistent_cache"
    assert warm["query_embedding_cache_hit"] is True


def test_query_embedding_is_not_started_past_deadline(tmp_path: Path):
    store = Store(str(tmp_path / "deadline-cache.db"))
    store.upsert_source_jobs("benchmark", "cache", [
        make_job("1", "Build Python APIs."),
    ])
    client = FakeEmbeddings()
    matches, metadata = asyncio.run(retrieve_and_rank_jobs(
        store, Preferences(), query="Python", top_k=10, rejected_ids=set(),
        client=client, candidate_limit=100, deadline_monotonic=time.perf_counter(),
    ))
    assert matches
    assert client.calls == []
    assert metadata["query_embedding_source"] == "deadline_skip"
    assert metadata["embedding_error"] == "insufficient_remaining_time_for_query_embedding"


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
    assert {match.job.external_id for match in matches} == {"known", "unknown"}
    assert all(match.status == "needs_verification" for match in matches)


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
        store=None,
    ))

    by_id = {match.job.external_id: match for match in matches}
    assert by_id["backend"].status == "recommended"
    assert by_id["sales"].status == "needs_verification"
    assert "query-specific relevance not verified" in by_id["sales"].risks



def test_generic_country_overlap_does_not_create_confident_relevance():
    client = FakeEmbeddings()
    jobs = [
        Job(
            source_kind="benchmark",
            source_key="cache",
            external_id="irrelevant",
            title="Sales Executive",
            company="Demo",
            country="India",
            description="Enterprise account management in India",
        ),
    ]

    matches = asyncio.run(hybrid_rank_jobs(
        jobs,
        Preferences(countries=["India"]),
        query="machine learning engineer India",
        top_k=10,
        rejected_ids=set(),
        client=client,
        store=None,
    ))

    assert matches[0].status == "needs_verification"
    assert (
        "query-specific relevance not verified"
        in matches[0].risks
    )



def test_specific_intent_requires_more_than_role_evidence():
    client = FakeEmbeddings()
    generic_ml = Job(
        source_kind="benchmark",
        source_key="cache",
        external_id="generic-ml",
        title="Machine Learning Engineer",
        company="Demo",
        country="India",
        description="Build recommendation models and Python ML services.",
    )
    anti_fraud = Job(
        source_kind="benchmark",
        source_key="cache",
        external_id="anti-fraud",
        title="Software Engineer",
        company="Demo",
        country="India",
        description=(
            "Design and deploy machine learning systems for anti-fraud, "
            "anti-abuse, and distributed services."
        ),
    )

    matches = asyncio.run(hybrid_rank_jobs(
        [generic_ml, anti_fraud],
        Preferences(roles=["AI/ML"], countries=["India"]),
        query="machine learning anti-fraud anti-abuse distributed systems India",
        top_k=10,
        rejected_ids=set(),
        client=client,
        store=None,
    ))

    by_id = {match.job.external_id: match for match in matches}
    assert by_id["anti-fraud"].status == "recommended"
    assert by_id["generic-ml"].status == "needs_verification"
    assert "query-specific relevance not verified" in by_id["generic-ml"].risks


def test_general_role_query_still_accepts_role_evidence():
    client = FakeEmbeddings()
    generic_ml = Job(
        source_kind="benchmark",
        source_key="cache",
        external_id="generic-ml",
        title="Machine Learning Engineer",
        company="Demo",
        country="India",
        description="Build recommendation models and Python ML services.",
    )

    matches = asyncio.run(hybrid_rank_jobs(
        [generic_ml],
        Preferences(roles=["AI/ML"], countries=["India"]),
        query="machine learning engineer India",
        top_k=10,
        rejected_ids=set(),
        client=client,
        store=None,
    ))

    assert matches[0].status == "recommended"
