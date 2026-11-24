import asyncio

from boardforge.agent import JobAgent
from boardforge.confidence import assess_query_relevance
from boardforge.matching import rank_jobs
from boardforge.models import Job, Preferences
from boardforge.semantic import EmbeddingClient
from boardforge.store import Store


def _job(external_id: str, title: str, description: str) -> Job:
    return Job(
        source_kind="test",
        source_key="confidence",
        external_id=external_id,
        title=title,
        company="Example",
        country="India",
        description=description,
    )


def test_incidental_keywords_across_long_description_do_not_create_confidence():
    filler = " ".join(["general product delivery and stakeholder collaboration"] * 30)
    candidate = _job(
        "incidental",
        "Software Engineer",
        f"Security reviews are routine. {filler} Python is occasionally used. "
        f"{filler} Incident documentation is maintained. {filler} Automation is encouraged.",
    )
    query = "security incident response detection automation Python India"
    prefs = Preferences(countries=["India"])

    assessment = assess_query_relevance(candidate, prefs, query)
    match = rank_jobs([candidate], prefs, query=query)[0]

    assert assessment.matched_terms
    assert assessment.confident_topic is False
    assert match.status == "needs_verification"


def test_mismatched_domain_is_retained_but_not_recommended():
    security = _job(
        "security",
        "Software Engineer, Security Platform",
        "Build security platform backend services and access-control APIs.",
    )
    commerce = _job(
        "commerce",
        "Software Engineer, Commerce Platform",
        "Build commerce platform backend services and payment APIs.",
    )
    prefs = Preferences(roles=["Backend"], countries=["India"])
    matches = rank_jobs(
        [commerce, security],
        prefs,
        query="security platform backend engineer services India",
        top_k=10,
    )
    by_id = {match.job.external_id: match for match in matches}

    assert by_id["security"].status == "recommended"
    assert by_id["commerce"].status == "needs_verification"


def test_manager_and_individual_contributor_intent_are_distinguished():
    manager = _job(
        "manager",
        "Engineering Manager, Data Platform",
        "Lead teams building distributed data platform systems and mentor engineers.",
    )
    individual = _job(
        "individual",
        "Software Engineer, Data Platform",
        "Build distributed data platform systems as an individual contributor.",
    )
    query = "engineering manager data platform distributed systems India"
    matches = rank_jobs(
        [individual, manager], Preferences(countries=["India"]), query=query, top_k=10
    )
    by_id = {match.job.external_id: match for match in matches}

    assert by_id["manager"].status == "recommended"
    assert by_id["individual"].status == "needs_verification"


def test_generic_engineering_title_requires_coherent_specific_work():
    strong = _job(
        "strong",
        "Software Engineer",
        "Design machine learning systems for anti-fraud and anti-abuse detection services.",
    )
    weak = _job(
        "weak",
        "Software Engineer",
        "Build machine learning systems for recommendations. "
        + "General product delivery and stakeholder collaboration. " * 30
        + "Fraud appears in compliance documentation only.",
    )
    prefs = Preferences(roles=["AI/ML"], countries=["India"])
    query = "machine learning anti-fraud anti-abuse distributed systems India"
    matches = rank_jobs([weak, strong], prefs, query=query, top_k=10)
    by_id = {match.job.external_id: match for match in matches}

    assert by_id["strong"].status == "recommended"
    assert by_id["weak"].status == "needs_verification"


class _FailingProvider:
    enabled = True
    model = "failing:query"
    dimensions = 2

    def __init__(self):
        self.calls = 0

    async def embed(self, texts, *, task_type="RETRIEVAL_DOCUMENT"):
        self.calls += 1
        raise RuntimeError("provider unavailable")


def test_provider_failure_attempts_query_once_and_keeps_deterministic_policy(tmp_path):
    store = Store(str(tmp_path / "provider.db"))
    store.upsert_source_jobs(
        "test",
        "confidence",
        [
            _job(
                "relevant",
                "Software Engineer, Security Platform",
                "Build security platform backend services and access-control APIs.",
            ),
            _job(
                "uncertain",
                "Software Engineer, Commerce Platform",
                "Build commerce platform backend services and payment APIs.",
            ),
        ],
    )
    provider = _FailingProvider()
    agent = JobAgent(store, embeddings=EmbeddingClient(provider))

    result = asyncio.run(
        agent.search_jobs("candidate", "security platform backend engineer services India")
    )
    by_id = {
        row["job"]["external_id"]: row for row in result.data["matches"]
    }

    assert provider.calls == 1
    assert result.data["retrieval"]["semantic_retrieval"] is False
    assert by_id["relevant"]["status"] == "recommended"
    assert by_id["uncertain"]["status"] == "needs_verification"


class _DisabledProvider:
    enabled = False
    model = "disabled"
    dimensions = 2

    async def embed(self, texts, *, task_type="RETRIEVAL_DOCUMENT"):
        raise AssertionError("disabled embeddings must not be called")


def test_missing_embeddings_use_the_same_confidence_policy(tmp_path, monkeypatch):
    monkeypatch.delenv("BOARDFORGE_ENABLE_EMBEDDINGS", raising=False)
    store = Store(str(tmp_path / "missing.db"))
    store.upsert_source_jobs(
        "test",
        "confidence",
        [_job("candidate", "Software Engineer", "General product platform development.")],
    )
    agent = JobAgent(store, embeddings=EmbeddingClient(_DisabledProvider()))

    result = asyncio.run(
        agent.search_jobs("candidate", "security incident response automation India")
    )

    assert result.data["matches"][0]["status"] == "needs_verification"
