import asyncio
import json
from pathlib import Path

import pytest

from boardforge.holdout import (
    DEV_CORPUS_COMPANIES,
    canonical_sha256,
    classify_job,
    load_sources,
)
from boardforge.holdout_eval import (
    CachedEmbeddingProvider,
    _aggregate_rows,
    assert_isolated_database_url,
    plan_vertex_usage,
)
from boardforge.models import Job


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "boardforge" / "data"


def test_frozen_holdout_is_independent_complete_and_checksummed():
    sources = load_sources(DATA / "holdout_sources_v1.json")
    assert 8 <= len(sources) <= 12
    assert not ({source.company.casefold() for source in sources} & DEV_CORPUS_COMPANIES)

    corpus = json.loads((DATA / "holdout_corpus_v1.json").read_text(encoding="utf-8"))
    assert corpus["status"] == "frozen"
    assert 80 <= len(corpus["records"]) <= 120
    assert len({record["job"]["company"] for record in corpus["records"]}) == 10
    assert canonical_sha256(corpus["records"]) == corpus["records_sha256"]
    assert all(record["source_url"] and record["collected_at"] for record in corpus["records"])
    assert len({record["stable_id"] for record in corpus["records"]}) == len(corpus["records"])


def test_frozen_labels_cover_every_job_and_are_explicitly_provisional():
    corpus = json.loads((DATA / "holdout_corpus_v1.json").read_text(encoding="utf-8"))
    labels = json.loads(
        (DATA / "holdout_labels_v1_provisional.json").read_text(encoding="utf-8")
    )
    stable_ids = {record["stable_id"] for record in corpus["records"]}
    assert labels["verification_status"] == "automated_provisional_not_human_verified"
    assert labels["status"] == "frozen_provisional"
    assert 30 <= len(labels["cases"]) <= 40
    assert canonical_sha256(labels["cases"]) == labels["labels_sha256"]
    assert labels["corpus_records_sha256"] == corpus["records_sha256"]
    for case in labels["cases"]:
        assert set(case["labels"]) == stable_ids
        assert {row["label"] for row in case["labels"].values()} <= {
            "direct", "adjacent", "irrelevant", "forbidden"
        }


def test_role_classification_does_not_promote_incidental_boilerplate():
    job = Job(
        source_kind="test", source_key="x", external_id="1",
        title="Commercial Account Executive", company="Example",
        description="Our engineering organization builds AI and backend data pipelines.",
    )
    assert classify_job(job)[0] == "nontechnical"


def test_eval_database_guard_refuses_application_database():
    with pytest.raises(ValueError, match="non-isolated"):
        assert_isolated_database_url("postgresql://u:p@localhost/boardforge")
    assert assert_isolated_database_url(
        "postgresql://u:p@localhost/boardforge_holdout_eval"
    ) == "boardforge_holdout_eval"


def test_usage_plan_is_read_only_and_counts_uncached_texts(tmp_path: Path):
    cache = tmp_path / "missing-cache.json"
    plan = plan_vertex_usage(
        DATA / "holdout_corpus_v1.json",
        DATA / "holdout_labels_v1_provisional.json",
        cache,
    )
    assert plan["status"] == "plan_only_no_api_calls_executed"
    assert plan["unique_documents"] == 100
    assert 30 <= plan["unique_queries"] <= 40
    assert plan["planned_total_vertex_prediction_requests"] == (
        plan["planned_document_embedding_requests"]
        + plan["planned_query_embedding_requests"]
    )
    assert not cache.exists()


class _FakeProvider:
    enabled = True
    model = "fake:768"
    dimensions = 768

    def __init__(self):
        self.calls = 0

    async def embed(self, texts, *, task_type):
        self.calls += 1
        return [[float(index)] * self.dimensions for index, _ in enumerate(texts)]


def test_embedding_cache_deduplicates_and_persists_across_variants(tmp_path: Path):
    provider = _FakeProvider()
    path = tmp_path / "cache.json"
    cached = CachedEmbeddingProvider(provider, path)
    first = asyncio.run(cached.embed(["same", "same", "other"], task_type="RETRIEVAL_QUERY"))
    second = asyncio.run(cached.embed(["same", "other"], task_type="RETRIEVAL_QUERY"))
    assert len(first) == 3 and len(second) == 2
    assert provider.calls == 1
    assert cached.provider_texts == 2
    assert cached.cache_hits == 2
    reloaded = CachedEmbeddingProvider(_FakeProvider(), path)
    asyncio.run(reloaded.embed(["same"], task_type="RETRIEVAL_QUERY"))
    assert reloaded.provider_calls == 0


def test_empty_ground_truth_is_excluded_and_false_recommendations_are_micro_counted():
    row = {
        "candidate_recall": None,
        "precision_at_10": 0.0,
        "recall_at_10": None,
        "ndcg_at_10": None,
        "graded_ndcg_at_10": None,
        "predicted": ["a", "b"],
        "recommended": ["a"],
        "false_recommendation_count": 1,
        "relevant_count": 0,
        "statuses": ["recommended", "needs_verification"],
        "forbidden_hits": [],
        "hard_constraint_hits": [],
    }
    summary = _aggregate_rows([row], [12.0])
    assert summary["recall_at_10"] is None
    assert summary["ndcg_at_10"] is None
    assert summary["recommended_precision"] == 0.0
    assert summary["recommended_irrelevant_rate"] == 1.0
    assert summary["recommendation_coverage"] == 0.5
    assert summary["abstention_rate"] == 0.5
