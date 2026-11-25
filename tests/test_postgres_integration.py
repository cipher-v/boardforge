import os
from urllib.parse import urlsplit
from datetime import datetime, timezone

import pytest

from boardforge.models import CandidateProfile, FeedbackRequest, Job, Preferences
from boardforge.postgres_store import PostgresStore
from boardforge.agent_live_eval import _usage_rows


POSTGRES_URL = os.getenv("BOARDFORGE_TEST_POSTGRES_URL")


def _assert_disposable_database(url: str) -> None:
    # Destructive integration tests must never run against application databases.
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.path != "/boardforge_test"
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("Integration tests require the disposable local boardforge_test database")
    application_url = os.getenv("DATABASE_URL", "")
    if application_url and urlsplit(application_url).path == parsed.path:
        raise RuntimeError("Integration database must differ from the application database")


@pytest.mark.parametrize("url", [
    "postgresql://localhost/boardforge",
    "postgresql://localhost/boardforge_holdout_eval",
    "postgresql://remote.example/boardforge_test",
    "postgresql://localhost/boardforge_test?dbname=boardforge",
])
def test_destructive_integration_database_guard(url):
    with pytest.raises(RuntimeError):
        _assert_disposable_database(url)



@pytest.mark.skipif(not POSTGRES_URL, reason="set BOARDFORGE_TEST_POSTGRES_URL for pgvector integration")
def test_postgres_store_migrations_fts_vectors_and_state():
    _assert_disposable_database(POSTGRES_URL)
    store = PostgresStore(POSTGRES_URL, min_size=1, max_size=2)
    try:
        with store.connect() as connection:
            actual_database = connection.execute("SELECT current_database() AS name").fetchone()
            # Check the server-reported database as well as the supplied URL.
            if (actual_database["name"] if isinstance(actual_database, dict) else actual_database[0]) != "boardforge_test":
                raise RuntimeError("Refusing to truncate a non-test database")
            connection.execute(
                """TRUNCATE jobs,job_events,sync_runs,job_embeddings,profiles,
                   candidate_profiles,feedback,discovery_requests,discovered_sources,
                   llm_usage_reservations,relevance_verifications,query_embeddings
                   RESTART IDENTITY CASCADE"""
            )

        backend = Job(
            source_kind="test", source_key="integration", external_id="backend",
            title="Backend Engineer", company="Acme", country="India",
            description="Python FastAPI PostgreSQL distributed systems",
        )
        vision = Job(
            source_kind="test", source_key="integration", external_id="vision",
            title="Computer Vision Engineer", company="Acme", country="India",
            description="PyTorch OpenCV object detection",
        )
        sales = Job(
            source_kind="test", source_key="integration", external_id="sales",
            title="Sales Executive", company="Acme", country="India",
            description="Enterprise account management",
        )
        senior = Job(
            source_kind="test", source_key="integration", external_id="senior",
            title="Senior Backend Engineer", company="Acme", country="India",
            description="Python distributed systems",
            min_experience_years=5,
        )
        summary = store.upsert_source_jobs(
            "test", "integration", [backend, vision, sales, senior]
        )
        assert summary.inserted == 4
        assert store.candidate_jobs("Python FastAPI", limit=2)[0].external_id == "backend"
        india_candidates = store.candidate_jobs(
            "engineer",
            limit=10,
            preferences=Preferences(countries=["India"]),
        )
        assert {job.external_id for job in india_candidates} == {
            "backend", "vision", "sales", "senior"
        }
        junior_candidates = store.candidate_jobs(
            "engineer",
            limit=10,
            preferences=Preferences(
                countries=["India"],
                max_required_experience_years=1,
            ),
        )
        assert {job.external_id for job in junior_candidates} == {
            "backend", "vision", "sales"
        }

        vector = [0.0] * 768
        vector[0] = 1.0
        store.save_embedding(backend.stable_id, "test-model", "hash", vector)
        assert store.load_embedding(backend.stable_id, "test-model", "hash") == vector
        store.save_query_embedding(
            "query-cache-key", "python backend", "test-model", "config-hash", vector,
        )
        assert store.load_query_embedding("query-cache-key") == vector
        store.save_relevance_verification(
            "relevance-cache-key",
            {
                "candidate_id": backend.stable_id,
                "relevance": "direct",
                "evidence_ids": ["E001"],
                "evidence_snippets": ["Python FastAPI PostgreSQL distributed systems"],
                "evidence_validated": True,
                "reasoning": "Backend responsibilities match.",
            },
        )
        assert store.load_relevance_verification("relevance-cache-key")["relevance"] == "direct"
        assert store.reserve_llm_usage(
            "vertex-gemini", "test-model", "integration-reservation", 100,
            tpm_limit=1000, tpd_limit=10000, reserved_cost_usd=0.01,
            cumulative_cost_limit_usd=0.10,
        )["allowed"] is True
        store.finalize_llm_usage(
            "integration-reservation", status="completed", total_tokens=90,
            estimated_cost_usd=0.009,
        )
        usage_summary = store.usage_cost_summary()
        assert usage_summary["reservations"] >= 1
        assert usage_summary["estimated_cost_usd"] >= 0.009
        rows = _usage_rows(store, datetime(2000, 1, 1, tzinfo=timezone.utc))
        assert any(row["reservation_id"] == "integration-reservation" for row in rows)
        assert store.candidate_jobs(
            "backend", limit=2, query_vector=vector, embedding_model="test-model"
        )[0].external_id == "backend"
        lexical_plan = store.explain_candidate_jobs("backend", limit=2)
        hybrid_plan = store.explain_candidate_jobs(
            "backend", limit=2, query_vector=vector, embedding_model="test-model"
        )
        assert isinstance(lexical_plan["actual_indexes"], list)
        assert isinstance(hybrid_plan["actual_indexes"], list)
        assert "plan" in lexical_plan and "plan" in hybrid_plan
        india_embedding_candidates = store.embedding_candidates(
            "another-model", limit=10, countries=["India"]
        )
        assert {job.external_id for job in india_embedding_candidates} == {
            "backend", "vision", "sales", "senior"
        }

        store.save_preferences("u", Preferences(roles=["Backend"], countries=["India"]))
        store.save_candidate_profile("u", CandidateProfile(skills=["Python"], target_roles=["Backend"]))
        store.record_feedback(FeedbackRequest(profile_id="u", job_id=sales.stable_id, verdict="reject"))
        assert store.load_preferences("u").roles == ["Backend"]
        assert store.load_candidate_profile("u").skills == ["Python"]
        assert store.rejected_job_ids("u") == {sales.stable_id}

        first = store.queue_discovery("u", "fingerprint", "backend")
        assert store.start_discovery(first["request_id"]) is True
        store.finish_discovery(first["request_id"], {"jobs_fetched": 3})
        assert store.discovery_request(first["request_id"])["status"] == "completed"
    finally:
        store.close()
