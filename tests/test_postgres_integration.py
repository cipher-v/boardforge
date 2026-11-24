import os

import pytest

from boardforge.models import CandidateProfile, FeedbackRequest, Job, Preferences
from boardforge.postgres_store import PostgresStore


POSTGRES_URL = os.getenv("BOARDFORGE_TEST_POSTGRES_URL")


@pytest.mark.skipif(not POSTGRES_URL, reason="set BOARDFORGE_TEST_POSTGRES_URL for pgvector integration")
def test_postgres_store_migrations_fts_vectors_and_state():
    store = PostgresStore(POSTGRES_URL, min_size=1, max_size=2)
    try:
        with store.connect() as connection:
            connection.execute(
                """TRUNCATE jobs,job_events,sync_runs,job_embeddings,profiles,
                   candidate_profiles,feedback,discovery_requests,discovered_sources
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
        summary = store.upsert_source_jobs("test", "integration", [backend, vision, sales])
        assert summary.inserted == 3
        assert store.candidate_jobs("Python FastAPI", limit=2)[0].external_id == "backend"
        india_candidates = store.candidate_jobs(
            "engineer",
            limit=10,
            preferences=Preferences(countries=["India"]),
        )
        assert {job.external_id for job in india_candidates} == {"backend", "vision", "sales"}
        junior_candidates = store.candidate_jobs(
            "engineer",
            limit=10,
            preferences=Preferences(
                countries=["India"],
                max_required_experience_years=1,
            ),
        )
        assert {job.external_id for job in junior_candidates} == {"backend", "vision", "sales"}

        vector = [0.0] * 768
        vector[0] = 1.0
        store.save_embedding(backend.stable_id, "test-model", "hash", vector)
        assert store.load_embedding(backend.stable_id, "test-model", "hash") == vector
        assert store.candidate_jobs(
            "backend", limit=2, query_vector=vector, embedding_model="test-model"
        )[0].external_id == "backend"

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
