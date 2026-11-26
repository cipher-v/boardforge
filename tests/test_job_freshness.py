from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

import httpx
import pytest

from boardforge.connectors.ashby import AshbyConnector
from boardforge.connectors.base import ConnectorError
from boardforge.connectors.greenhouse import GreenhouseConnector
from boardforge.connectors.lever import LeverConnector
from boardforge.models import Job, Preferences, SourceRef, SyncSummary
from boardforge.postgres_store import PostgresStore
from boardforge.semantic import EmbeddingClient
from boardforge.store import Store
from boardforge.sync import sync_url
from boardforge.worker import run_worker_cycle
from boardforge.workers import sync_due_sources
from tests.test_postgres_integration import _assert_disposable_database


POSTGRES_URL = os.getenv("BOARDFORGE_TEST_POSTGRES_URL")


def _make_job(
    ext: str,
    *,
    kind: str = "greenhouse",
    key: str = "acme",
    title: str = "Backend Engineer",
    description: str = "Build Python backend services and PostgreSQL APIs.",
    apply_url: str = "https://job-boards.greenhouse.io/acme/jobs/1",
    salary_min: float | None = None,
) -> Job:
    return Job(
        source_kind=kind,
        source_key=key,
        external_id=ext,
        title=title,
        company=key,
        location="Bengaluru, India",
        country="India",
        description=description,
        apply_url=apply_url,
        salary_min=salary_min,
    )


def test_genuinely_emptied_boards_and_incomplete_or_transient_safeguards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = Store(str(tmp_path / "empty_safeguards.db"))
    monkeypatch.setattr("boardforge.sync.validate_public_url", lambda url: None)

    # 1. Malformed HTTP 200 ATS responses (missing expected list) raise ConnectorError
    class _DummyAsyncClient:
        def __init__(self, status_code: int, body):
            self._status_code = status_code
            self._body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url: str):
            return httpx.Response(self._status_code, json=self._body, request=httpx.Request("GET", url))

    monkeypatch.setattr("boardforge.connectors.greenhouse.httpx.AsyncClient", lambda **kw: _DummyAsyncClient(200, {"error": "degraded"}))
    with pytest.raises(ConnectorError, match="missing jobs list"):
        asyncio.run(GreenhouseConnector().fetch(SourceRef(kind="greenhouse", key="acme", board_url="https://job-boards.greenhouse.io/acme")))

    monkeypatch.setattr("boardforge.connectors.ashby.httpx.AsyncClient", lambda **kw: _DummyAsyncClient(200, {}))
    with pytest.raises(ConnectorError, match="missing jobs list"):
        asyncio.run(AshbyConnector().fetch(SourceRef(kind="ashby", key="acme", board_url="https://jobs.ashbyhq.com/acme")))

    monkeypatch.setattr("boardforge.connectors.lever.httpx.AsyncClient", lambda **kw: _DummyAsyncClient(200, {"ok": False}))
    with pytest.raises(ConnectorError, match="expected postings list"):
        asyncio.run(LeverConnector().fetch(SourceRef(kind="lever", key="acme", board_url="https://jobs.lever.co/acme")))

    # 2. Small board (<10 active jobs): unverified empty rejected; 1st verified empty keeps active; 2nd closes cleanly
    store.upsert_source_jobs("greenhouse", "smallco", [_make_job("1", key="smallco"), _make_job("2", key="smallco")])
    with pytest.raises(ValueError, match="empty fetch"):
        store.upsert_source_jobs("greenhouse", "smallco", [])

    # Transient HTTP 503 failure does not touch missing_count or close jobs
    monkeypatch.setattr("boardforge.connectors.greenhouse.httpx.AsyncClient", lambda **kw: _DummyAsyncClient(503, {}))
    with pytest.raises(ConnectorError):
        asyncio.run(sync_url("https://job-boards.greenhouse.io/smallco", store))
    assert store.job_status("greenhouse:smallco:1") == "active"

    # First verified HTTP 200 {"jobs": []} increments missing_count to 1, closes 0 jobs
    monkeypatch.setattr("boardforge.connectors.greenhouse.httpx.AsyncClient", lambda **kw: _DummyAsyncClient(200, {"jobs": []}))
    first_empty = asyncio.run(sync_url("https://job-boards.greenhouse.io/smallco", store))
    assert first_empty.fetched == 0
    assert first_empty.missing == 2
    assert first_empty.closed == 0
    assert store.job_status("greenhouse:smallco:1") == "active"

    # Second consecutive verified HTTP 200 {"jobs": []} closes all active jobs without deleting rows
    second_empty = asyncio.run(sync_url("https://job-boards.greenhouse.io/smallco", store))
    assert second_empty.closed == 2
    assert store.job_status("greenhouse:smallco:1") == "closed"
    assert store.job_status("greenhouse:smallco:2") == "closed"
    assert store.active_jobs() == []
    assert store.job_events("greenhouse:smallco:1") == ["discovered", "closed"]
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs WHERE source_key='smallco'").fetchone()[0] == 2

    # 3. Larger board (>=10 active jobs): partial shrink still blocked; requires 3 consecutive verified-empty fetches
    large_jobs = [_make_job(str(i), key="largeco") for i in range(12)]
    store.upsert_source_jobs("greenhouse", "largeco", large_jobs)

    # Truncated partial response (2 of 12 < 25%) is rejected
    with pytest.raises(ValueError, match="abnormal source shrink"):
        store.upsert_source_jobs("greenhouse", "largeco", large_jobs[:2], allow_verified_empty=True)

    # 1st and 2nd verified-empty responses keep all 12 jobs active; 3rd consecutive verified-empty closes them
    e1 = asyncio.run(sync_url("https://job-boards.greenhouse.io/largeco", store))
    e2 = asyncio.run(sync_url("https://job-boards.greenhouse.io/largeco", store))
    assert e1.closed == 0 and e2.closed == 0
    assert len(store.active_jobs()) == 12
    e3 = asyncio.run(sync_url("https://job-boards.greenhouse.io/largeco", store))
    assert e3.closed == 12
    assert store.active_jobs() == []


def test_configurable_stale_job_visibility_policy_preserves_records_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = Store(str(tmp_path / "stale_vis.db"))
    fresh_job = _make_job("fresh-1", key="freshco", title="Fresh Backend Engineer")
    stale_job = _make_job("stale-1", key="deadboard", title="Stale Backend Engineer")
    store.upsert_source_jobs("greenhouse", "freshco", [fresh_job])
    store.upsert_source_jobs("lever", "deadboard", [stale_job])

    now = datetime.now(timezone.utc)
    old_45d = (now - timedelta(days=45)).isoformat()
    with store.connect() as conn:
        conn.execute("UPDATE jobs SET first_seen=?, last_seen=? WHERE source_key='deadboard'", (old_45d, old_45d))

    # Default 30-day policy hides the 45-day unrefreshed source from active_jobs and candidate_jobs,
    # while keeping the stored record intact and inspectable via include_stale=True.
    monkeypatch.setenv("BOARDFORGE_STALE_JOB_MAX_AGE_DAYS", "30")
    assert [j.external_id for j in store.active_jobs()] == ["fresh-1"]
    assert [j.external_id for j in store.candidate_jobs("Backend Engineer", limit=10)] == ["fresh-1"]
    assert {j.external_id for j in store.active_jobs(include_stale=True)} == {"fresh-1", "stale-1"}
    assert store.job_status(stale_job.stable_id) == "active"

    # Configurable policy: widening to 60 days or disabling (0) restores visibility without DB writes
    monkeypatch.setenv("BOARDFORGE_STALE_JOB_MAX_AGE_DAYS", "60")
    assert {j.external_id for j in store.active_jobs()} == {"fresh-1", "stale-1"}
    monkeypatch.setenv("BOARDFORGE_STALE_JOB_MAX_AGE_DAYS", "0")
    assert {j.external_id for j in store.active_jobs()} == {"fresh-1", "stale-1"}

    # Recovery: under 30-day policy, once the source refreshes again, last_seen updates and visibility returns
    monkeypatch.setenv("BOARDFORGE_STALE_JOB_MAX_AGE_DAYS", "30")
    assert [j.external_id for j in store.active_jobs()] == ["fresh-1"]
    recovered = store.upsert_source_jobs("lever", "deadboard", [stale_job])
    assert recovered.unchanged == 1
    assert {j.external_id for j in store.active_jobs()} == {"fresh-1", "stale-1"}


def test_bounded_worker_cycles_refresh_all_sources_without_starvation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = Store(str(tmp_path / "anti_starvation.db"))
    now = datetime.now(timezone.utc)
    oldest_72h = now - timedelta(hours=72)
    older_48h = now - timedelta(hours=48)

    # Register 13 boards (> per-cycle limit of 5):
    # B01..B05 are 72h old AND persistently fail; B06..B13 are 48h old and healthy!
    failing_keys = {f"b{i:02d}" for i in range(1, 6)}
    healthy_keys = {f"b{i:02d}" for i in range(6, 14)}
    for idx in range(1, 14):
        key = f"b{idx:02d}"
        ts = oldest_72h if key in failing_keys else older_48h
        store.record_discovered_source(
            "greenhouse",
            key,
            f"https://job-boards.greenhouse.io/{key}",
            "indexed-backfill",
            synced=True,
            status="synced",
            synced_at=ts,
            discovered_at=ts,
        )

    synced_seen: list[str] = []

    async def fake_sync(url: str, target_store: Store) -> SyncSummary:
        key = url.rsplit("/", 1)[-1]
        if key in failing_keys:
            raise ConnectorError(f"Simulated 503 outage for {key}")
        synced_seen.append(key)
        return target_store.upsert_source_jobs("greenhouse", key, [_make_job("1", key=key)])

    monkeypatch.setattr("boardforge.workers.sync_url", fake_sync)

    # Cycle 1: attempts oldest B01..B05 (all 5 fail and enter backoff; last_discovered updated to now)
    c1 = asyncio.run(sync_due_sources(store, interval_seconds=3600, limit=5, concurrency=2))
    assert c1["due"] == 5
    assert c1["failed"] == 5
    assert c1["synced"] == 0

    # Cycle 2: B01..B05 do NOT starve B06..B10!
    c2 = asyncio.run(sync_due_sources(store, interval_seconds=3600, limit=5, concurrency=2))
    assert c2["due"] == 5
    assert c2["synced"] == 5

    # Cycle 3: refreshes remaining healthy B11..B13!
    c3 = asyncio.run(sync_due_sources(store, interval_seconds=3600, limit=5, concurrency=2))
    assert c3["due"] == 3
    assert c3["synced"] == 3
    assert set(synced_seen) == healthy_keys

    # Cycle 4 (immediately after): 0 due (healthy are fresh, failed are in retry backoff)
    c4 = asyncio.run(sync_due_sources(store, interval_seconds=3600, limit=5, concurrency=2))
    assert c4["due"] == 0


def test_unchanged_jobs_do_not_require_regenerated_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("BOARDFORGE_SEARCH_PROVIDER", "none")
    monkeypatch.setenv("BOARDFORGE_EMBEDDING_JOBS_PER_CYCLE", "20")
    store = Store(str(tmp_path / "embed_reuse.db"))

    j1 = _make_job("1", key="reuseco", title="Backend Engineer", description="Python FastAPI services")
    j2 = _make_job("2", key="reuseco", title="ML Engineer", description="PyTorch vision models")
    store.upsert_source_jobs("greenhouse", "reuseco", [j1, j2])

    class _CountingEmbedProvider:
        enabled = True
        model = "counting:768"
        dimensions = 4

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def embed(self, texts: list[str], *, task_type: str = "RETRIEVAL_DOCUMENT"):
            self.calls.append(list(texts))
            return [[1.0, 0.0, 0.5, 0.25] for _ in texts]

    provider = _CountingEmbedProvider()
    client = EmbeddingClient(provider)

    # Initial worker cycle embeds both jobs once
    first = asyncio.run(run_worker_cycle(store, client))
    assert first["embeddings"]["processed"] == 2
    assert len(provider.calls) == 1

    # Re-syncing unchanged jobs updates last_seen (unchanged=2) and requires 0 embedding calls
    resync = store.upsert_source_jobs("greenhouse", "reuseco", [j1, j2])
    assert resync.unchanged == 2
    second = asyncio.run(run_worker_cycle(store, client))
    assert second["embeddings"]["processed"] == 0
    assert len(provider.calls) == 1

    # Non-text metadata update (apply_url / salary_min) keeps job_text unchanged -> 0 re-embeddings!
    j1_meta_updated = j1.model_copy(update={"apply_url": "https://job-boards.greenhouse.io/reuseco/jobs/999", "salary_min": 1800000})
    meta_sync = store.upsert_source_jobs("greenhouse", "reuseco", [j1_meta_updated, j2])
    assert meta_sync.updated == 1
    third = asyncio.run(run_worker_cycle(store, client))
    assert third["embeddings"]["processed"] == 0
    assert len(provider.calls) == 1

    # Modifying actual description text on j2 re-embeds ONLY j2 (processed == 1)
    j2_text_changed = j2.model_copy(update={"description": "PyTorch computer vision and TensorRT inference"})
    store.upsert_source_jobs("greenhouse", "reuseco", [j1_meta_updated, j2_text_changed])
    fourth = asyncio.run(run_worker_cycle(store, client))
    assert fourth["embeddings"]["processed"] == 1
    assert len(provider.calls) == 2


@pytest.mark.skipif(not POSTGRES_URL, reason="set BOARDFORGE_TEST_POSTGRES_URL for PostgreSQL freshness test")
def test_postgres_freshness_stale_visibility_starvation_and_embedding_reuse(monkeypatch: pytest.MonkeyPatch):
    _assert_disposable_database(POSTGRES_URL)
    store = PostgresStore(POSTGRES_URL, min_size=1, max_size=2)
    try:
        with store.connect() as conn:
            actual_db = conn.execute("SELECT current_database() AS name").fetchone()["name"]
            assert actual_db == "boardforge_test"
            conn.execute(
                """TRUNCATE jobs,job_events,sync_runs,job_embeddings,profiles,
                   candidate_profiles,feedback,discovery_requests,discovered_sources,
                   llm_usage_reservations,relevance_verifications,query_embeddings
                   RESTART IDENTITY CASCADE"""
            )

        fresh = _make_job("pg-fresh", key="freshpg", description="Python FastAPI backend")
        stale = _make_job("pg-stale", key="stalepg", description="Python FastAPI backend")
        store.upsert_source_jobs("greenhouse", "freshpg", [fresh])
        store.upsert_source_jobs("greenhouse", "stalepg", [stale])

        now = datetime.now(timezone.utc)
        old_40d = now - timedelta(days=40)
        with store.connect() as conn:
            conn.execute("UPDATE jobs SET first_seen=%s, last_seen=%s WHERE source_key='stalepg'", (old_40d, old_40d))

        # 1. Stale visibility policy on PostgreSQL (active_jobs, candidate_jobs, embedding_candidates)
        monkeypatch.setenv("BOARDFORGE_STALE_JOB_MAX_AGE_DAYS", "30")
        assert [j.external_id for j in store.active_jobs()] == ["pg-fresh"]
        assert [j.external_id for j in store.candidate_jobs("Python", limit=10, preferences=Preferences(countries=["India"]))] == ["pg-fresh"]
        assert len(store.active_jobs(include_stale=True)) == 2

        # 2. Embedding reuse on PostgreSQL across unchanged re-sync and non-text metadata update
        vec = [0.0] * 768
        vec[0] = 1.0
        cands = store.embedding_candidates("vertex:gemini-embedding-001:768", limit=10)
        assert [j.external_id for j in cands] == ["pg-fresh"]
        with store.connect() as conn:
            emb_hash = conn.execute("SELECT embedding_hash FROM jobs WHERE stable_id=%s", (fresh.stable_id,)).fetchone()["embedding_hash"]
        store.save_embedding(fresh.stable_id, "vertex:gemini-embedding-001:768", emb_hash, vec)
        assert store.embedding_candidates("vertex:gemini-embedding-001:768", limit=10) == []

        store.upsert_source_jobs("greenhouse", "freshpg", [fresh.model_copy(update={"salary_min": 1500000})])
        assert store.embedding_candidates("vertex:gemini-embedding-001:768", limit=10) == []

        # 3. Anti-starvation on PostgreSQL when oldest source fails
        t_72h = now - timedelta(hours=72)
        t_48h = now - timedelta(hours=48)
        store.record_discovered_source(
            "greenhouse", "broken", "https://job-boards.greenhouse.io/broken", "indexed-backfill",
            synced=True, status="synced", synced_at=t_72h, discovered_at=t_72h,
        )
        store.record_discovered_source(
            "greenhouse", "healthy", "https://job-boards.greenhouse.io/healthy", "indexed-backfill",
            synced=True, status="synced", synced_at=t_48h, discovered_at=t_48h,
        )
        # First due with limit=1 is oldest 'broken'; after it fails, next due with limit=1 is 'healthy'!
        assert [r["source_key"] for r in store.sources_due(interval_seconds=3600, limit=1)] == ["broken"]
        store.record_discovered_source(
            "greenhouse", "broken", "https://job-boards.greenhouse.io/broken", "scheduled-refresh",
            status="failed",
        )
        assert [r["source_key"] for r in store.sources_due(interval_seconds=3600, limit=1)] == ["healthy"]
    finally:
        store.close()


def test_optional_cloud_scheduler_12h_configuration_is_present_and_opt_in():
    deploy_script = (Path(__file__).parents[1] / "deploy" / "gcp" / "deploy.ps1").read_text(encoding="utf-8")
    assert "[switch] $EnableScheduledMaintenance" in deploy_script
    assert '$MaintenanceSchedule = "0 */12 * * *"' in deploy_script
    assert "if ($EnableScheduledMaintenance)" in deploy_script
    assert "cloudscheduler.googleapis.com" in deploy_script
    assert "gcloud scheduler jobs create http" in deploy_script
    assert "gcloud scheduler jobs update http" in deploy_script
    assert "/jobs/$JobName`:run" in deploy_script
    assert "--oauth-service-account-email $serviceAccount" in deploy_script
    assert "BOARDFORGE_STALE_JOB_MAX_AGE_DAYS=" in deploy_script
