import asyncio

from boardforge.discovery.service import queue_discovery, run_pending_discoveries
from boardforge.discovery.web import discover_and_sync
from boardforge.models import CandidateProfile, Preferences, SyncSummary
from boardforge.store import Store


class EnabledClient:
    enabled = True
    name = "test"

    async def search_urls(self, query: str, *, count: int = 20) -> list[str]:
        if "greenhouse" in query:
            return ["https://job-boards.greenhouse.io/acme/jobs/123"]
        return []


def test_discovery_requests_are_durable_and_cached(tmp_path):
    store = Store(str(tmp_path / "discovery.db"))
    profile = CandidateProfile(role_signals=["Backend"])
    prefs = Preferences(countries=["India"])

    first = queue_discovery(store, "u", profile, prefs, query="backend", client=EnabledClient())
    repeated = queue_discovery(store, "u", profile, prefs, query="backend", client=EnabledClient())
    different = queue_discovery(store, "u", profile, prefs, query="AI", client=EnabledClient())

    assert first["status"] == "queued"
    assert repeated["request_id"] == first["request_id"]
    assert repeated["cached"] is True
    assert different["request_id"] != first["request_id"]


def test_discovery_records_source_provenance(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "provenance.db"))

    async def fake_sync(url, target_store):
        return SyncSummary(
            source_kind="greenhouse", source_key="acme", fetched=4,
            inserted=4, updated=0, unchanged=0, closed=0,
        )

    monkeypatch.setattr("boardforge.discovery.web.sync_url", fake_sync)
    report = asyncio.run(discover_and_sync(
        store,
        CandidateProfile(role_signals=["Backend"]),
        Preferences(countries=["India"]),
        query="backend",
        client=EnabledClient(),
    ))

    assert report.synced_boards == 1
    assert report.jobs_fetched == 4
    sources = store.discovered_sources()
    assert len(sources) == 1
    assert sources[0]["source_kind"] == "greenhouse"
    assert sources[0]["source_key"] == "acme"
    assert sources[0]["board_url"] == "https://job-boards.greenhouse.io/acme"
    assert sources[0]["discovery_query"].startswith("manual:")
    assert "backend" not in sources[0]["discovery_query"]
    assert sources[0]["last_status"] == "synced"
    assert sources[0]["last_synced"] is not None


def test_worker_completes_queued_request_and_reuses_result(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "worker.db"))
    profile = CandidateProfile(role_signals=["Backend"])
    prefs = Preferences(countries=["India"])
    store.save_candidate_profile("u", profile)
    store.save_preferences("u", prefs)

    async def fake_sync(url, target_store):
        return SyncSummary(
            source_kind="greenhouse", source_key="acme", fetched=2,
            inserted=2, updated=0, unchanged=0, closed=0,
        )

    monkeypatch.setattr("boardforge.discovery.web.sync_url", fake_sync)
    queued = queue_discovery(store, "u", profile, prefs, query="backend", client=EnabledClient())
    worker = asyncio.run(run_pending_discoveries(store, client=EnabledClient()))
    state = store.discovery_request(queued["request_id"])
    repeated = queue_discovery(store, "u", profile, prefs, query="backend", client=EnabledClient())

    assert worker == {"processed": 1, "requeued_stale": 0}
    assert state["status"] == "completed"
    assert state["report"]["jobs_fetched"] == 2
    assert repeated["request_id"] == queued["request_id"]
    assert repeated["status"] == "completed"
    assert repeated["cached"] is True
