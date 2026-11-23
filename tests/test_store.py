from pathlib import Path

import pytest

from boardforge.models import FeedbackRequest, Job, Preferences
from boardforge.store import Store


def j(ext: str, title: str = "AI Engineer") -> Job:
    return Job(
        source_kind="greenhouse",
        source_key="demo",
        external_id=ext,
        title=title,
        company="Demo",
        country="India",
        description="Python machine learning",
    )


def test_temporal_sync_marks_missing_job_closed(tmp_path: Path):
    store = Store(str(tmp_path / "db.sqlite"))
    first = store.upsert_source_jobs("greenhouse", "demo", [j("1"), j("2")])
    assert first.inserted == 2
    second = store.upsert_source_jobs("greenhouse", "demo", [j("1")])
    assert second.missing == 1
    assert second.closed == 0
    assert store.job_status("greenhouse:demo:2") == "active"
    third = store.upsert_source_jobs("greenhouse", "demo", [j("1")])
    assert third.closed == 1
    assert store.job_status("greenhouse:demo:2") == "closed"
    assert store.job_events("greenhouse:demo:2") == ["discovered", "closed"]


def test_empty_fetch_cannot_close_an_existing_source(tmp_path: Path):
    store = Store(str(tmp_path / "db.sqlite"))
    store.upsert_source_jobs("greenhouse", "demo", [j("1")])
    with pytest.raises(ValueError, match="empty fetch"):
        store.upsert_source_jobs("greenhouse", "demo", [])
    assert store.job_status("greenhouse:demo:1") == "active"


def test_job_update_and_reopen_are_recorded(tmp_path: Path):
    store = Store(str(tmp_path / "db.sqlite"))
    store.upsert_source_jobs("greenhouse", "demo", [j("1")])
    store.upsert_source_jobs("greenhouse", "demo", [j("1", "ML Engineer")])
    store.upsert_source_jobs("greenhouse", "demo", [j("2")])
    store.upsert_source_jobs("greenhouse", "demo", [j("2")])
    store.upsert_source_jobs("greenhouse", "demo", [j("1", "ML Engineer"), j("2")])
    assert store.job_events("greenhouse:demo:1") == ["discovered", "updated", "closed", "reopened"]


def test_preferences_and_feedback_persist(tmp_path: Path):
    store = Store(str(tmp_path / "db.sqlite"))
    p = Preferences(roles=["AI/ML"], avoid_terms=["hardware"])
    store.save_preferences("u", p)
    assert store.load_preferences("u").roles == ["AI/ML"]

    store.record_feedback(FeedbackRequest(profile_id="u", job_id="x", verdict="reject"))
    assert "x" in store.rejected_job_ids("u")
