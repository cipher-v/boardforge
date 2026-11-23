from pathlib import Path

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
    assert second.closed == 1
    assert store.job_status("greenhouse:demo:2") == "closed"


def test_preferences_and_feedback_persist(tmp_path: Path):
    store = Store(str(tmp_path / "db.sqlite"))
    p = Preferences(roles=["AI/ML"], avoid_terms=["hardware"])
    store.save_preferences("u", p)
    assert store.load_preferences("u").roles == ["AI/ML"]

    store.record_feedback(FeedbackRequest(profile_id="u", job_id="x", verdict="reject"))
    assert "x" in store.rejected_job_ids("u")
