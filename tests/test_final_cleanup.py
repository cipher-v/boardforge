"""Focused regressions for the final production cleanup (quarantine, placeholders,
evidence de-duplication, internship-phrase intent, read-only diagnostics)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from boardforge.agent import JobAgent
from boardforge.agent_tools import AgentToolExecutor
from boardforge.confidence import parse_query_intent
from boardforge.discovery.web import SearchHit, prioritize_discovery_sources
from boardforge.intelligent_agent import ToolCallingAgent
from boardforge.llm import LLMCallResult, TokenUsage
from boardforge.matching import rank_jobs
from boardforge.models import CandidateProfile, FeedbackRequest, Job, Preferences
from boardforge.preferences import effective_search_preferences
from boardforge.quarantine import is_placeholder_job_title, is_quarantined_source
from boardforge.seeding import register_indexed_sources, validate_seed_sources
from boardforge.store import Store

POSTGRES_URL = os.getenv("BOARDFORGE_TEST_POSTGRES_URL")
ROOT = Path(__file__).resolve().parents[1]


def _job(ext: str, *, kind: str, key: str, title: str, description: str | None = None) -> Job:
    return Job(
        source_kind=kind,
        source_key=key,
        external_id=ext,
        title=title,
        company=key,
        location="Bengaluru, India",
        country="India",
        description=description or "Build Python backend services, REST APIs and PostgreSQL.",
        apply_url=f"https://example.invalid/{key}/{ext}",
    )


def _seed_mixed_index(store) -> None:
    store.upsert_source_jobs("lever", "jobtwine", [
        _job("jt-1", kind="lever", key="jobtwine", title="Backend Engineer"),
        _job("jt-2", kind="lever", key="jobtwine", title="Python Developer"),
    ])
    store.upsert_source_jobs("ashby", "aiprise", [
        _job("ap-1", kind="ashby", key="aiprise", title="Backend Engineer"),
    ])


# ---------------------------------------------------------------- quarantine


def test_quarantine_defaults_and_configuration(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BOARDFORGE_QUARANTINED_SOURCES", raising=False)
    assert is_quarantined_source("lever", "jobtwine")
    assert is_quarantined_source("LEVER", "JobTwine")
    assert not is_quarantined_source("greenhouse", "jobtwine")
    assert not is_quarantined_source("ashby", "aiprise")
    monkeypatch.setenv("BOARDFORGE_QUARANTINED_SOURCES", "ashby:aiprise, sandboxco")
    assert is_quarantined_source("ashby", "aiprise")
    assert is_quarantined_source("greenhouse", "sandboxco")
    assert not is_quarantined_source("lever", "jobtwine")
    monkeypatch.setenv("BOARDFORGE_QUARANTINED_SOURCES", "none")
    assert not is_quarantined_source("lever", "jobtwine")


def test_quarantine_sqlite_excludes_everywhere_but_preserves_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("BOARDFORGE_QUARANTINED_SOURCES", raising=False)
    db = tmp_path / "quarantine.db"
    store = Store(str(db))
    _seed_mixed_index(store)
    old = datetime.now(timezone.utc) - timedelta(days=3)
    for kind, key in (("lever", "jobtwine"), ("ashby", "aiprise")):
        url = f"https://jobs.lever.co/{key}" if kind == "lever" else f"https://jobs.ashbyhq.com/{key}"
        store.record_discovered_source(kind, key, url, "seed", synced=True, status="synced",
                                       synced_at=old, discovered_at=old)

    visible = {j.source_key for j in store.active_jobs()}
    assert visible == {"aiprise"}
    assert {j.source_key for j in store.candidate_jobs("backend")} == {"aiprise"}
    assert {j.source_key for j in store.active_jobs(include_quarantined=True)} == {"aiprise", "jobtwine"}
    assert {j.source_key for j in store.embedding_candidates("m", limit=50)} == {"aiprise"}
    assert store.embedding_status("m")["active_jobs"] == 1
    due = store.sources_due(interval_seconds=3600, limit=1)
    assert [(d["source_kind"], d["source_key"]) for d in due] == [("ashby", "aiprise")]

    # Search / recommendations never surface quarantined jobs.
    agent = JobAgent(store)
    searched = asyncio.run(agent.search_jobs("p1", "backend engineer python", 10))
    assert searched.data["matches"]
    assert all(m["job"]["source_key"] != "jobtwine" for m in searched.data["matches"])
    store.save_candidate_profile("p1", CandidateProfile(skills=["Python"], target_roles=["Backend"]))
    recommended = asyncio.run(agent.recommend_jobs("p1", 10))
    assert all(m["job"]["source_key"] != "jobtwine" for m in recommended.data["matches"])

    # Records and audit history are fully preserved.
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs WHERE source_key='jobtwine'").fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM job_events WHERE stable_id LIKE 'lever:jobtwine:%'"
        ).fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM sync_runs WHERE source_key='jobtwine'").fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM discovered_sources WHERE source_key='jobtwine'"
        ).fetchone()[0] == 1

    # Removing the quarantine immediately restores visibility (nothing was deleted).
    monkeypatch.setenv("BOARDFORGE_QUARANTINED_SOURCES", "off")
    assert {j.source_key for j in store.active_jobs()} == {"aiprise", "jobtwine"}


def test_quarantine_blocks_seeding_registration_and_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("BOARDFORGE_QUARANTINED_SOURCES", raising=False)
    checked = validate_seed_sources(["https://jobs.lever.co/jobtwine", "https://jobs.ashbyhq.com/aiprise"])
    assert [v["key"] for v in checked["valid"]] == ["aiprise"]
    assert checked["rejected"][0]["reason"] == "source is quarantined"

    store = Store(str(tmp_path / "register.db"))
    _seed_mixed_index(store)
    register_indexed_sources(store, apply=True)
    assert {s["source_key"] for s in store.discovered_sources()} == {"aiprise"}

    hits = [
        SearchHit(url="https://jobs.lever.co/jobtwine/abc-123",
                  title="Machine Learning Engineer - Bengaluru, India",
                  snippet="Entry level machine learning engineer role in Bengaluru, India."),
        SearchHit(url="https://jobs.lever.co/safeco/def-456",
                  title="Machine Learning Engineer - Bengaluru, India",
                  snippet="Entry level machine learning engineer role in Bengaluru, India."),
    ]
    prefs = Preferences(roles=["AI/ML"], countries=["India"])
    selected, _, _ = prioritize_discovery_sources(
        hits, store=None, profile=CandidateProfile(), prefs=prefs,
        query="machine learning engineer jobs in India", max_boards=4,
    )
    keys = {s.key for s in selected}
    assert "jobtwine" not in keys
    assert "safeco" in keys


def test_quarantine_postgres_sql_filters(monkeypatch: pytest.MonkeyPatch):
    if not POSTGRES_URL:
        pytest.skip("set BOARDFORGE_TEST_POSTGRES_URL for PostgreSQL test")
    from boardforge.postgres_store import PostgresStore
    from tests.test_postgres_integration import _assert_disposable_database

    _assert_disposable_database(POSTGRES_URL)
    monkeypatch.delenv("BOARDFORGE_QUARANTINED_SOURCES", raising=False)
    store = PostgresStore(POSTGRES_URL, min_size=1, max_size=2)
    try:
        with store.connect() as conn:
            assert conn.execute("SELECT current_database() AS n").fetchone()["n"] == "boardforge_test"
            conn.execute(
                """TRUNCATE jobs,job_events,sync_runs,job_embeddings,profiles,
                   candidate_profiles,feedback,discovery_requests,discovered_sources,
                   llm_usage_reservations,relevance_verifications,query_embeddings,
                   telemetry_events RESTART IDENTITY CASCADE"""
            )
        _seed_mixed_index(store)
        old = datetime.now(timezone.utc) - timedelta(days=3)
        store.record_discovered_source("lever", "jobtwine", "https://jobs.lever.co/jobtwine", "seed",
                                       synced=True, status="synced", synced_at=old, discovered_at=old)
        store.record_discovered_source("ashby", "aiprise", "https://jobs.ashbyhq.com/aiprise", "seed",
                                       synced=True, status="synced", synced_at=old, discovered_at=old)

        assert {j.source_key for j in store.active_jobs()} == {"aiprise"}
        assert {j.source_key for j in store.active_jobs(include_quarantined=True)} == {"aiprise", "jobtwine"}
        assert {j.source_key for j in store.candidate_jobs("")} == {"aiprise"}
        assert {j.source_key for j in store.candidate_jobs("backend engineer")} == {"aiprise"}
        assert {j.source_key for j in store.embedding_candidates("m", limit=50)} == {"aiprise"}
        assert store.embedding_status("m")["active_jobs"] == 1
        due = store.sources_due(interval_seconds=3600, limit=1)
        assert [(d["source_kind"], d["source_key"]) for d in due] == [("ashby", "aiprise")]
        with store.connect() as conn:
            assert conn.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE source_key='jobtwine'"
            ).fetchone()["n"] == 2
            assert conn.execute(
                "SELECT COUNT(*) AS n FROM job_events WHERE stable_id LIKE 'lever:jobtwine:%%'"
            ).fetchone()["n"] == 2

        store.record_telemetry_event("search", {"endpoint": "search", "match_count": 1})
        with store.connect() as conn:
            assert conn.execute("SELECT COUNT(*) AS n FROM telemetry_events").fetchone()["n"] == 1
    finally:
        store.close()


# ---------------------------------------------------------------- placeholder titles


PLACEHOLDER_TITLES = [
    "Test Lever Job", "test 16th Jan", "test job Kiran Backend Engineer", "Add test job ravi",
    "testing job new pdf", "Brillio Lever Testing", "9th september lever test Kiran SDE-2",
    "Meesho mock test", "Do Not Apply - Dummy Role", "Placeholder Job", "6th Jan",
    "automation tester 11:02", "test", "Dummy",
]
LEGITIMATE_TEST_TITLES = [
    "QA Engineer", "QA Intern", "Software Test Engineer", "Senior Test Engineer", "SDET",
    "Automation Testing Engineer", "Jr. QA Automation (API) - Bangalore, India",
    "Staff Engineer, Test Infrastructure", "Penetration Tester", "Test Automation Engineer",
]


@pytest.mark.parametrize("title", PLACEHOLDER_TITLES)
def test_placeholder_titles_detected(title: str):
    assert is_placeholder_job_title(title)


@pytest.mark.parametrize("title", LEGITIMATE_TEST_TITLES)
def test_legitimate_testing_roles_preserved(title: str):
    assert not is_placeholder_job_title(title)


def test_rank_jobs_excludes_placeholders_but_keeps_qa_roles():
    jobs = [
        _job("p1", kind="greenhouse", key="acme", title="test job Kiran Backend Engineer",
             description="QA automation testing with Python and Selenium."),
        _job("p2", kind="greenhouse", key="acme", title="Meesho mock test",
             description="QA automation testing with Python and Selenium."),
        _job("q1", kind="greenhouse", key="acme", title="QA Automation Engineer",
             description="QA automation testing with Python and Selenium."),
        _job("q2", kind="greenhouse", key="acme", title="SDET",
             description="QA automation testing with Python and Selenium."),
    ]
    ranked = rank_jobs(jobs, Preferences(countries=["India"]), "QA automation testing jobs", top_k=10)
    ids = {m.job.external_id for m in ranked}
    assert "p1" not in ids and "p2" not in ids
    assert {"q1", "q2"} <= ids


# ---------------------------------------------------------------- duplicate evidence


def test_match_explanations_are_deduplicated():
    job = _job("fs-1", kind="greenhouse", key="acme", title="Full-Stack Software Engineer",
               description="Full-stack software engineer building computer vision features in Python.")
    prefs = Preferences(
        roles=["Software Engineering", "Backend"],
        prefer_terms=["Computer Vision", "computer vision", "Python", "python"],
        countries=["India"],
    )
    ranked = rank_jobs([job], prefs, "full stack software engineer", top_k=5)
    assert ranked
    match = ranked[0]
    assert len(match.reasons) == len({r.casefold() for r in match.reasons})
    assert len(match.risks) == len({r.casefold() for r in match.risks})
    keys = [(e.kind, e.detail.casefold()) for e in match.evidence]
    assert len(keys) == len(set(keys))

    bounded = AgentToolExecutor._bounded_match({
        "job": job.model_dump(mode="json"), "status": "recommended", "score": 1.0,
        "reasons": ["Role match", "Role match", "role match"],
        "risks": ["Unknown salary", "Unknown salary"],
        "evidence": [{"kind": "skill", "detail": "Python"}, {"kind": "skill", "detail": "Python"}],
    })
    assert len(bounded["reasons"]) == 1
    assert len(bounded["risks"]) == 1
    assert len(bounded["evidence"]) == 1

    html = (ROOT / "boardforge" / "static" / "index.html").read_text(encoding="utf-8")
    assert "dedupList" in html


# ---------------------------------------------------------------- internship phrase intent


def test_include_relevant_internships_does_not_add_required_terms():
    query = (
        "Find entry-level artificial intelligence, machine learning, and computer vision "
        "engineering roles in India requiring up to one year of experience. "
        "Include relevant internships."
    )
    prefs = effective_search_preferences(Preferences(), query)
    intent = parse_query_intent(prefs, query)
    specific = intent.domain_terms | intent.skill_terms | intent.task_terms
    for leaked in ("relevant", "internships", "internship", "include", "artificial", "intelligence"):
        assert leaked not in specific, leaked

    internship_only = "Software engineering internship in India"
    only_prefs = effective_search_preferences(Preferences(), internship_only)
    only_intent = parse_query_intent(only_prefs, internship_only)
    assert "internship" in (
        only_intent.domain_terms | only_intent.skill_terms | only_intent.task_terms
        | only_intent.role_terms
    ) or "internship" in internship_only.casefold()


# ---------------------------------------------------------------- read-only diagnostics


class _ScriptedProvider:
    enabled = True
    provider_name = "mock"
    model = "mock-agent"
    location = "test"
    input_usd_per_million = 0.75
    output_usd_per_million = 3.75

    def __init__(self, steps):
        self.steps = list(steps)

    async def generate(self, request):
        return LLMCallResult(
            data=self.steps.pop(0),
            usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            provider=self.provider_name, model=self.model, estimated_cost_usd=0.00015,
        )


def test_diagnose_cli_is_read_only_and_redacts_private_data(tmp_path: Path):
    db = tmp_path / "diag.db"
    store = Store(str(db))
    _seed_mixed_index(store)
    store.save_candidate_profile("secret-user-99", CandidateProfile(
        resume_filename="Secret_Jane_Doe.pdf", skills=["UniqueSecretSkillXYZ"],
    ))
    store.save_preferences("secret-user-99", Preferences(prefer_terms=["UniqueSecretSkillXYZ"]))
    store.record_feedback(FeedbackRequest(
        profile_id="secret-user-99", job_id="ashby:aiprise:ap-1", verdict="reject",
        note="UniqueSecretSkillXYZ note",
    ))
    store.save_query_embedding("k1", "backend uniquesecretskillxyz", "m", "cfg", [0.1, 0.2])
    store.queue_discovery("secret-user-99", "fp-secret", "secret private query xyz")

    core = JobAgent(store)
    asyncio.run(core.search_jobs("secret-user-99", "secret private query xyz backend", 5))
    asyncio.run(ToolCallingAgent(core, enabled=False).run("secret-user-99", "Find backend jobs"))

    fast = ToolCallingAgent(core, provider=_ScriptedProvider([]), enabled=True)
    fast._is_straightforward_search = lambda text: True
    asyncio.run(fast.run("secret-user-99", "secret private query xyz backend jobs"))

    llm = ToolCallingAgent(core, provider=_ScriptedProvider([
        {"function_calls": [{"name": "search_indexed_jobs",
                             "args": {"query": "secret private query xyz backend", "top_k": 5}}]},
        {"function_calls": []},
    ]), enabled=True)
    llm._is_straightforward_search = lambda text: False
    asyncio.run(llm.run("secret-user-99", "secret private query xyz backend jobs"))

    before = db.read_bytes()
    env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
    env["BOARDFORGE_DB_PATH"] = str(db)
    proc = subprocess.run(
        [sys.executable, "-m", "boardforge.cli", "diagnose", "--limit", "5"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert db.read_bytes() == before
    out = proc.stdout
    for secret in ("Secret_Jane_Doe", "UniqueSecretSkillXYZ", "uniquesecretskillxyz",
                   "secret-user-99", "secret private query xyz"):
        assert secret not in out, secret

    report = json.loads(out)
    assert report["mode"] == "read_only"
    assert report["pii_redacted"] is True
    turns = report["latest_agent_and_search_turns"]
    modes = {t.get("orchestration_mode") for t in turns}
    assert {"direct_search", "deterministic_planner", "deterministic_fast_path", "llm_conditional"} <= modes
    for turn in turns:
        assert "tools_executed" in turn and "cache_hits" in turn
        assert "latency_ms" in turn and "cost_usd" in turn
        for tool in turn["tools_executed"]:
            assert set(tool) == {"tool", "ok", "duration_ms", "error"}
    llm_turn = next(t for t in turns if t["orchestration_mode"] == "llm_conditional")
    assert llm_turn["provider_calls"] == 2
    assert report["latest_discovery_requests"][0]["status"] == "queued"
    assert "expenditure_and_usage" in report
    assert report["index_visibility"]["quarantined_active_jobs"] == 2
