from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import boardforge.api as api_module
from boardforge.agent import JobAgent
from boardforge.agent_tools import AgentExecutionState, AgentToolExecutor
from boardforge.confidence import parse_query_intent
from boardforge.enrich import enrich_job, infer_remote_policy, reenrich_store_jobs
from boardforge.intelligent_agent import ToolCallingAgent
from boardforge.matching import rank_jobs, role_evidence
from boardforge.models import CandidateProfile, FeedbackRequest, Job, Preferences
from boardforge.planning import tool_names
from boardforge.preferences import (
    deterministic_extract,
    effective_search_preferences,
    extract_experience_cap,
    extract_experience_floor,
)
from boardforge.semantic import (
    EmbeddingClient,
    lexical_query_text,
    retrieve_and_rank_jobs,
    warm_job_embeddings,
)
from boardforge.store import Store


class _DeterministicExtractor:
    async def extract(self, text: str, current: Preferences | None = None) -> Preferences:
        return deterministic_extract(text, current)


class _DisabledEmbeddings:
    enabled = False
    model = "disabled"


def _make_agent(tmp_path: Path) -> tuple[JobAgent, Store]:
    store = Store(str(tmp_path / "qa_remediation.db"))
    agent = JobAgent(store, extractor=_DeterministicExtractor(), embeddings=_DisabledEmbeddings())
    return agent, store


def _make_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[TestClient, Store]:
    monkeypatch.setenv("BOARDFORGE_ENABLE_INTELLIGENT_AGENT", "0")
    monkeypatch.setenv("BOARDFORGE_ENABLE_EMBEDDINGS", "0")
    monkeypatch.setenv("BOARDFORGE_ENABLE_RELEVANCE_VERIFIER", "0")
    store = Store(str(tmp_path / "api_remediation.db"))
    core = JobAgent(store, extractor=_DeterministicExtractor(), embeddings=_DisabledEmbeddings())
    tool = ToolCallingAgent(core)
    monkeypatch.setattr(api_module, "store", store)
    monkeypatch.setattr(api_module, "core_agent", core)
    monkeypatch.setattr(api_module, "tool_agent", tool)
    return TestClient(api_module.app), store


# ---------------------------------------------------------------------------
# Phase A — Search execution, preference isolation, endpoints, and traces
# ---------------------------------------------------------------------------


def test_noun_phrase_searches_execute_without_imperative_verbs(tmp_path: Path):
    agent, store = _make_agent(tmp_path)
    store.upsert_source_jobs("test", "phase_a", [
        Job(
            source_kind="test", source_key="phase_a", external_id="swe-jr",
            title="Software Engineer", company="Acme", country="India",
            location="Bengaluru, India", remote_policy="hybrid",
            min_experience_years=0.0,
            description="Build backend services and Python APIs.",
        ),
        Job(
            source_kind="test", source_key="phase_a", external_id="de-1",
            title="Data Engineer", company="Acme", country="India",
            location="Pune, India", remote_policy="onsite",
            description="Build Spark and Airflow ETL data pipelines.",
        ),
    ])

    queries = [
        "Software engineering jobs for candidates with no formal employment experience",
        "Hybrid Python backend roles in India",
        "Only Bengaluru backend roles",
        "Data engineering roles in India",
    ]
    for query in queries:
        steps = tool_names(query)
        assert steps == ["search_jobs"], f"Expected search-only plan for {query!r}, got {steps}"
        result = asyncio.run(agent.handle("u", query))
        assert "matches" in result.data
        assert len(result.data["matches"]) >= 1
        assert store.load_preferences("u") == Preferences()


def test_ordinary_search_queries_do_not_poison_saved_preferences(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    client, store = _make_client(monkeypatch, tmp_path)
    store.upsert_source_jobs("test", "poison", [
        Job(
            source_kind="test", source_key="poison", external_id="chennai-be",
            title="Backend Engineer", company="ChennaiCo", country="India",
            location="Chennai, India", remote_policy="onsite",
            min_experience_years=1.0, salary_text="12 LPA",
            description="Build Python backend microservices.",
        ),
        Job(
            source_kind="test", source_key="poison", external_id="blr-ai",
            title="AI Engineer", company="BlrAI", country="India",
            location="Bengaluru, India", remote_policy="remote",
            min_experience_years=3.0,
            description="Train and deploy LLM and machine learning systems.",
        ),
    ])

    strict_turn = client.post("/agent", json={
        "profile_id": "u",
        "query": (
            "Find onsite-only backend roles in Chennai requiring up to 1 year "
            "of experience, paid only, avoid hardware and embedded work"
        ),
    })
    assert strict_turn.status_code == 200
    strict_body = strict_turn.json()
    assert [m["job"]["external_id"] for m in strict_body["matches"]] == ["chennai-be"]

    # Durable preferences must remain completely untouched after an ordinary search.
    saved = client.get("/preferences/u")
    assert saved.status_code == 200
    assert saved.json()["preferences"] == Preferences().model_dump()

    # Follow-up broader query must NOT inherit Chennai / onsite-only / <=1yr caps.
    followup = client.post("/agent", json={
        "profile_id": "u",
        "query": "AI/ML roles in India",
    })
    assert followup.status_code == 200
    followup_ids = [m["job"]["external_id"] for m in followup.json()["matches"]]
    assert "blr-ai" in followup_ids


def test_explicit_preference_updates_still_persist_and_tool_guard_blocks_searches(tmp_path: Path):
    agent, store = _make_agent(tmp_path)
    store.upsert_source_jobs("test", "prefs", [
        Job(
            source_kind="test", source_key="prefs", external_id="ai-1",
            title="AI Engineer", company="Demo", country="India",
            location="Bengaluru, India", description="Machine learning systems.",
        ),
    ])

    r1 = asyncio.run(agent.handle("u", "From now on, only India roles and avoid QA"))
    assert [s["tool"] for s in r1.data["trace"]] == ["update_preferences"]
    prefs = store.load_preferences("u")
    assert prefs.countries == ["India"]
    assert "qa" in [t.casefold() for t in prefs.avoid_terms]

    # Conversational "changed my mind" without explicit persistence is transient.
    r_transient = asyncio.run(agent.handle("u", "Changed my mind, show me QA roles"))
    assert [s["tool"] for s in r_transient.data["trace"]] == ["search_jobs"]
    assert "qa" in [t.casefold() for t in store.load_preferences("u").avoid_terms]

    # Explicit durable persistence rehabilitates previously avoided roles.
    r2 = asyncio.run(agent.handle("u", "From now on, include QA roles"))
    assert [s["tool"] for s in r2.data["trace"]] == ["update_preferences"]
    prefs2 = store.load_preferences("u")
    assert "QA" in prefs2.roles
    assert "qa" not in [t.casefold() for t in prefs2.avoid_terms]

    # AgentToolExecutor must refuse update_preferences on an ordinary search turn
    # even if an LLM passes a verbatim quote.
    executor = AgentToolExecutor(agent)
    state = AgentExecutionState(
        profile_id="u2",
        user_text="Find onsite-only backend roles in Chennai",
    )
    denied = asyncio.run(executor.execute(state, "update_preferences", {
        "text": "onsite-only backend roles in Chennai",
        "authorization_quote": "onsite-only backend roles in Chennai",
    }))
    assert denied["ok"] is False
    assert denied["error"] == "preference_update_not_authorized"
    assert store.load_preferences("u2") == Preferences()


def test_preferences_endpoints_auth_isolation_headers_and_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    client, store = _make_client(monkeypatch, tmp_path)
    monkeypatch.setenv("BOARDFORGE_PROFILE_TOKENS", json.dumps({
        "token-a": "profile-a",
        "token-b": "profile-b",
    }))

    store.save_candidate_profile("profile-a", CandidateProfile(
        skills=["Python", "FastAPI"], role_signals=["Backend"], resume_filename="cv.pdf",
    ))
    store.upsert_source_jobs("test", "fb", [
        Job(source_kind="test", source_key="fb", external_id="j1", title="Backend Engineer", company="C"),
    ])
    asyncio.run(api_module.core_agent.record_feedback(FeedbackRequest(
        profile_id="profile-a", job_id="test:fb:j1", verdict="save", note="keep",
    )))

    # Unauthenticated -> 401
    assert client.get("/preferences/profile-a").status_code == 401
    assert client.delete("/preferences/profile-a").status_code == 401

    # Cross-profile -> 403
    headers_b = {"Authorization": "Bearer token-b"}
    assert client.get("/preferences/profile-a", headers=headers_b).status_code == 403
    assert client.delete("/preferences/profile-a", headers=headers_b).status_code == 403

    # Authorized write + read + cache headers
    headers_a = {"Authorization": "Bearer token-a"}
    updated = client.post(
        "/preferences",
        headers=headers_a,
        json={"profile_id": "profile-a", "text": "Only India roles, remote only"},
    )
    assert updated.status_code == 200
    got = client.get("/preferences/profile-a", headers=headers_a)
    assert got.status_code == 200
    assert got.headers.get("Cache-Control") == "no-store, private"
    assert "Authorization" in got.headers.get("Vary", "")
    assert got.json()["preferences"]["countries"] == ["India"]
    assert got.json()["preferences"]["onsite_ok"] is False

    # Reset clears preferences ONLY, keeping candidate_profile and feedback intact.
    cleared = client.delete("/preferences/profile-a", headers=headers_a)
    assert cleared.status_code == 200
    assert cleared.headers.get("Cache-Control") == "no-store, private"
    assert cleared.json()["preferences"] == Preferences().model_dump()
    assert store.load_preferences("profile-a") == Preferences()
    assert store.load_candidate_profile("profile-a") is not None
    assert any(f["job_id"] == "test:fb:j1" for f in store.feedback_history("profile-a"))


def test_deterministic_tool_agent_trace_marks_steps_completed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    client, store = _make_client(monkeypatch, tmp_path)
    store.upsert_source_jobs("test", "trace", [
        Job(source_kind="test", source_key="trace", external_id="1", title="Backend Engineer", company="C"),
    ])
    resp = client.post("/agent", json={"profile_id": "u", "query": "Find backend jobs"})
    assert resp.status_code == 200
    trace = resp.json()["trace"]
    assert len(trace) == 1
    assert trace[0]["tool"] == "search_jobs"
    assert trace[0]["ok"] is True


# ---------------------------------------------------------------------------
# Phase B — Workplace classification, eligibility, and occupational relevance
# ---------------------------------------------------------------------------


def test_workplace_classification_prioritizes_structured_location_and_title():
    # Structured Hybrid overrides company-level "remote-first" boilerplate
    assert infer_remote_policy(
        "Bengaluru, India (Hybrid)",
        "Backend Engineer",
        "We are a remote-first company with hubs in India.",
    ) == "hybrid"

    # Structured Onsite overrides remote boilerplate
    assert infer_remote_policy(
        "Chennai, India - In-Office",
        "Software Engineer",
        "Collaborate with remote teammates around the world.",
    ) == "onsite"

    # Technical "hybrid cloud / hybrid search" does not misclassify as hybrid workplace
    assert infer_remote_policy(
        "Bengaluru, India",
        "Hybrid Cloud & Hybrid Search Engineer",
        "Design hybrid retrieval and hybrid cloud architectures.",
    ) == "unknown"

    # Incidental "remote-first" boilerplate without structured remote location stays unknown
    job = enrich_job(Job(
        source_kind="greenhouse", source_key="demo", external_id="1",
        title="Software Engineer", company="Demo",
        location="Bengaluru, India",
        description="Join our remote-first engineering culture working from our Bengaluru hub.",
    ))
    assert job.remote_policy == "unknown"
    assert job.country == "India"

    # Explicit remote location is classified as remote
    remote_job = enrich_job(Job(
        source_kind="greenhouse", source_key="demo", external_id="2",
        title="Backend Engineer", company="Demo",
        location="Remote - India",
        description="Build distributed services.",
    ))
    assert remote_job.remote_policy == "remote"
    assert remote_job.country == "India"


def test_hyphenated_remote_only_and_onsite_only_constraints():
    remote_prefs = effective_search_preferences(
        Preferences(), "Find remote-only Python backend jobs"
    )
    assert remote_prefs.remote_ok is True
    assert remote_prefs.onsite_ok is False

    onsite_prefs = effective_search_preferences(
        Preferences(), "Find onsite-only backend roles in Bengaluru"
    )
    assert onsite_prefs.remote_ok is False
    assert onsite_prefs.onsite_ok is True


def test_international_remote_jobs_blocked_by_city_and_country_constraints(tmp_path: Path):
    store = Store(str(tmp_path / "geo.db"))
    jobs = [
        Job(
            source_kind="test", source_key="geo", external_id="us-remote",
            title="Backend Engineer", company="USCo",
            location="Remote - United States", country="United States",
            remote_policy="remote", description="Python backend APIs.",
        ),
        Job(
            source_kind="test", source_key="geo", external_id="ie-remote",
            title="Backend Engineer", company="IECo",
            location="Remote, Ireland", country="Ireland",
            remote_policy="remote", description="Python backend APIs.",
        ),
        Job(
            source_kind="test", source_key="geo", external_id="in-remote",
            title="Backend Engineer", company="INRemote",
            location="Remote - India", country="India",
            remote_policy="remote", description="Python backend APIs.",
        ),
        Job(
            source_kind="test", source_key="geo", external_id="blr-onsite",
            title="Backend Engineer", company="BlrOnsite",
            location="Bengaluru, India", country="India",
            remote_policy="onsite", description="Python backend APIs.",
        ),
    ]
    store.upsert_source_jobs("test", "geo", jobs)

    # City query without explicit "India" must NOT admit US/Ireland remote jobs.
    city_prefs = effective_search_preferences(Preferences(), "Backend jobs in Bengaluru")
    city_matches = rank_jobs(store.active_jobs(), city_prefs, query="Backend jobs in Bengaluru")
    city_ids = {m.job.external_id for m in city_matches}
    assert "us-remote" not in city_ids
    assert "ie-remote" not in city_ids
    assert "blr-onsite" in city_ids

    # Onsite-only in Bengaluru must also exclude Remote - India.
    onsite_prefs = effective_search_preferences(Preferences(), "Onsite-only backend jobs in Bengaluru")
    onsite_matches = rank_jobs(store.active_jobs(), onsite_prefs, query="Onsite-only backend jobs in Bengaluru")
    assert [m.job.external_id for m in onsite_matches] == ["blr-onsite"]

    # Also verify PostgresStore SQL candidate filter when test Postgres is configured.
    pg_url = os.getenv("BOARDFORGE_TEST_POSTGRES_URL")
    if pg_url:
        from boardforge.postgres_store import PostgresStore
        pg = PostgresStore(pg_url)
        with pg.connect() as conn:
            conn.execute("DELETE FROM jobs WHERE source_kind='test' AND source_key='geo'")
        pg.upsert_source_jobs("test", "geo", jobs)
        try:
            pg_candidates = pg.candidate_jobs("backend", preferences=city_prefs)
            pg_ids = {j.external_id for j in pg_candidates if j.source_key == "geo"}
            assert "us-remote" not in pg_ids
            assert "ie-remote" not in pg_ids
            assert "blr-onsite" in pg_ids
        finally:
            with pg.connect() as conn:
                conn.execute("DELETE FROM jobs WHERE source_kind='test' AND source_key='geo'")


def test_entry_level_fresher_and_no_experience_caps_with_numeric_precedence():
    no_exp = deterministic_extract(
        "Software engineering jobs for candidates with no formal employment experience"
    )
    assert no_exp.max_required_experience_years == 0.0

    fresher = deterministic_extract("Entry-level backend jobs for freshers in India")
    assert fresher.max_required_experience_years == 1.0

    # Explicit numeric cap overrides default entry-level 1.0 cap in both directions.
    two_yr = deterministic_extract(
        "Entry-level backend jobs requiring up to 2 years of experience"
    )
    assert two_yr.max_required_experience_years == 2.0

    zero_yr = deterministic_extract(
        "Entry-level backend jobs requiring at most 0 years of experience"
    )
    assert zero_yr.max_required_experience_years == 0.0


def test_unnegated_minimum_experience_not_inverted_and_reported_transparently(tmp_path: Path):
    # Unnegated minimum phrase must NOT become a maximum cap.
    q = "Backend jobs in India requiring more than 5 years of experience"
    assert extract_experience_cap(q) is None
    assert extract_experience_floor(q) == 5.0
    prefs = effective_search_preferences(Preferences(), q)
    assert prefs.max_required_experience_years is None

    # Negated exceeding phrase MUST still become a maximum cap.
    neg = "Avoid roles requiring more than 2 years of experience"
    assert extract_experience_cap(neg) == 2.0
    assert extract_experience_floor(neg) is None

    agent, store = _make_agent(tmp_path)
    store.upsert_source_jobs("test", "minexp", [
        Job(
            source_kind="test", source_key="minexp", external_id="sr-6",
            title="Senior Backend Engineer", company="Demo", country="India",
            location="Bengaluru, India", min_experience_years=6.0,
            description="Python backend distributed systems.",
        ),
        Job(
            source_kind="test", source_key="minexp", external_id="jr-1",
            title="Backend Engineer", company="Demo", country="India",
            location="Bengaluru, India", min_experience_years=1.0,
            description="Python backend distributed systems.",
        ),
    ])
    result = asyncio.run(agent.search_jobs("u", q, 10)).data
    assert result["retrieval"]["unsupported_constraints"]
    by_id = {m["job"]["external_id"]: m for m in result["matches"]}
    assert "sr-6" in by_id
    assert by_id["sr-6"]["status"] == "recommended"
    assert by_id["jr-1"]["status"] == "needs_verification"
    assert any("below requested 5+ years" in r for r in by_id["jr-1"]["risks"])


def test_negated_clauses_stripped_from_positive_query_intent():
    q = "Find AI and backend roles in India, avoid hardware and embedded work"
    prefs = effective_search_preferences(Preferences(), q)
    assert "hardware" in prefs.avoid_terms
    assert "embedded" in prefs.avoid_terms
    intent = parse_query_intent(prefs, q)
    assert "hardware" not in intent.specific_terms
    assert "embedded" not in intent.specific_terms
    assert "work" not in intent.specific_terms

    followup_q = "Changed my mind: show me worldwide remote AI/ML roles in Pune or anywhere"
    followup_prefs = effective_search_preferences(Preferences(), followup_q)
    assert parse_query_intent(followup_prefs, followup_q).specific_terms == frozenset()

    now_q = "Now show me senior backend roles in Hyderabad"
    now_prefs = effective_search_preferences(Preferences(), now_q)
    assert "now" not in parse_query_intent(now_prefs, now_q).specific_terms


def test_missing_occupational_aliases_and_backend_role_boundaries():
    assert deterministic_extract("Data engineering roles in India").roles == ["Data Engineering"]
    assert deterministic_extract("Analytics engineering jobs").roles == ["Data Engineering"]
    assert deterministic_extract("Data science roles").roles == ["Data Science"]
    assert deterministic_extract("Automation testing and SDET roles").roles == ["QA"]

    fe_focused = Job(
        source_kind="test", source_key="role", external_id="fe",
        title="Full Stack Engineer (Frontend Focused)", company="Demo",
        country="India", description="React, TypeScript, CSS design systems.",
    )
    sdet_backend = Job(
        source_kind="test", source_key="role", external_id="sdet",
        title="Senior SDET - Backend Automation", company="Demo",
        country="India", description="Automated API testing suites.",
    )
    true_backend = Job(
        source_kind="test", source_key="role", external_id="be",
        title="Backend Software Engineer", company="Demo",
        country="India", description="Build backend services and APIs.",
    )
    assert role_evidence(fe_focused, "Backend") is None
    assert role_evidence(sdet_backend, "Backend") is None
    assert role_evidence(true_backend, "Backend") is not None


def test_seniority_titles_penalized_and_flagged_for_entry_level_without_fabricating_years():
    senior_unstated = Job(
        source_kind="test", source_key="sen", external_id="sr",
        title="Senior Backend Engineer", company="Demo", country="India",
        location="Bengaluru, India", min_experience_years=None,
        description="Build Python backend services and APIs.",
    )
    ic_unstated = Job(
        source_kind="test", source_key="sen", external_id="ic",
        title="Backend Engineer", company="Demo", country="India",
        location="Bengaluru, India", min_experience_years=None,
        description="Build Python backend services and APIs.",
    )
    q = "Entry-level backend jobs in India requiring up to 1 year of experience"
    prefs = effective_search_preferences(Preferences(), q)
    ranked = rank_jobs([senior_unstated, ic_unstated], prefs, query=q, top_k=10)
    assert ranked[0].job.external_id == "ic"
    sr_match = next(m for m in ranked if m.job.external_id == "sr")
    assert sr_match.job.min_experience_years is None
    assert sr_match.status == "needs_verification"
    assert "seniority title conflicts with entry-level target" in sr_match.risks


def test_safe_reenrich_store_jobs_preserves_embedding_hashes(tmp_path: Path):
    store = Store(str(tmp_path / "reenrich.db"))
    legacy_job = Job(
        source_kind="greenhouse", source_key="demo", external_id="gh-1",
        title="Backend Engineer", company="Demo",
        location="Bengaluru, India (Hybrid)", country="India",
        remote_policy="remote",  # stale legacy value caused by "remote-first" boilerplate
        description="Join our remote-first engineering organization in our Bengaluru hub.",
    )
    store.upsert_source_jobs("greenhouse", "demo", [legacy_job])
    client = EmbeddingClient(provider=_FakeSemanticProvider())
    asyncio.run(warm_job_embeddings(store, client))
    assert store.embedding_status(client.model)["embedded_jobs"] == 1

    with store.connect() as conn:
        before_hash = conn.execute(
            "SELECT content_hash FROM jobs WHERE stable_id=?",
            (legacy_job.stable_id,),
        ).fetchone()["content_hash"]

    dry = reenrich_store_jobs(store, apply=False)
    assert dry["mode"] == "dry_run"
    assert dry["jobs_changed"] == 1
    assert dry["transitions"] == {"remote->hybrid": 1}
    assert store.active_jobs()[0].remote_policy == "remote"

    applied = reenrich_store_jobs(store, apply=True)
    assert applied["mode"] == "applied"
    assert applied["jobs_changed"] == 1
    assert store.active_jobs()[0].remote_policy == "hybrid"

    # Cached embedding remains valid because job_text excludes remote_policy/country.
    assert store.embedding_status(client.model)["embedded_jobs"] == 1
    with store.connect() as conn:
        after_hash = conn.execute(
            "SELECT content_hash FROM jobs WHERE stable_id=?",
            (legacy_job.stable_id,),
        ).fetchone()["content_hash"]
    assert after_hash != before_hash


# ---------------------------------------------------------------------------
# Phase C — Retrieval overlap, FTS lexical separation, validation, and metrics
# ---------------------------------------------------------------------------


class _FakeSemanticProvider:
    enabled = True
    model = "fake:768"
    dimensions = 4

    async def embed(self, texts: list[str], *, task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        del task_type
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def test_zero_overlap_queries_return_empty_while_preserving_semantic_candidates(tmp_path: Path):
    store = Store(str(tmp_path / "overlap.db"))
    job = Job(
        source_kind="test", source_key="ov", external_id="py-1",
        title="Backend Engineer", company="Demo", country="India",
        location="Bengaluru, India",
        description="Build distributed microservices in Python and PostgreSQL.",
    )
    store.upsert_source_jobs("test", "ov", [job])

    # Unrelated non-semantic query returns honest empty results instead of arbitrary top-k.
    unrelated = "underwater quantum cobol mainframe roles"
    prefs = effective_search_preferences(Preferences(), unrelated)
    assert rank_jobs(store.active_jobs(), prefs, query=unrelated, top_k=10) == []

    # When embeddings are active and return semantic neighbors, semantic candidates survive.
    client = EmbeddingClient(provider=_FakeSemanticProvider())
    asyncio.run(warm_job_embeddings(store, client))
    matches, meta = asyncio.run(retrieve_and_rank_jobs(
        store,
        Preferences(),
        query="server architecture roles",
        top_k=5,
        rejected_ids=set(),
        client=client,
        candidate_limit=50,
    ))
    assert meta["semantic_retrieval"] is True
    assert len(matches) == 1
    assert matches[0].job.external_id == "py-1"


def test_lexical_query_text_isolates_explicit_query_from_resume_prefer_terms():
    prefs = Preferences(
        roles=["Backend"],
        prefer_terms=[
            "Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes",
            "Terraform", "PyTorch", "Redis", "Kafka", "GraphQL",
        ],
    )
    lex = lexical_query_text(prefs, "Remote Python backend jobs")
    assert " or " in lex
    assert "python" in lex
    assert "backend" in lex
    assert "pytorch" not in lex
    assert "terraform" not in lex
    assert "kubernetes" not in lex


def test_nul_bytes_and_top_k_bounds_rejected_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    client, _ = _make_client(monkeypatch, tmp_path)

    for endpoint, payload in (
        ("/search", {"profile_id": "u", "query": "python\x00backend"}),
        ("/agent", {"profile_id": "u", "query": "python\x00backend"}),
        ("/preferences", {"profile_id": "u", "text": "India\x00only"}),
        ("/feedback", {"profile_id": "u", "job_id": "a\x00b", "verdict": "save", "note": ""}),
    ):
        resp = client.post(endpoint, json=payload)
        assert resp.status_code in (400, 422), f"{endpoint} returned {resp.status_code}"

    for bad_k in (0, -5, 51, 999999):
        assert client.post("/search", json={"profile_id": "u", "query": "python", "top_k": bad_k}).status_code == 422
        assert client.post("/agent", json={"profile_id": "u", "query": "python", "top_k": bad_k}).status_code == 422
        assert client.get(f"/recommendations/u?top_k={bad_k}").status_code == 422


def test_unmatched_urls_use_bounded_metric_route_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    client, _ = _make_client(monkeypatch, tmp_path)
    sentinel = "/unmatched-probe-9f8e7d6c5b"
    miss = client.get(sentinel)
    assert miss.status_code == 404

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert sentinel not in metrics.text
    assert 'route="__unmatched__"' in metrics.text
