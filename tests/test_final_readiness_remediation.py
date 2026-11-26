from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys

import pytest

from boardforge.agent import JobAgent
from boardforge.agent_tools import AgentToolExecutor
from boardforge.confidence import assess_query_relevance, occupational_query_text, parse_query_intent
from boardforge.discovery import (
    SearchHit,
    build_discovery_query,
    discover_and_sync,
    discovery_fingerprint,
    prioritize_discovery_sources,
)
from boardforge.enrich import (
    enrich_job,
    extract_fresher_eligibility_evidence,
    extract_job_experience,
)
from boardforge.intelligent_agent import ToolCallingAgent
from boardforge.llm import LLMCallResult, TokenUsage
from boardforge.matching import hard_violations, rank_jobs, role_evidence
from boardforge.models import CandidateProfile, Job, Match, Preferences, SyncSummary
from boardforge.preferences import (
    deterministic_extract,
    effective_search_preferences,
    strip_orchestration_instructions,
)
from boardforge.relevance import (
    RelevanceVerifier,
    evidence_segments,
    normalize_intent,
    verification_cache_key,
)
from boardforge.seeding import (
    apply_seed_sources,
    load_seed_file,
    matches_expected_database,
    register_indexed_sources,
    validate_seed_sources,
)
from boardforge.semantic import (
    EmbeddingClient,
    lexical_query_text,
    query_embedding_cache_identity,
    semantic_query_text,
)
from boardforge.store import Store
from boardforge.worker import run_worker_cycle


AIPRISE_SWE1_DESCRIPTION = (
    "ABOUT AIPRISE\n\n"
    "AiPrise is a global compliance orchestration platform helping fintechs, marketplaces, "
    "and payment companies verify and onboard customers worldwide.\n\n"
    "ROLE OVERVIEW\n\n"
    "We’re hiring a Software Engineer I to help build and scale the core platform behind "
    "AiPrise’s compliance orchestration and AI agent systems.\n\n"
    "This is a build it, ship it, own it role. You’ll take features end to end, from design "
    "to code to deploy to iterate, across our case management product, vendor integrations, "
    "orchestration APIs, and customer facing tools.\n\n"
    "WHAT YOU'LL DO\n\n"
    " - Design, develop, and ship production features across our platform stack "
    "(backend APIs, data pipelines, integrations, case management UI)\n\n"
    " - Own complete features from requirements gathering → implementation → deployment → monitoring\n\n"
    " - Build scalable REST APIs and microservices that orchestrate checks across 80+ compliance vendors\n\n"
    " - Write clean, maintainable, well-tested code with comprehensive documentation\n\n"
    "MUST-HAVE:\n\n"
    " - 0-2 years of professional software engineering experience building and shipping production systems\n"
)

BOLNA_SOLUTIONS_DESCRIPTION = (
    "About Bolna\n\n"
    "Bolna is Voice AI infrastructure built for India. We help businesses deploy intelligent voice agents.\n\n"
    "The Role\n\n"
    "As an AI Solutions Engineer, you'll sit at the intersection of our product and our customers. "
    "Your primary job is to design, write, and iterate on the prompts and tools that power Bolna's voice agents.\n\n"
    "You'll work closely with customers to understand their goals, build agent flows, test conversations, "
    "and fix what breaks. No heavy coding required-if you can vibe-code a basic script or write a solid "
    "system prompt, you're qualified.\n\n"
    "Why Bolna\n\n"
    " - Freshers who joined 3 months back are single-handedly working with CXOs and Senior PMs\n"
    " - Competitive compensation for early-career profiles\n"
)

ZIMPERIUM_DEVOPS_DESCRIPTION = (
    "Zimperium is an industry leader in enterprise mobile security. Our MTD and award-winning "
    "machine learning-based engine protects against device, network, and phishing attacks.\n\n"
    "We are seeking a Software Engineer with strong AI and Python expertise and deep understanding "
    "of cloud infrastructure and distributed systems.\n\n"
    "Responsibilities:\n"
    "Design, develop, and maintain backend services using Python\n"
    "Work with cloud platforms (AWS, GCP, OCI, or Azure) to design and operate scalable systems\n"
    "Implement Infrastructure-as-Code using tools such as Terraform, Helm, or CloudFormation\n"
    "Collaborate with DevOps, SRE, Security, and Product Engineering teams\n"
)

NEURAFLASH_BOILERPLATE = (
    "Why NeuraFlash, Part of Accenture: At NeuraFlash, Part of Accenture, we are redefining the "
    "future of business through the power of AI and groundbreaking technologies like Agentforce. "
    "As a trusted leader in AI, AWS, and Salesforce innovation, we craft intelligent solutions."
)


class _MockEmbeddingProvider:
    enabled = True
    model = "mock:768"
    dimensions = 4

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def embed(self, texts: list[str], *, task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        self.calls.append((task_type, list(texts)))
        return [[1.0, 0.5, 0.25, 0.125] for _ in texts]


class _MockRelevanceProvider:
    enabled = True
    provider_name = "vertex-gemini"
    model = "gemini-3.8-flash-mock"
    location = "global"
    input_usd_per_million = 0.75
    output_usd_per_million = 3.75

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def verify_batch(self, payload: dict) -> LLMCallResult:
        self.calls.append(payload)
        data = self.responses.pop(0)
        return LLMCallResult(
            data=data,
            usage=TokenUsage(input_tokens=400, output_tokens=80, thinking_tokens=20, total_tokens=500),
            provider=self.provider_name,
            model=self.model,
            latency_ms=42.5,
            attempts=1,
            estimated_cost_usd=0.000675,
        )


def test_orchestration_clauses_do_not_contaminate_retrieval_or_discovery():
    plain = "Find entry-level computer vision jobs in India requiring up to one year of experience."
    with_cond_a = (
        "Find entry-level computer vision jobs in India requiring up to one year of experience. "
        "If there are insufficient matches, discover more sources."
    )
    with_cond_b = (
        "Find entry-level computer vision jobs in India requiring up to one year of experience. "
        "If fewer than three suitable opportunities exist, discover additional job sources "
        "while preserving my location and experience constraints."
    )
    agent_rewrite = "entry-level computer vision jobs in India <=1 year experience"

    prefs_plain = effective_search_preferences(Preferences(), plain)
    for variant in (with_cond_a, with_cond_b, agent_rewrite):
        prefs_var = effective_search_preferences(Preferences(), variant)
        assert prefs_var.roles == prefs_plain.roles == ["Computer Vision"]
        assert prefs_var.countries == prefs_plain.countries == ["India"]
        assert prefs_var.max_required_experience_years == prefs_plain.max_required_experience_years == 1.0

    # Equivalent occupational retrieval intent before discovery
    occ_plain = occupational_query_text(plain)
    intent_plain = parse_query_intent(prefs_plain, plain)
    assert intent_plain.specific_terms == frozenset()

    for variant in (with_cond_a, with_cond_b):
        assert occupational_query_text(variant) == occ_plain
        assert parse_query_intent(prefs_plain, variant) == intent_plain
        assert lexical_query_text(prefs_plain, variant) == lexical_query_text(prefs_plain, plain)
        assert semantic_query_text(prefs_plain, variant) == semantic_query_text(prefs_plain, plain)
        assert normalize_intent(variant) == normalize_intent(plain)
        assert build_discovery_query(CandidateProfile(), Preferences(), variant) == build_discovery_query(
            CandidateProfile(), Preferences(), plain
        )
        assert discovery_fingerprint(
            CandidateProfile(), Preferences(), variant, profile_id="u"
        ) == discovery_fingerprint(CandidateProfile(), Preferences(), plain, profile_id="u")

    # Smoke Test B regression: never emit "only 1/12 specific query terms were supported"
    smoke_b_query = (
        "Find entry-level computer vision and machine learning engineering jobs in India "
        "requiring up to one year of experience. If fewer than three suitable opportunities "
        "exist, discover additional job sources while preserving my location and experience constraints."
    )
    cv_job = Job(
        source_kind="ashby",
        source_key="visionco",
        external_id="cv-1",
        title="Computer Vision Engineer",
        company="VisionCo",
        location="Bengaluru, India",
        country="India",
        description="Train PyTorch computer vision perception models and deploy inference pipelines.",
        min_experience_years=0.0,
        max_experience_years=1.0,
    )
    smoke_prefs = effective_search_preferences(Preferences(), smoke_b_query)
    assessment = assess_query_relevance(cv_job, smoke_prefs, smoke_b_query)
    assert assessment.confident_topic is True
    assert all("specific query terms were supported" not in r for r in assessment.reasons)

    # Verification cache key is also invariant to orchestration suffixes
    assert verification_cache_key(cv_job, plain, prefs_plain, "gemini-3.8-flash") == (
        verification_cache_key(cv_job, with_cond_b, prefs_plain, "gemini-3.8-flash")
    )


def test_occupational_relevance_vs_candidate_eligibility_and_fresher_evidence():
    direct_ml_fresher = enrich_job(Job(
        source_kind="greenhouse",
        source_key="mlco",
        external_id="ml-fresh",
        title="Machine Learning Engineer",
        company="MLCo",
        location="Bengaluru, India",
        country="India",
        description=(
            "Build and train PyTorch machine learning models for ranking and vision. "
            "Requirements: Strong Python and ML fundamentals. No prior experience required; freshers welcome."
        ),
    ))
    bolna = enrich_job(Job(
        source_kind="ashby",
        source_key="bolna",
        external_id="bolna-sol",
        title="AI Solutions Engineer",
        company="bolna",
        location="Bengaluru",
        country="India",
        description=BOLNA_SOLUTIONS_DESCRIPTION,
    ))
    adjacent_unknown_exp = enrich_job(Job(
        source_kind="ashby",
        source_key="voiceco",
        external_id="sol-unk",
        title="AI Solutions Engineer",
        company="VoiceCo",
        location="Bengaluru, India",
        country="India",
        description="Design LLM prompts and deploy voice AI agent workflows for enterprise customers.",
    ))
    direct_unknown_exp = enrich_job(Job(
        source_kind="lever",
        source_key="visionx",
        external_id="cv-unk",
        title="Computer Vision Engineer",
        company="VisionX",
        location="Pune, India",
        country="India",
        description="Develop OpenCV and deep learning object detection models.",
    ))
    zimperium_devops = enrich_job(Job(
        source_kind="lever",
        source_key="zimperium",
        external_id="zimp-devops",
        title="AI Software Engineer (DevOps) - Bangalore, India",
        company="zimperium",
        location="India (Remote)",
        country="India",
        description=ZIMPERIUM_DEVOPS_DESCRIPTION,
    ))
    hundred_ms_security = enrich_job(Job(
        source_kind="lever",
        source_key="100ms",
        external_id="100ms-sec",
        title="Security Engineer - AI Agents (Healthcare)",
        company="100ms",
        location="Bengaluru",
        country="India",
        description="",
    ))
    four_year_ml = enrich_job(Job(
        source_kind="greenhouse",
        source_key="seniorml",
        external_id="ml-4yr",
        title="Machine Learning Engineer",
        company="SeniorML",
        location="Bengaluru, India",
        country="India",
        description="Requirements: 4+ years of hands-on machine learning engineering experience.",
    ))

    # Missing numerical experience is NEVER fabricated as 0.0 years
    assert direct_ml_fresher.min_experience_years is None
    assert bolna.min_experience_years is None
    assert direct_unknown_exp.min_experience_years is None
    assert four_year_ml.min_experience_years == 4.0

    # Explicit fresher vs contextual early-career vs absent evidence
    assert extract_fresher_eligibility_evidence(direct_ml_fresher.title, direct_ml_fresher.description)[0] == "explicit"
    assert extract_fresher_eligibility_evidence(bolna.title, bolna.description)[0] == "early_career_signal"
    assert extract_fresher_eligibility_evidence(direct_unknown_exp.title, direct_unknown_exp.description) is None

    query = "Find entry-level computer vision and machine learning engineering jobs in India requiring up to one year of experience."
    prefs = effective_search_preferences(Preferences(), query)
    assert "experience" in hard_violations(four_year_ml, prefs, query=query)

    ranked = rank_jobs(
        [
            four_year_ml,
            zimperium_devops,
            hundred_ms_security,
            direct_unknown_exp,
            adjacent_unknown_exp,
            bolna,
            direct_ml_fresher,
        ],
        prefs,
        query=query,
        top_k=10,
    )
    by_id = {m.job.external_id: m for m in ranked}

    # 4+ year requirement and unrelated DevOps / empty-JD Security roles are excluded on CV/ML search
    assert "ml-4yr" not in by_id
    assert "zimp-devops" not in by_id
    assert "100ms-sec" not in by_id

    # Direct ML role with explicit fresher/no-experience eligibility -> recommended
    assert by_id["ml-fresh"].status == "recommended"

    # Bolna AI Solutions Engineer -> preserved as adjacent 'alternative', never 'recommended'
    assert by_id["bolna-sol"].status == "alternative"
    assert any("adjacent" in r for r in by_id["bolna-sol"].risks)
    assert any("numeric experience" in r for r in by_id["bolna-sol"].risks)

    # Adjacent role with completely unknown experience communicates BOTH facts
    assert by_id["sol-unk"].status == "alternative"
    assert any("adjacent" in r for r in by_id["sol-unk"].risks)
    assert "experience requirement not verified" in by_id["sol-unk"].risks

    # Direct CV role with unknown experience remains needs_verification
    assert by_id["cv-unk"].status == "needs_verification"
    assert "experience requirement not verified" in by_id["cv-unk"].risks

    # Ordering: recommended -> needs_verification -> alternative
    statuses = [m.status for m in ranked]
    assert statuses == ["recommended", "needs_verification", "alternative", "alternative"]


def test_aiprise_software_engineer_i_verifier_accepts_substantive_bullet_segments(tmp_path: Path):
    store = Store(str(tmp_path / "aiprise_ver.db"))
    aiprise = Job(
        source_kind="ashby",
        source_key="aiprise",
        external_id="3763c791",
        title="Software Engineer I (Bangalore, India)",
        company="aiprise",
        location="Bengaluru",
        country="India",
        description=AIPRISE_SWE1_DESCRIPTION,
        min_experience_years=0.0,
        max_experience_years=2.0,
    )
    segs = evidence_segments(aiprise.description, max_chars=6000)
    # Find actual bullet segments (including curly-apostrophe "You’ll" and "Own complete features" / "Write clean...")
    own_seg = next(s["evidence_id"] for s in segs if "Own complete features" in s["text"])
    write_seg = next(s["evidence_id"] for s in segs if "Write clean, maintainable" in s["text"])

    for query in (
        "Find entry-level software engineering roles in India",
        "Find entry-level software engineering and backend roles in India suitable for my profile. "
        "I have no formal employment experience. Include roles requiring up to one year of experience.",
    ):
        prefs = effective_search_preferences(Preferences(), query)
        provider = _MockRelevanceProvider([{
            "judgments": [{
                "candidate_ref": "C001",
                "relevance": "direct",
                "evidence_ids": [own_seg, write_seg],
                "reasoning": "Core responsibilities are end-to-end feature ownership and production software development.",
            }]
        }])
        verifier = RelevanceVerifier(store, provider, enabled=True)
        initial = rank_jobs([aiprise], prefs, query=query, top_k=5)
        verified, meta = asyncio.run(verifier.verify_matches(
            initial, intent=query, preferences=prefs, rejected_ids=set()
        ))
        assert verified[0].status == "recommended"
        assert meta["candidate_validation_failures"] == {}


def test_discovery_rejects_neuraflash_crm_boilerplate_and_handles_failed_boards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = Store(str(tmp_path / "disc_smoke_b.db"))
    # Seed Jumio as already-indexed
    store.upsert_source_jobs("greenhouse", "jumio", [
        Job(
            source_kind="greenhouse",
            source_key="jumio",
            external_id="j-1",
            title="Senior Computer Vision Engineer",
            company="jumio",
            location="Bengaluru, India",
            country="India",
            description="5+ years in biometrics and computer vision.",
            min_experience_years=5.0,
        )
    ])
    store.record_discovered_source(
        "greenhouse", "jumio", "https://boards.greenhouse.io/jumio", "seed", synced=True, status="synced"
    )

    hits = [
        # NeuraFlash: Salesforce/CRM + referral + support hits with heavy "AI & Agentforce" boilerplate
        SearchHit(
            url="https://job-boards.greenhouse.io/neuraflash/jobs/6231611004",
            title="Salesforce Developer, AI & Agentforce (India)",
            snippet=NEURAFLASH_BOILERPLATE + " Remote - India (Bengaluru, Pune).",
            provider_score=0.88,
        ),
        SearchHit(
            url="https://job-boards.greenhouse.io/neuraflash/jobs/5675154004",
            title="Trying to submit a referral for a role not currently posted on our job site? Submit that referral here!",
            snippet=NEURAFLASH_BOILERPLATE + " Pune - India; Bangalore.",
            provider_score=0.82,
        ),
        SearchHit(
            url="https://job-boards.greenhouse.io/neuraflash/jobs/6249909004",
            title="Support Engineer, Amazon Connect",
            snippet=NEURAFLASH_BOILERPLATE + " Remote - India.",
            provider_score=0.79,
        ),
        # Already-indexed Jumio
        SearchHit(
            url="https://boards.greenhouse.io/jumio/jobs/999",
            title="Junior Computer Vision Engineer - Bengaluru, India",
            snippet="0-1 years of experience in OpenCV and PyTorch in India.",
            provider_score=0.90,
        ),
        # Dead/delisted 404 board (like zind-erprogram) that looks relevant in snippet
        SearchHit(
            url="https://job-boards.greenhouse.io/zind-erprogram/jobs/111",
            title="Junior Machine Learning & Computer Vision Engineer - India",
            snippet="Entry-level ML and computer vision engineer in Bengaluru, India (0-1 years experience).",
            provider_score=0.86,
        ),
        # Valid backup unindexed India CV/ML board from the SAME single Tavily search
        SearchHit(
            url="https://jobs.ashbyhq.com/visionindia/abc-123",
            title="Computer Vision Engineer I (Bengaluru, India)",
            snippet="0-1 years experience building PyTorch computer vision models in Bengaluru, India.",
            provider_score=0.84,
        ),
    ]

    class _HitClient:
        enabled = True
        name = "tavily"

        def __init__(self) -> None:
            self.calls = 0

        async def search_results(self, query: str, *, count: int = 15) -> list[SearchHit]:
            self.calls += 1
            return list(hits[:count])

        async def search_urls(self, query: str, *, count: int = 15) -> list[str]:
            return [h.url for h in await self.search_results(query, count=count)]

    async def fake_sync(url: str, target_store: Store) -> SyncSummary:
        if "neuraflash" in url:
            raise AssertionError("neuraflash should never be selected for a CV/ML engineering search")
        if "zind-erprogram" in url:
            raise RuntimeError("Greenhouse fetch failed: 404")
        if "visionindia" in url:
            job = enrich_job(Job(
                source_kind="ashby",
                source_key="visionindia",
                external_id="abc-123",
                title="Computer Vision Engineer I",
                company="visionindia",
                location="Bengaluru, India",
                country="India",
                description=(
                    "Design and train PyTorch computer vision models for inspection systems.\n"
                    "Requirements: 0-1 years of experience in computer vision and Python."
                ),
            ))
            return target_store.upsert_source_jobs("ashby", "visionindia", [job])
        raise AssertionError(f"Unexpected board synced: {url}")

    monkeypatch.setattr("boardforge.discovery.web.sync_url", fake_sync)
    query = (
        "Find entry-level computer vision and machine learning engineering jobs in India "
        "requiring up to one year of experience. If fewer than three suitable opportunities "
        "exist, discover additional job sources while preserving my location and experience constraints."
    )
    client = _HitClient()
    report = asyncio.run(discover_and_sync(
        store,
        CandidateProfile(),
        Preferences(),
        query=query,
        max_boards=1,
        client=client,
    ))

    # Strictly 1 Tavily search; neuraflash rejected; jumio skipped as existing;
    # zind-erprogram 404 handled transparently and fell through to visionindia!
    assert client.calls == 1
    assert report.searched_queries == 1
    assert report.skipped_existing_boards == 1
    assert report.skipped_low_relevance_boards == 1  # neuraflash deduplicated as 1 low-relevance board
    assert report.failed_boards == 1
    assert report.synced_boards == 1
    assert report.jobs_fetched == 1
    assert report.jobs_inserted == 1
    assert report.new_matching_jobs == 1
    assert report.new_recommended_jobs == 1
    assert "total_ms" in report.latency_ms

    # Subsequent discovery skips BOTH newly-synced visionindia AND recently-failed zind-erprogram (cooldown)!
    selected_again, skipped_existing_again, _ = prioritize_discovery_sources(
        hits,
        store=store,
        profile=CandidateProfile(),
        prefs=Preferences(),
        query=query,
        max_boards=2,
    )
    assert selected_again == []
    assert skipped_existing_again == 3  # jumio + zind-erprogram (failed cooldown) + visionindia
    # And never-synced failed 404 board is backed off in sources_due rather than spinning immediately
    assert store.sources_due(interval_seconds=3600, limit=10) == []


def test_repeated_identical_search_reuses_caches_and_exposes_observability(tmp_path: Path):
    store = Store(str(tmp_path / "cache_obs.db"))
    job_item = enrich_job(Job(
        source_kind="ashby",
        source_key="aiprise",
        external_id="swe-1",
        title="Software Engineer I (Bangalore, India)",
        company="aiprise",
        location="Bengaluru, India",
        country="India",
        description=AIPRISE_SWE1_DESCRIPTION,
    ))
    store.upsert_source_jobs("ashby", "aiprise", [job_item])

    emb_provider = _MockEmbeddingProvider()
    emb_client = EmbeddingClient(emb_provider, budget_store=store)
    segs = evidence_segments(job_item.description, max_chars=6000)
    design_seg = next(s["evidence_id"] for s in segs if "Design, develop, and ship" in s["text"])
    rel_provider = _MockRelevanceProvider([{
        "judgments": [{
            "candidate_ref": "C001",
            "relevance": "direct",
            "evidence_ids": [design_seg],
            "reasoning": "Direct backend and platform software engineering responsibilities.",
        }]
    }])
    verifier = RelevanceVerifier(store, rel_provider, enabled=True)
    core = JobAgent(store, embeddings=emb_client, relevance_verifier=verifier)
    tool_agent = ToolCallingAgent(
        core,
        provider=type("UnusedAgentLLM", (), {
            "enabled": True,
            "provider_name": "vertex-gemini",
            "model": "gemini-3.8-flash",
            "location": "global",
            "generate": staticmethod(lambda req: (_ for _ in ()).throw(AssertionError("Fast path must not call agent LLM"))),
        })(),
        executor=AgentToolExecutor(core, min_sufficient_results=1),
        enabled=True,
    )

    q = "Entry-level software engineering roles in India"
    first = asyncio.run(tool_agent.run("u", q, top_k=5))
    assert first["agent"]["orchestration_mode"] == "deterministic_fast_path"
    assert first["agent"]["provider_calls"] == 0
    first_ret = first["tool_results"][0]["result"]["retrieval"]
    assert first_ret["query_embedding_cache_hit"] is False
    assert first_ret["verification"]["provider_calls"] == 1
    assert first_ret["verification"]["cache_hits"] == 0
    assert first_ret["cost_usd"]["total"] == pytest.approx(0.000675)
    assert first["agent"]["cost_breakdown_usd"]["total_incremental_usd"] == pytest.approx(0.000675)
    assert set(first_ret["latency_ms"]) >= {
        "query_embedding", "database_retrieval", "ranking", "relevance_verification", "total_search"
    }

    # Second identical search reuses BOTH query_embeddings and relevance_verifications with 0 provider calls!
    second = asyncio.run(tool_agent.run("u", q, top_k=5))
    second_ret = second["tool_results"][0]["result"]["retrieval"]
    assert len(emb_provider.calls) == 1
    assert len(rel_provider.calls) == 1
    assert second_ret["query_embedding_cache_hit"] is True
    assert second_ret["verification"]["provider_calls"] == 0
    assert second_ret["verification"]["cache_hits"] == 1
    assert second_ret["cost_usd"]["total"] == 0.0
    assert second["agent"]["cost_breakdown_usd"]["total_incremental_usd"] == 0.0
    assert second["matches"][0]["status"] == "recommended"

    # Worker cycle also exposes per-stage latency observability
    worker_out = asyncio.run(run_worker_cycle(store, EmbeddingClient(type("Off", (), {"enabled": False, "model": "off", "dimensions": 4})())))
    assert set(worker_out["latency_ms"]) == {"discovery", "source_sync", "embedding_backfill", "total"}


def test_seed_sources_cli_dry_run_and_isolated_guarded_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seed_path = tmp_path / "curated_sources.json"
    seed_path.write_text(json.dumps({
        "sources": [
            {"url": "https://boards.greenhouse.io/exampleml/jobs/123", "company": "ExampleML"},
            {"url": "https://boards.greenhouse.io/exampleml", "company": "Duplicate"},
            {"url": "https://jobs.lever.co/indiavision", "company": "IndiaVision"},
            {"url": "https://untrusted.example.com/careers", "company": "InvalidHost"},
            {"url": "https://jobs.ashbyhq.com/sec?api_key=secret", "company": "SensitiveQuery"},
        ]
    }), encoding="utf-8")

    checked = validate_seed_sources(load_seed_file(seed_path))
    assert [item["key"] for item in checked["valid"]] == ["exampleml", "indiavision"]
    assert len(checked["rejected"]) == 3

    # CLI dry-run makes 0 network calls and requires no DB mutation
    isolated_db = tmp_path / "seed_cli_test.db"
    env = {
        **{k: v for k, v in __import__("os").environ.items() if k != "DATABASE_URL"},
        "BOARDFORGE_DB_PATH": str(isolated_db),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "boardforge.cli", "seed-sources", str(seed_path)],
        cwd=str(Path(__file__).parents[1]),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    dry_out = json.loads(proc.stdout)
    assert dry_out["mode"] == "dry_run"
    assert dry_out["network_calls"] == 0
    assert len(dry_out["valid"]) == 2
    assert not isolated_db.exists()

    # CLI --apply refuses to run without --expect-database or when DB mismatches
    refused = subprocess.run(
        [
            sys.executable, "-m", "boardforge.cli", "seed-sources", str(seed_path),
            "--apply", "--expect-database", "wrong_db_name",
        ],
        cwd=str(Path(__file__).parents[1]),
        env=env,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "does not match --expect-database" in refused.stderr

    # Isolated apply_seed_sources with mocked sync_url records provenance cleanly
    store = Store(str(isolated_db))

    async def fake_seed_sync(url: str, target_store: Store) -> SyncSummary:
        key = "exampleml" if "exampleml" in url else "indiavision"
        kind = "greenhouse" if "greenhouse" in url else "lever"
        return target_store.upsert_source_jobs(kind, key, [
            Job(
                source_kind=kind,
                source_key=key,
                external_id="1",
                title="Machine Learning Engineer",
                company=key,
                location="Bengaluru, India",
                country="India",
                description="0-1 years experience in Python and PyTorch.",
                min_experience_years=0.0,
                max_experience_years=1.0,
            )
        ])

    monkeypatch.setattr("boardforge.seeding.sync_url", fake_seed_sync)
    applied = asyncio.run(apply_seed_sources(store, checked["valid"], concurrency=2))
    assert all(r["status"] == "synced" and r["inserted"] == 1 for r in applied)
    sources = store.discovered_sources()
    assert {s["discovery_query"] for s in sources} == {"curated-seed"}


def test_register_indexed_sources_13_boards_dry_run_guards_and_freshness(tmp_path: Path):
    from datetime import datetime, timedelta, timezone

    # Exact identity matching prevents substring collisions across boardforge / boardforge_smoke / boardforge_test
    assert matches_expected_database("postgres:boardforge", "boardforge") is True
    assert matches_expected_database("postgres:boardforge", "postgres:boardforge") is True
    assert matches_expected_database("postgres:boardforge_smoke", "boardforge") is False
    assert matches_expected_database("postgres:boardforge_test", "boardforge") is False

    isolated_db = tmp_path / "thirteen_seeded_boards.db"
    store = Store(str(isolated_db))

    boards_13 = [
        ("ashby", "aiprise", "https://jobs.ashbyhq.com/aiprise/1/application"),
        ("ashby", "bolna", "https://jobs.ashbyhq.com/bolna/1/application"),
        ("ashby", "cartesia", "https://jobs.ashbyhq.com/cartesia/1/application"),
        ("ashby", "ema", "https://jobs.ashbyhq.com/ema/1/application"),
        # Coinbase uses a custom vanity apply_url in real data; canonical board URL must still resolve cleanly.
        ("greenhouse", "coinbase", "https://www.coinbase.com/careers/positions/6784618?gh_jid=6784618"),
        ("greenhouse", "jumio", "https://job-boards.greenhouse.io/jumio/jobs/1"),
        ("greenhouse", "razorpaysoftwareprivatelimited", "https://job-boards.greenhouse.io/razorpaysoftwareprivatelimited/jobs/1"),
        ("greenhouse", "twilio", "https://job-boards.greenhouse.io/twilio/jobs/1"),
        ("lever", "100ms", "https://jobs.lever.co/100ms/1"),
        ("lever", "acceldata", "https://jobs.lever.co/acceldata/1"),
        ("lever", "levelai", "https://jobs.lever.co/levelai/1"),
        ("lever", "safe", "https://jobs.lever.co/safe/1"),
        ("lever", "zimperium", "https://jobs.lever.co/zimperium/1"),
    ]
    for kind, key, apply_url in boards_13:
        store.upsert_source_jobs(kind, key, [
            Job(
                source_kind=kind,
                source_key=key,
                external_id="job-1",
                title="Software Engineer",
                company=key,
                location="Bengaluru, India",
                country="India",
                description="Build backend services.",
                apply_url=apply_url,
            )
        ])
    # Also insert a non-ATS test source to confirm it is never registered as a live ATS board
    store.upsert_source_jobs("test", "integration", [
        Job(
            source_kind="test",
            source_key="integration",
            external_id="t-1",
            title="Synthetic Engineer",
            company="Test",
            location="Bengaluru, India",
            country="India",
            description="Synthetic fixture.",
        )
    ])

    now = datetime.now(timezone.utc)
    stale_24h = (now - timedelta(hours=24)).isoformat()
    stale_2h = (now - timedelta(hours=2)).isoformat()
    fresh_10m = (now - timedelta(minutes=10)).isoformat()

    with store.connect() as conn:
        conn.execute("UPDATE jobs SET first_seen=?, last_seen=? WHERE source_kind != 'test'", (stale_24h, stale_24h))
        conn.execute("UPDATE sync_runs SET started_at=?, completed_at=? WHERE source_kind != 'test'", (stale_24h, stale_24h))
        conn.execute("UPDATE jobs SET first_seen=?, last_seen=? WHERE source_key='zimperium'", (stale_2h, stale_2h))
        conn.execute("UPDATE sync_runs SET started_at=?, completed_at=? WHERE source_key='zimperium'", (stale_2h, stale_2h))
        conn.execute("UPDATE jobs SET first_seen=?, last_seen=? WHERE source_key='bolna'", (fresh_10m, fresh_10m))
        conn.execute("UPDATE sync_runs SET started_at=?, completed_at=? WHERE source_key='bolna'", (fresh_10m, fresh_10m))

    assert store.discovered_sources() == []

    env = {
        **{k: v for k, v in __import__("os").environ.items() if k != "DATABASE_URL"},
        "BOARDFORGE_DB_PATH": str(isolated_db),
    }
    root_cwd = str(Path(__file__).parents[1])

    # 1. Default dry-run previews all 13 trusted boards without mutating discovered_sources
    dry_proc = subprocess.run(
        [sys.executable, "-m", "boardforge.cli", "register-indexed-sources", "--interval-seconds", "3600"],
        cwd=root_cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    dry_json = json.loads(dry_proc.stdout)
    assert dry_json["mode"] == "dry_run"
    assert dry_json["network_calls"] == 0
    assert dry_json["indexed_trusted_boards"] == 13
    assert dry_json["to_register_count"] == 13
    assert dry_json["registered_count"] == 0
    assert dry_json["due_for_maintenance_count"] == 12
    assert dry_json["fresh_within_interval_count"] == 1
    assert dry_json["skipped_unsupported_count"] == 1
    coinbase_row = next(r for r in dry_json["to_register"] if r["source_key"] == "coinbase")
    assert coinbase_row["board_url"] == "https://job-boards.greenhouse.io/coinbase"
    assert store.discovered_sources() == []

    # 2. --apply without --expect-database or with mismatched database fails closed
    no_guard = subprocess.run(
        [sys.executable, "-m", "boardforge.cli", "register-indexed-sources", "--apply"],
        cwd=root_cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert no_guard.returncode != 0
    assert "--expect-database" in no_guard.stderr
    assert store.discovered_sources() == []

    wrong_guard = subprocess.run(
        [
            sys.executable, "-m", "boardforge.cli", "register-indexed-sources",
            "--apply", "--expect-database", "boardforge",
        ],
        cwd=root_cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert wrong_guard.returncode != 0
    assert "does not match --expect-database" in wrong_guard.stderr
    assert store.discovered_sources() == []

    # 3. --apply with matching --expect-database registers all 13 boards and preserves historical freshness
    apply_proc = subprocess.run(
        [
            sys.executable, "-m", "boardforge.cli", "register-indexed-sources",
            "--apply", "--expect-database", isolated_db.name, "--interval-seconds", "3600",
        ],
        cwd=root_cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    applied_json = json.loads(apply_proc.stdout)
    assert applied_json["mode"] == "applied"
    assert applied_json["registered_count"] == 13
    assert applied_json["network_calls"] == 0

    registered = store.discovered_sources()
    assert len(registered) == 13
    by_key = {r["source_key"]: r for r in registered}
    assert by_key["bolna"]["last_synced"] == fresh_10m
    assert by_key["zimperium"]["last_synced"] == stale_2h
    assert by_key["coinbase"]["last_synced"] == stale_24h
    assert by_key["coinbase"]["board_url"] == "https://job-boards.greenhouse.io/coinbase"

    # Freshness verification: sources_due returns ONLY the 12 stale boards (24h boards before 2h zimperium),
    # and does NOT immediately re-sync the fresh 10m-old bolna board!
    due = store.sources_due(interval_seconds=3600, limit=25)
    due_keys = [r["source_key"] for r in due]
    assert len(due_keys) == 12
    assert "bolna" not in due_keys
    assert due_keys[-1] == "zimperium"

    # Discovery freshness also respects preserved timestamps: 10m-old bolna is skipped, 24h-old coinbase is stale
    selected, skipped_existing, _ = prioritize_discovery_sources(
        [
            SearchHit(
                url="https://jobs.ashbyhq.com/bolna/999",
                title="Machine Learning Engineer (Bengaluru, India)",
                snippet="0-1 years experience in PyTorch and ML in India.",
                provider_score=0.9,
            ),
            SearchHit(
                url="https://job-boards.greenhouse.io/coinbase/jobs/999",
                title="Machine Learning Engineer (Bengaluru, India)",
                snippet="0-1 years experience in PyTorch and ML in India.",
                provider_score=0.9,
            ),
        ],
        store=store,
        profile=CandidateProfile(),
        prefs=Preferences(roles=["AI/ML"], countries=["India"], max_required_experience_years=1.0),
        query="entry-level machine learning engineer India",
        max_boards=4,
        fresh_source_seconds=3600,
    )
    assert skipped_existing == 1
    assert [s.key for s in selected] == ["coinbase"]

    # 4. Idempotent re-run registers 0 new boards and preserves existing timestamps
    second = register_indexed_sources(store, apply=True, interval_seconds=3600)
    assert second["registered_count"] == 0
    assert second["already_registered_count"] == 13


def test_register_indexed_sources_postgres_test_db_freshness_and_identity_guard():
    import os
    from datetime import datetime, timedelta, timezone
    from boardforge.postgres_store import PostgresStore
    from tests.test_postgres_integration import _assert_disposable_database

    pg_url = os.getenv("BOARDFORGE_TEST_POSTGRES_URL")
    if not pg_url:
        pytest.skip("set BOARDFORGE_TEST_POSTGRES_URL for PostgreSQL test")
    _assert_disposable_database(pg_url)

    store = PostgresStore(pg_url, min_size=1, max_size=2)
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

        store.upsert_source_jobs("greenhouse", "coinbase", [
            Job(
                source_kind="greenhouse",
                source_key="coinbase",
                external_id="cb-1",
                title="Software Engineer",
                company="coinbase",
                location="Bengaluru, India",
                country="India",
                description="Build distributed services.",
                apply_url="https://www.coinbase.com/careers/positions/1",
            )
        ])
        store.upsert_source_jobs("ashby", "aiprise", [
            Job(
                source_kind="ashby",
                source_key="aiprise",
                external_id="ap-1",
                title="Software Engineer I",
                company="aiprise",
                location="Bengaluru, India",
                country="India",
                description="Build backend APIs.",
                apply_url="https://jobs.ashbyhq.com/aiprise/1/application",
            )
        ])
        store.upsert_source_jobs("test", "fixture", [
            Job(
                source_kind="test",
                source_key="fixture",
                external_id="fx-1",
                title="Test Job",
                company="fixture",
                location="Bengaluru, India",
                country="India",
                description="Synthetic row.",
            )
        ])

        now = datetime.now(timezone.utc)
        stale_ts = now - timedelta(hours=3)
        fresh_ts = now - timedelta(minutes=5)
        with store.connect() as conn:
            conn.execute("UPDATE jobs SET first_seen=%s, last_seen=%s WHERE source_key='coinbase'", (stale_ts, stale_ts))
            conn.execute("UPDATE sync_runs SET started_at=%s, completed_at=%s WHERE source_key='coinbase'", (stale_ts, stale_ts))
            conn.execute("UPDATE jobs SET first_seen=%s, last_seen=%s WHERE source_key='aiprise'", (fresh_ts, fresh_ts))
            conn.execute("UPDATE sync_runs SET started_at=%s, completed_at=%s WHERE source_key='aiprise'", (fresh_ts, fresh_ts))

        env = {**os.environ, "DATABASE_URL": pg_url}
        root_cwd = str(Path(__file__).parents[1])

        # Exact-match guard: passing --expect-database boardforge MUST fail against boardforge_test!
        denied = subprocess.run(
            [
                sys.executable, "-m", "boardforge.cli", "register-indexed-sources",
                "--apply", "--expect-database", "boardforge",
            ],
            cwd=root_cwd,
            env=env,
            capture_output=True,
            text=True,
        )
        assert denied.returncode != 0
        assert "does not match --expect-database" in denied.stderr
        assert store.discovered_sources() == []

        # Matching --expect-database postgres:boardforge_test succeeds
        ok_proc = subprocess.run(
            [
                sys.executable, "-m", "boardforge.cli", "register-indexed-sources",
                "--apply", "--expect-database", "postgres:boardforge_test",
                "--interval-seconds", "3600",
            ],
            cwd=root_cwd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(ok_proc.stdout)
        assert payload["database"] == "postgres:boardforge_test"
        assert payload["registered_count"] == 2
        assert payload["skipped_unsupported_count"] == 1

        due = store.sources_due(interval_seconds=3600, limit=10)
        assert [r["source_key"] for r in due] == ["coinbase"]
    finally:
        store.close()
