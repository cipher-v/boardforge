from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from boardforge.discovery import (
    SearchHit,
    TavilyDiscoveryClient,
    build_discovery_query,
    discover_and_sync,
    is_individual_posting_url,
    prioritize_discovery_sources,
)
from boardforge.enrich import enrich_job
from boardforge.models import CandidateProfile, Job, Preferences, SyncSummary
from boardforge.store import Store


LIVE_CV_ML_QUERY = (
    "Find entry-level computer vision and machine learning engineering jobs "
    "in India requiring up to one year of experience."
)


class _MockRichTavilyClient:
    enabled = True
    name = "tavily"

    def __init__(self, hits: list[SearchHit]):
        self.hits = list(hits)
        self.queries: list[tuple[str, int]] = []

    async def search_results(self, query: str, *, count: int = 15) -> list[SearchHit]:
        self.queries.append((query, count))
        return list(self.hits[:count])

    async def search_urls(self, query: str, *, count: int = 15) -> list[str]:
        return [h.url for h in await self.search_results(query, count=count)]


def test_discovery_query_targets_individual_postings_domain_geography_and_seniority():
    q = build_discovery_query(CandidateProfile(), Preferences(), LIVE_CV_ML_QUERY)
    low = q.casefold()

    # Targets concrete role titles, India + major Indian tech hubs, and junior/entry-level cues
    assert '"computer vision engineer"' in low
    assert '"machine learning engineer"' in low
    assert '"india"' in low
    assert '"bengaluru"' in low
    assert '"entry level"' in low or '"junior"' in low

    # Strips conversational search-engine noise
    assert "find entry-level" not in low
    assert "requiring up to" not in low

    # Distinguishes individual posting URLs from generic career board roots
    assert is_individual_posting_url("https://jobs.ashbyhq.com/pixxel/9f8e7d6c-1234") is True
    assert is_individual_posting_url("https://jobs.ashbyhq.com/dandy") is False
    assert is_individual_posting_url("https://job-boards.greenhouse.io/jumio/jobs/4740028") is True
    assert is_individual_posting_url("https://job-boards.greenhouse.io/jumio") is False
    assert is_individual_posting_url("https://jobs.lever.co/sarvam/abc-def-123") is True
    assert is_individual_posting_url("https://jobs.lever.co/sarvam") is False


def test_discovery_prioritizes_unindexed_india_postings_and_skips_dandy_and_jumio(
    monkeypatch, tmp_path: Path
):
    store = Store(str(tmp_path / "disc_rel.db"))
    # Pre-index Jumio in the store (simulating already-indexed 17 jobs with 0 entry-level matches).
    store.upsert_source_jobs("greenhouse", "jumio", [
        enrich_job(Job(
            source_kind="greenhouse",
            source_key="jumio",
            external_id="4665626",
            title="Senior Machine Learning Engineer - Biometrics",
            company="jumio",
            location="Bengaluru, India",
            description="5+ years of experience in computer vision and biometrics.",
        )),
    ])

    hits = [
        # 1. Generic US career board root (Dandy) -> must NOT be synced even with spare board budget
        SearchHit(
            url="https://jobs.ashbyhq.com/dandy",
            title="Jobs at Dandy",
            snippet="Explore open engineering and dental operations roles in New York, NY and Remote (US).",
            provider_score=0.82,
        ),
        # 2. US-only individual CV posting (Dandy) -> foreign geo + Senior + 5+ yrs -> rejected
        SearchHit(
            url="https://jobs.ashbyhq.com/dandy/us-cv-1",
            title="Senior Computer Vision Engineer - New York, NY",
            snippet="Location: New York, NY (Hybrid). 5+ years of experience in 3D computer vision.",
            provider_score=0.80,
        ),
        # 3. Already-indexed recently synchronized board (Jumio) -> skipped so slot is not wasted
        SearchHit(
            url="https://job-boards.greenhouse.io/jumio/jobs/4749999",
            title="Computer Vision Engineer I - Bengaluru, India",
            snippet="Bengaluru, India. 0-1 years of experience in computer vision and deep learning.",
            provider_score=0.88,
        ),
        # 4. India Senior-only posting on an unindexed board -> rejected on entry-level search
        SearchHit(
            url="https://jobs.lever.co/senioronlyco/sr-ml-99",
            title="Staff Machine Learning Engineer - Bengaluru, India",
            snippet="Bengaluru, India. Requirements: 8+ years of machine learning experience.",
            provider_score=0.79,
        ),
        # 5. Relevant unindexed India Junior Computer Vision posting -> MUST be selected!
        SearchHit(
            url="https://jobs.ashbyhq.com/pixxel/jr-cv-01",
            title="Computer Vision Engineer I - Bengaluru",
            snippet="Location: Bengaluru, India. MUST-HAVE: 0-1 years of experience in computer vision and PyTorch.",
            provider_score=0.76,
        ),
        # 6. Relevant unindexed India Entry-Level ML posting -> MUST be selected!
        SearchHit(
            url="https://jobs.lever.co/sarvamai/ml-grad-02",
            title="Machine Learning Engineer (New Grad) - Bangalore, India",
            snippet="Bangalore, India. Entry-level ML role building vision and multimodal models in PyTorch.",
            provider_score=0.74,
        ),
    ]
    client = _MockRichTavilyClient(hits)
    synced_urls: list[str] = []

    async def fake_sync_url(url: str, target_store) -> SyncSummary:
        synced_urls.append(url)
        if "pixxel" in url:
            jobs = [
                # 1 eligible entry-level India CV job
                enrich_job(Job(
                    source_kind="ashby",
                    source_key="pixxel",
                    external_id="jr-cv-01",
                    title="Computer Vision Engineer I",
                    company="pixxel",
                    location="Bengaluru, India",
                    description="Build satellite computer vision models in Python and PyTorch. 0-1 years of experience.",
                    apply_url="https://jobs.ashbyhq.com/pixxel/jr-cv-01",
                )),
                # 1 US job on same board (fetched & indexed, but NOT a match for India!)
                enrich_job(Job(
                    source_kind="ashby",
                    source_key="pixxel",
                    external_id="us-sales-02",
                    title="Account Executive",
                    company="pixxel",
                    location="Los Angeles, CA",
                    description="Enterprise satellite imagery sales. 5+ years of sales experience.",
                    apply_url="https://jobs.ashbyhq.com/pixxel/us-sales-02",
                )),
                # 1 Senior India job on same board (fetched & indexed, but NOT an entry-level match!)
                enrich_job(Job(
                    source_kind="ashby",
                    source_key="pixxel",
                    external_id="sr-cv-03",
                    title="Senior Computer Vision Engineer",
                    company="pixxel",
                    location="Bengaluru, India",
                    description="Lead hyperspectral vision pipelines. 6+ years of experience.",
                    apply_url="https://jobs.ashbyhq.com/pixxel/sr-cv-03",
                )),
            ]
            return target_store.upsert_source_jobs("ashby", "pixxel", jobs)

        jobs = [
            enrich_job(Job(
                source_kind="lever",
                source_key="sarvamai",
                external_id="ml-grad-02",
                title="Machine Learning Engineer",
                company="sarvamai",
                location="Bengaluru, India",
                description="Train machine learning and computer vision models. 0-1 years of experience.",
                apply_url="https://jobs.lever.co/sarvamai/ml-grad-02",
            )),
        ]
        return target_store.upsert_source_jobs("lever", "sarvamai", jobs)

    monkeypatch.setattr("boardforge.discovery.web.sync_url", fake_sync_url)

    # Even with max_boards=6, ONLY pixxel and sarvamai are synced — Dandy, Jumio, and SeniorOnlyCo are NOT!
    report = asyncio.run(
        discover_and_sync(
            store,
            CandidateProfile(),
            Preferences(),
            query=LIVE_CV_ML_QUERY,
            max_boards=6,
            client=client,
        )
    )

    assert len(client.queries) == 1, "Must preserve single-search Tavily budget"
    assert synced_urls == [
        "https://jobs.ashbyhq.com/pixxel",
        "https://jobs.lever.co/sarvamai",
    ]
    assert report.discovered_boards == 2
    assert report.synced_boards == 2
    assert report.skipped_existing_boards == 1  # Jumio
    assert report.skipped_low_relevance_boards == 2  # Dandy + SeniorOnlyCo

    # Fetched jobs (4 total across boards) are reported separately from newly indexed (4)
    # and newly surfaced matching results (only the 2 eligible India entry-level CV/ML jobs!).
    assert report.jobs_fetched == 4
    assert report.jobs_inserted == 4
    assert report.new_matching_jobs == 2
    assert report.new_recommended_jobs == 2


def test_discovery_does_not_force_sync_when_only_dandy_and_jumio_exist(
    monkeypatch, tmp_path: Path
):
    store = Store(str(tmp_path / "no_force.db"))
    store.upsert_source_jobs("greenhouse", "jumio", [
        enrich_job(Job(
            source_kind="greenhouse",
            source_key="jumio",
            external_id="1",
            title="Senior ML Engineer",
            company="jumio",
            location="Bengaluru, India",
            description="5+ years experience.",
        )),
    ])
    client = _MockRichTavilyClient([
        SearchHit(
            url="https://jobs.ashbyhq.com/dandy",
            title="Dandy Careers",
            snippet="75 open roles in New York and Remote across the United States.",
        ),
        SearchHit(
            url="https://job-boards.greenhouse.io/jumio/jobs/1",
            title="Machine Learning Engineer - Bengaluru, India",
            snippet="Bengaluru, India. 0-1 years of experience.",
        ),
    ])

    async def fail_if_synced(url: str, target_store):
        raise AssertionError(f"Should not have synchronized {url}")

    monkeypatch.setattr("boardforge.discovery.web.sync_url", fail_if_synced)

    report = asyncio.run(
        discover_and_sync(
            store,
            CandidateProfile(),
            Preferences(),
            query=LIVE_CV_ML_QUERY,
            max_boards=6,
            client=client,
        )
    )
    assert report.discovered_boards == 0
    assert report.synced_boards == 0
    assert report.jobs_fetched == 0
    assert report.jobs_inserted == 0
    assert report.new_matching_jobs == 0
    assert report.skipped_existing_boards == 1
    assert report.skipped_low_relevance_boards == 1
    assert "already-indexed" in report.reason


def test_unindexed_boards_beat_stale_indexed_and_explicit_board_name_overrides_skip(
    tmp_path: Path,
):
    store = Store(str(tmp_path / "stale_vs_new.db"))
    stale_time = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    with store.connect() as conn:
        conn.execute(
            """INSERT INTO discovered_sources
               (source_kind,source_key,board_url,discovery_query,first_discovered,
                last_discovered,last_synced,last_status)
               VALUES('greenhouse','staleindia','https://job-boards.greenhouse.io/staleindia',
                      'manual:old',?,?,?,'synced')""",
            (stale_time, stale_time, stale_time),
        )
    store.record_discovered_source(
        "greenhouse", "freshindia", "https://job-boards.greenhouse.io/freshindia",
        "manual:recent", synced=True, status="synced",
    )

    hits = [
        SearchHit(
            url="https://job-boards.greenhouse.io/staleindia/jobs/10",
            title="Machine Learning Engineer I - Bengaluru, India",
            snippet="Bengaluru, India. 0-1 years of experience in PyTorch.",
            provider_score=0.90,
        ),
        SearchHit(
            url="https://jobs.lever.co/brandnewindia/job-20",
            title="Machine Learning Engineer I - Bengaluru, India",
            snippet="Bengaluru, India. 0-1 years of experience in PyTorch.",
            provider_score=0.80,
        ),
        SearchHit(
            url="https://job-boards.greenhouse.io/freshindia/jobs/30",
            title="Machine Learning Engineer I - Bengaluru, India",
            snippet="Bengaluru, India. 0-1 years of experience in PyTorch.",
            provider_score=0.95,
        ),
    ]

    # With max_boards=1, unindexed 'brandnewindia' beats stale 'staleindia', while recent 'freshindia' is skipped.
    selected, skipped_existing, _ = prioritize_discovery_sources(
        hits,
        store=store,
        profile=CandidateProfile(),
        prefs=Preferences(),
        query=LIVE_CV_ML_QUERY,
        max_boards=1,
    )
    assert [(s.kind, s.key) for s in selected] == [("lever", "brandnewindia")]
    assert skipped_existing == 1

    # Explicitly naming 'freshindia' in the query permits re-synchronizing that board.
    explicit_selected, explicit_skipped, _ = prioritize_discovery_sources(
        [hits[2]],
        store=store,
        profile=CandidateProfile(),
        prefs=Preferences(),
        query="Find entry-level machine learning jobs at freshindia in India",
        max_boards=1,
    )
    assert [(s.kind, s.key) for s in explicit_selected] == [("greenhouse", "freshindia")]
    assert explicit_skipped == 0


def test_tavily_client_search_results_parses_title_snippet_and_score(monkeypatch):
    class _Resp:
        status_code = 200

        def json(self):
            return {
                "results": [
                    {
                        "url": "https://jobs.ashbyhq.com/pixxel/jr-cv-01",
                        "title": "Computer Vision Engineer I - Bengaluru",
                        "content": "Bengaluru, India · 0-1 years of PyTorch experience.",
                        "score": 0.87,
                    }
                ]
            }

    class _Client:
        def __init__(self, **kwargs):
            del kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, **kwargs):
            del url, kwargs
            return _Resp()

    monkeypatch.setattr("boardforge.discovery.web.httpx.AsyncClient", _Client)
    hits = asyncio.run(TavilyDiscoveryClient("test-key").search_results("cv india", count=5))
    assert len(hits) == 1
    assert hits[0].url == "https://jobs.ashbyhq.com/pixxel/jr-cv-01"
    assert hits[0].title == "Computer Vision Engineer I - Bengaluru"
    assert "Bengaluru, India" in hits[0].snippet
    assert hits[0].provider_score == 0.87
