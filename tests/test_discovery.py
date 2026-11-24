from boardforge.connectors.detect import detect_source
from boardforge.discovery import build_discovery_queries, canonical_sources
from boardforge.models import CandidateProfile, Preferences


def test_greenhouse_job_url_canonicalizes_to_board():
    source = detect_source("https://job-boards.greenhouse.io/acme/jobs/123456")
    assert source.kind == "greenhouse"
    assert source.key == "acme"
    assert source.board_url == "https://job-boards.greenhouse.io/acme"


def test_lever_job_url_canonicalizes_to_board():
    source = detect_source("https://jobs.lever.co/acme/abc-123")
    assert source.kind == "lever"
    assert source.key == "acme"
    assert source.board_url == "https://jobs.lever.co/acme"


def test_ashby_job_url_canonicalizes_to_board():
    source = detect_source("https://jobs.ashbyhq.com/acme/abc-123")
    assert source.kind == "ashby"
    assert source.key == "acme"
    assert source.board_url == "https://jobs.ashbyhq.com/acme"


def test_discovery_queries_use_resume_roles_and_geography():
    profile = CandidateProfile(
        role_signals=["AI/ML", "Backend"],
        skills=["Python", "PyTorch", "FastAPI"],
    )
    prefs = Preferences(countries=["India"])
    queries = build_discovery_queries(profile, prefs)
    assert len(queries) == 3
    assert all("India" in query for query in queries)
    assert any("machine learning engineer" in query for query in queries)
    assert any("backend engineer" in query for query in queries)


def test_canonical_sources_deduplicate_same_board():
    sources = canonical_sources([
        "https://job-boards.greenhouse.io/acme/jobs/123",
        "https://job-boards.greenhouse.io/acme/jobs/456",
        "https://jobs.lever.co/other/abc",
        "https://example.com/jobs/ignored",
    ])
    assert [(source.kind, source.key) for source in sources] == [
        ("greenhouse", "acme"),
        ("lever", "other"),
    ]
