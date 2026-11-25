import asyncio
import pytest
from boardforge.connectors.detect import detect_source
from boardforge.discovery import build_discovery_query, canonical_sources
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


def test_discovery_query_uses_resume_roles_and_geography():
    profile = CandidateProfile(
        role_signals=["AI/ML", "Backend"],
        skills=["Python", "PyTorch", "FastAPI"],
    )
    prefs = Preferences(countries=["India"])
    query = build_discovery_query(profile, prefs)
    assert "India" in query
    assert "machine learning engineer" in query
    assert "backend engineer" in query


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


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {"results": []}

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, capture, response, **kwargs):
        self.capture = capture
        self.response = response
        self.capture["client_kwargs"] = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        self.capture["method"] = "POST"
        self.capture["url"] = url
        self.capture["request"] = kwargs
        return self.response


def test_tavily_http_contract(monkeypatch):
    from boardforge.discovery.web import TAVILY_DOMAINS, TavilyDiscoveryClient

    capture = {}
    response = _FakeResponse(payload={
        "results": [
            {"url": "https://job-boards.greenhouse.io/acme/jobs/123"},
            {"url": "https://jobs.ashbyhq.com/demo/abc"},
        ]
    })
    monkeypatch.setattr(
        "boardforge.discovery.web.httpx.AsyncClient",
        lambda **kwargs: _FakeAsyncClient(capture, response, **kwargs),
    )

    class Budget:
        def __init__(self):
            self.reserved = self.finalized = 0
        def reserve_llm_usage(self, *args, **kwargs):
            self.reserved += 1
            return {"allowed": True}
        def finalize_llm_usage(self, *args, **kwargs):
            self.finalized += 1

    budget = Budget()
    urls = asyncio.run(TavilyDiscoveryClient(
        "test-key", budget_store=budget
    ).search_urls("backend engineer India", count=7))

    assert capture["method"] == "POST"
    assert capture["url"] == "https://api.tavily.com/search"
    assert capture["request"]["headers"]["Authorization"] == "Bearer test-key"
    body = capture["request"]["json"]
    assert body["search_depth"] == "basic"
    assert body["max_results"] == 7
    assert body["include_domains"] == list(TAVILY_DOMAINS)
    assert urls == [
        "https://job-boards.greenhouse.io/acme/jobs/123",
        "https://jobs.ashbyhq.com/demo/abc",
    ]
    assert (budget.reserved, budget.finalized) == (1, 1)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, "SearchProviderAuthError"),
        (403, "SearchProviderAuthError"),
        (429, "SearchProviderRateLimitError"),
        (500, "SearchProviderTransientError"),
        (400, "SearchProviderRequestError"),
    ],
)
def test_tavily_http_error_classification(monkeypatch, status_code, expected):
    from boardforge.discovery.web import TavilyDiscoveryClient

    capture = {}
    response = _FakeResponse(status_code=status_code)
    monkeypatch.setattr(
        "boardforge.discovery.web.httpx.AsyncClient",
        lambda **kwargs: _FakeAsyncClient(capture, response, **kwargs),
    )

    with pytest.raises(Exception) as exc:
        asyncio.run(TavilyDiscoveryClient("test-key").search_urls("backend", count=5))
    assert type(exc.value).__name__ == expected
