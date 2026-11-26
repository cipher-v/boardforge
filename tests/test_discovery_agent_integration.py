from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

import boardforge.api as api_module
import boardforge.discovery.service as discovery_service
from boardforge.agent import JobAgent
from boardforge.agent_tools import AgentExecutionState, AgentToolExecutor
from boardforge.discovery.service import queue_and_dispatch_discovery
from boardforge.enrich import enrich_job
from boardforge.intelligent_agent import ToolCallingAgent
from boardforge.llm import LLMCallResult, TokenUsage
from boardforge.models import CandidateProfile, Job, Preferences, SyncSummary
from boardforge.planning import tool_names
from boardforge.preferences import LLMPreferenceExtractor, deterministic_extract
from boardforge.store import Store


ROOT = Path(__file__).parents[1]
UI_PATH = ROOT / "boardforge" / "static" / "index.html"


def _chrome() -> str | None:
    for candidate in (
        shutil.which("chrome"),
        shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


class _DeterministicExtractor:
    async def extract(self, text: str, current: Preferences | None = None) -> Preferences:
        return deterministic_extract(text, current)


class _DisabledEmbeddings:
    enabled = False
    model = "disabled"


class _ScriptedReasoningProvider:
    enabled = True
    provider_name = "vertex-gemini"
    model = "gemini-3.8-flash"
    location = "global"
    input_usd_per_million = 0.75
    output_usd_per_million = 3.75

    def __init__(self, steps: list[dict]):
        self.steps = list(steps)
        self.calls = 0

    async def generate(self, request):
        del request
        self.calls += 1
        step = self.steps.pop(0) if self.steps else {"function_calls": []}
        return LLMCallResult(
            data=step,
            usage=TokenUsage(input_tokens=120, output_tokens=24, total_tokens=144),
            provider=self.provider_name,
            model=self.model,
            estimated_cost_usd=0.00018,
        )


class _MockTavilyProvider:
    enabled = True
    name = "tavily"

    def __init__(self, urls: list[str]):
        self.urls = list(urls)
        self.calls: list[tuple[str, int]] = []

    async def search_urls(self, query: str, *, count: int = 15) -> list[str]:
        self.calls.append((query, count))
        return list(self.urls)


def test_ordinary_unsuccessful_searches_do_not_spend_discovery_or_llm_resources(tmp_path: Path):
    store = Store(str(tmp_path / "ordinary.db"))
    core = JobAgent(store, extractor=_DeterministicExtractor(), embeddings=_DisabledEmbeddings())
    discovery_client = _MockTavilyProvider(["https://job-boards.greenhouse.io/acme/jobs/1"])
    executor = AgentToolExecutor(core, min_sufficient_results=3, discovery_client=discovery_client)
    provider = _ScriptedReasoningProvider([
        {"function_calls": [{"name": "discover_job_sources", "args": {"query": "rust", "reason": "empty"}}]}
    ])
    agent = ToolCallingAgent(core, provider=provider, executor=executor, enabled=True)

    for query in (
        "Find Rust compiler roles in Bengaluru",
        "Software engineering jobs for candidates with no formal employment experience",
        "Changed my mind, show me backend jobs in Pune",
    ):
        res = asyncio.run(agent.run("u", query))
        assert res["agent"]["orchestration_mode"] == "deterministic_fast_path"
        assert res["agent"]["provider_calls"] == 0
        assert res["agent"]["discovery_requested"] is False
        assert res["agent"]["discovery_request_id"] is None
        assert res["discovery"] is None
        assert res["matches"] == []

    assert provider.calls == 0
    assert discovery_client.calls == []
    with store.connect() as conn:
        assert conn.execute("SELECT count(*) FROM discovery_requests").fetchone()[0] == 0


def test_explicit_conditional_agent_discovery_exposes_request_id_and_stops_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("BOARDFORGE_DISCOVERY_DISPATCH_MODE", "none")
    store = Store(str(tmp_path / "cond_disc.db"))
    core = JobAgent(store, extractor=_DeterministicExtractor(), embeddings=_DisabledEmbeddings())
    discovery_client = _MockTavilyProvider(["https://job-boards.greenhouse.io/acme/jobs/1"])
    executor = AgentToolExecutor(core, min_sufficient_results=2, discovery_client=discovery_client)
    provider = _ScriptedReasoningProvider([
        {
            "function_calls": [{
                "name": "search_indexed_jobs",
                "args": {
                    "query": "rust compiler engineer India",
                    "top_k": 10,
                    "mandatory_constraints": ["India"],
                    "soft_preferences": [],
                    "conditional_alternatives": [],
                },
            }]
        },
        {
            "function_calls": [{
                "name": "discover_job_sources",
                "args": {
                    "query": "rust compiler engineer India",
                    "reason": "zero indexed recommendations",
                },
            }]
        },
        # If the agent wasted a 3rd LLM call after queueing discovery, provider.calls would be 3.
        {"function_calls": []},
    ])
    agent = ToolCallingAgent(core, provider=provider, executor=executor, enabled=True)

    prompt = "Find Rust compiler roles in India; if indexed matches are scarce, discover new job sources."
    result = asyncio.run(agent.run("u", prompt))

    assert result["agent"]["orchestration_mode"] == "llm_conditional"
    assert provider.calls == 2, "Agent must not spend an extra reasoning call after queueing discovery"
    assert result["agent"]["stopped_reason"] == "discovery_queued"
    assert result["agent"]["discovery_requested"] is True
    assert result["agent"]["discovery_request_id"] == 1
    assert result["discovery"] is not None
    assert result["discovery"]["request_id"] == 1
    assert result["discovery"]["status"] == "queued"
    assert "queued" in result["message"].lower()


def test_queue_to_worker_to_ingestion_to_subsequent_search_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("BOARDFORGE_DISCOVERY_DISPATCH_MODE", "local")
    store = Store(str(tmp_path / "e2e_disc.db"))
    core = JobAgent(store, extractor=_DeterministicExtractor(), embeddings=_DisabledEmbeddings())

    discovery_client = _MockTavilyProvider([
        "https://job-boards.greenhouse.io/acmeai/jobs/101",
        "https://job-boards.greenhouse.io/acmeai/jobs/102",  # same board -> deduplicated!
        "https://jobs.lever.co/punecloud/abc-1",
        "https://untrusted.example.com/jobs/999",  # non-ATS -> ignored!
    ])
    synced_boards: list[str] = []

    async def fake_sync_url(url: str, target_store) -> SyncSummary:
        synced_boards.append(url)
        if "greenhouse" in url:
            jobs = [enrich_job(Job(
                source_kind="greenhouse",
                source_key="acmeai",
                external_id="101",
                title="Backend Engineer",
                company="AcmeAI",
                location="Pune, India",
                description="Build distributed backend services and Python APIs.",
                apply_url="https://job-boards.greenhouse.io/acmeai/jobs/101",
            ))]
            return target_store.upsert_source_jobs("greenhouse", "acmeai", jobs)
        jobs = [enrich_job(Job(
            source_kind="lever",
            source_key="punecloud",
            external_id="abc-1",
            title="Platform Backend Engineer",
            company="PuneCloud",
            location="Pune, India",
            description="Design scalable Python backend microservices.",
            apply_url="https://jobs.lever.co/punecloud/abc-1",
        ))]
        return target_store.upsert_source_jobs("lever", "punecloud", jobs)

    monkeypatch.setattr("boardforge.discovery.web.sync_url", fake_sync_url)

    async def scenario():
        # 1. Initial search before discovery finds 0 jobs.
        before = await core.search_jobs("u", "Find Python backend roles in Pune", 10)
        assert before.data["matches"] == []

        # 2. Enqueue & dispatch via the unified helper used by both /discover and /agent.
        queued = await queue_and_dispatch_discovery(
            store,
            "u",
            CandidateProfile(skills=["Python"], role_signals=["Backend"]),
            Preferences(countries=["India"], locations=["Pune"]),
            query="Python backend roles in Pune",
            client=discovery_client,
        )
        req_id = queued["request_id"]
        assert req_id == 1
        assert queued["status"] == "queued"
        assert queued["dispatch"]["mode"] == "local"
        assert queued["dispatch"]["dispatched"] is True

        # Wait for the coalesced in-process worker drain to finish.
        for _ in range(50):
            rec = store.discovery_request(req_id)
            if rec and rec["status"] == "completed":
                break
            await asyncio.sleep(0.02)
        rec = store.discovery_request(req_id)
        assert rec is not None and rec["status"] == "completed"
        assert rec["report"]["discovered_boards"] == 2
        assert rec["report"]["synced_boards"] == 2
        assert rec["report"]["jobs_fetched"] == 2
        assert len(discovery_client.calls) == 1
        assert len(synced_boards) == 2

        # 3. Duplicate submission within TTL reuses the completed request without re-querying Tavily.
        dup = await queue_and_dispatch_discovery(
            store,
            "u",
            CandidateProfile(skills=["Python"], role_signals=["Backend"]),
            Preferences(countries=["India"], locations=["Pune"]),
            query="Python backend roles in Pune",
            client=discovery_client,
        )
        assert dup["request_id"] == req_id
        assert dup["cached"] is True
        assert dup["status"] == "completed"
        assert len(discovery_client.calls) == 1

        # 4. Subsequent search retrieves and recommends the newly ingested jobs.
        after = await core.search_jobs("u", "Find Python backend roles in Pune", 10)
        titles = [m["job"]["title"] for m in after.data["matches"]]
        assert "Backend Engineer" in titles
        assert "Platform Backend Engineer" in titles
        assert any(m["status"] == "recommended" for m in after.data["matches"])

    asyncio.run(scenario())


def test_discovery_ownership_and_cloud_run_job_dispatch_coalescing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    store = Store(str(tmp_path / "auth_disc.db"))
    core = JobAgent(store, extractor=_DeterministicExtractor(), embeddings=_DisabledEmbeddings())
    tool = ToolCallingAgent(core, enabled=False)
    monkeypatch.setattr(api_module, "store", store)
    monkeypatch.setattr(api_module, "core_agent", core)
    monkeypatch.setattr(api_module, "tool_agent", tool)
    monkeypatch.setenv("BOARDFORGE_PROFILE_TOKENS", json.dumps({
        "token-a": "profile-a",
        "token-b": "profile-b",
    }))
    monkeypatch.setenv("BOARDFORGE_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-secret-not-called")
    monkeypatch.setenv("BOARDFORGE_DISCOVERY_DISPATCH_MODE", "cloud_run_job")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-proj")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("BOARDFORGE_MAINTENANCE_JOB", "boardforge-maintenance")
    monkeypatch.setenv("BOARDFORGE_JOB_DISPATCH_COOLDOWN_SECONDS", "60")
    monkeypatch.setattr(discovery_service, "_LAST_CLOUD_DISPATCH_MONOTONIC", 0.0)

    cloud_calls: list[str] = []

    class _FakeResp:
        status_code = 200

    class _FakeHttpClient:
        def __init__(self, **kwargs):
            del kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, **kwargs):
            assert kwargs["headers"]["Authorization"] == "Bearer fake-adc-token"
            cloud_calls.append(url)
            return _FakeResp()

    monkeypatch.setattr(discovery_service, "_cloud_run_access_token", lambda: "fake-adc-token")
    monkeypatch.setattr(discovery_service.httpx, "AsyncClient", _FakeHttpClient)

    client = TestClient(api_module.app)
    headers_a = {"Authorization": "Bearer token-a"}
    headers_b = {"Authorization": "Bearer token-b"}

    # Unauthenticated & cross-profile POST /discover/{id} rejected
    assert client.post("/discover/profile-a", json={"profile_id": "profile-a", "query": "backend"}).status_code == 401
    assert client.post("/discover/profile-a", headers=headers_b, json={"profile_id": "profile-b", "query": "backend"}).status_code == 403

    # Authorized POST /discover/profile-a queues and triggers Cloud Run Job :run once
    r1 = client.post("/discover/profile-a", headers=headers_a, json={"profile_id": "profile-b", "query": "backend"})
    assert r1.status_code == 200
    disc1 = r1.json()["discovery"]
    req_id = disc1["request_id"]
    assert disc1["status"] == "queued"
    assert disc1["dispatch"] == {"mode": "cloud_run_job", "dispatched": True, "job": "boardforge-maintenance"}
    assert cloud_calls == [
        "https://run.googleapis.com/v2/projects/demo-proj/locations/us-central1/jobs/boardforge-maintenance:run"
    ]

    # Duplicate query for profile-a reuses cached request_id and coalesces Cloud Run dispatch
    r2 = client.post("/discover/profile-a", headers=headers_a, json={"profile_id": "profile-a", "query": "backend"})
    assert r2.status_code == 200
    assert r2.json()["discovery"]["request_id"] == req_id
    assert r2.json()["discovery"]["cached"] is True
    assert r2.json()["discovery"]["dispatch"]["coalesced"] is True
    assert len(cloud_calls) == 1

    # Ownership check on GET /discovery/{id}: owner gets 200, other profile gets 404, unauth gets 401
    assert client.get(f"/discovery/{req_id}").status_code == 401
    assert client.get(f"/discovery/{req_id}", headers=headers_b).status_code == 404
    own = client.get(f"/discovery/{req_id}", headers=headers_a)
    assert own.status_code == 200
    assert own.json()["request_id"] == req_id


def test_conversational_preference_correction_is_transient_and_openai_prefs_gated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    store = Store(str(tmp_path / "prefs_policy.db"))
    store.save_preferences("u", Preferences(countries=["India"], roles=["AI/ML"]))
    store.upsert_source_jobs("test", "pune", [
        Job(
            source_kind="test", source_key="pune", external_id="pune-be",
            title="Backend Engineer", company="PuneTech", country="India",
            location="Pune, India", description="Python backend APIs.",
        ),
    ])
    agent = JobAgent(store, extractor=_DeterministicExtractor(), embeddings=_DisabledEmbeddings())

    for turn in (
        "Changed my mind, show me backend jobs in Pune",
        "I changed my mind, Pune is fine too",
        "On second thought, find backend roles in Pune",
    ):
        assert tool_names(turn) == ["search_jobs"]
        res = asyncio.run(agent.handle("u", turn))
        assert [s["tool"] for s in res.data["trace"]] == ["search_jobs"]
        # Saved preferences MUST remain untouched!
        assert store.load_preferences("u") == Preferences(countries=["India"], roles=["AI/ML"])

    # AgentToolExecutor also blocks update_preferences on conversational "Changed my mind..." turns
    executor = AgentToolExecutor(agent)
    state = AgentExecutionState(
        profile_id="u",
        user_text="Changed my mind, show me backend jobs in Pune",
    )
    denied = asyncio.run(executor.execute(state, "update_preferences", {
        "text": "backend jobs in Pune",
        "authorization_quote": "Changed my mind, show me backend jobs in Pune",
    }))
    assert denied["ok"] is False
    assert denied["error"] == "preference_update_not_authorized"

    # Explicit persistence with "from now on" IS authorized and persisted
    state_ok = AgentExecutionState(
        profile_id="u",
        user_text="Changed my mind: from now on I only want backend roles in Pune",
    )
    allowed = asyncio.run(executor.execute(state_ok, "update_preferences", {
        "text": "from now on I only want backend roles in Pune",
        "authorization_quote": "from now on I only want backend",
    }))
    assert allowed["ok"] is True
    updated = store.load_preferences("u")
    assert "Pune" in updated.locations

    # OpenAI preference extraction stays disabled even if OPENAI_API_KEY is present unless explicitly enabled
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-be-used")
    monkeypatch.delenv("BOARDFORGE_ENABLE_LLM_PREFERENCES", raising=False)
    extractor = LLMPreferenceExtractor()
    assert extractor.enabled is False
    extracted = asyncio.run(extractor.extract("Only India roles"))
    assert extracted.countries == ["India"]


@pytest.mark.skipif(_chrome() is None, reason="Chrome or Edge is required for browser discovery validation")
def test_browser_discovery_submission_polling_refresh_and_signout(tmp_path: Path):
    mock_fetch = r"""
    window.__DISCOVERY_POLL_MS = 25;
    window.__DISCOVERY_MAX_POLLS = 10;
    window.__calls = [];
    window.__poll77 = 0;
    window.__poll78 = 0;
    const sleep = ms => new Promise(r => setTimeout(r, ms));
    const response = (status, data, delay = 0) => new Promise(r => setTimeout(() => r({
      status, ok: status >= 200 && status < 300, json: async () => data
    }), delay));

    window.fetch = (path, options = {}) => {
      const headers = new Headers(options.headers || {});
      const auth = headers.get('Authorization') || '';
      const method = options.method || 'GET';
      window.__calls.push({path: String(path), method, auth});

      if (path === '/health') return response(200, {version: 'test'});
      if (path === '/session') {
        if (auth === 'Bearer token-a') return response(200, {profile_id: 'profile-a', auth_required: true});
        if (auth === 'Bearer token-b') return response(200, {profile_id: 'profile-b', auth_required: true});
        return response(401, {detail: 'unauthorized'});
      }
      if (String(path).startsWith('/candidate-profile/')) return response(200, {profile: null});
      if (String(path).startsWith('/preferences/')) return response(200, {profile_id: 'p', preferences: {}});

      if (path === '/agent') {
        const body = JSON.parse(options.body || '{}');
        if (body.query === 'disabled case') {
          return response(200, {
            message: 'No suitable indexed jobs.',
            trace: [{tool: 'search_jobs', ok: true, arguments: {query: body.query}}],
            matches: []
          });
        }
        return response(200, {
          message: 'Found 1 possible result.',
          trace: [{tool: 'search_jobs', ok: true, arguments: {query: body.query}}],
          matches: [{
            job: {stable_id: 'test:idx:1', title: 'Existing Indexed Role', company: 'IdxCo', location: 'India'},
            score: 0.42, status: 'needs_verification', reasons: ['Partial match'], risks: ['Unverified'], evidence: []
          }]
        });
      }

      if (path === '/discover/profile-a') {
        const body = JSON.parse(options.body || '{}');
        if (body.query === 'disabled case') {
          return response(200, {
            discovery: {request_id: null, status: 'disabled', cached: false, reason: 'Web discovery disabled.'},
            matches: []
          }, 15);
        }
        if (body.query === 'signout case') {
          return response(200, {
            discovery: {request_id: 78, status: 'queued', cached: false, report: null},
            matches: []
          }, 10);
        }
        return response(200, {
          discovery: {request_id: 77, status: 'queued', cached: false, report: null},
          matches: []
        }, 20);
      }

      if (path === '/discovery/77') {
        window.__poll77++;
        if (window.__poll77 === 1) {
          return response(200, {request_id: 77, status: 'running', report: null});
        }
        return response(200, {
          request_id: 77,
          status: 'completed',
          cached: false,
          report: {synced_boards: 2, jobs_fetched: 5}
        });
      }

      if (path === '/discovery/78') {
        window.__poll78++;
        return response(200, {request_id: 78, status: 'running', report: null});
      }

      if (path === '/search') {
        return response(200, {
          matches: [
            {
              job: {stable_id: 'gh:new:1', title: 'Discovered Rust Engineer', company: 'NewBoardCo', location: 'Bengaluru, India'},
              score: 0.88, status: 'recommended', reasons: ['Matched Rust'], risks: [], evidence: []
            },
            {
              job: {stable_id: 'test:idx:1', title: 'Existing Indexed Role', company: 'IdxCo', location: 'India'},
              score: 0.42, status: 'needs_verification', reasons: ['Partial match'], risks: ['Unverified'], evidence: []
            }
          ]
        }, 10);
      }

      return response(404, {detail: 'unmocked ' + path});
    };
    """

    harness = r"""
    (async () => {
      const waitFor = async (pred, label) => {
        for (let i = 0; i < 120; i++) { if (pred()) return; await sleep(15); }
        throw new Error('timeout: ' + label);
      };
      const assert = (cond, msg) => { if (!cond) throw new Error(msg); };
      try {
        await connect('token-a');
        await waitFor(() => !document.querySelector('main').hidden, 'signed in');

        // 1. Search with <3 recommended matches exposes Discover button while rendering existing match
        document.getElementById('query').value = 'rust backend';
        document.getElementById('searchBtn').click();
        await waitFor(() => document.querySelectorAll('.job').length === 1, 'initial result');
        assert(!document.getElementById('discoveryBar').hidden, 'discoveryBar should be visible');
        assert(!document.getElementById('discoverBtn').disabled, 'discoverBtn should be enabled');

        // 2. Click Discover twice rapidly -> duplicate prevented, existing result preserved during polling, then refreshed via /search
        const btn = document.getElementById('discoverBtn');
        btn.click();
        btn.click();
        assert(btn.disabled, 'discoverBtn must lock during submission');
        assert(document.querySelectorAll('.job').length === 1, 'current results must stay visible while discovery runs');

        await waitFor(() => document.querySelectorAll('.job').length === 2, 'refreshed results after discovery');
        assert(document.getElementById('discoveryStateBadge').textContent === 'Completed', 'completed badge missing');
        assert(document.getElementById('discoveryStatus').textContent.includes('2 board(s) synced'), 'board count missing');
        assert(document.querySelector('.job h3').textContent === 'Discovered Rust Engineer', 'discovered job not rendered');

        const discoverPosts = window.__calls.filter(c => c.path === '/discover/profile-a');
        const searchRefreshes = window.__calls.filter(c => c.path === '/search');
        assert(discoverPosts.length === 1, 'duplicate /discover submission was not prevented');
        assert(searchRefreshes.length === 1, 'refresh should run exactly once via /search without looping');

        // 3. Disabled discovery state renders cleanly
        document.getElementById('query').value = 'disabled case';
        document.getElementById('searchBtn').click();
        await waitFor(() => document.querySelectorAll('.job').length === 0, 'disabled search empty');
        document.getElementById('discoverBtn').click();
        await waitFor(() => document.getElementById('discoveryStateBadge').textContent === 'Disabled', 'disabled badge');
        assert(document.getElementById('discoverBtn').disabled, 'discoverBtn should stay disabled when discovery is off');

        // 4. Switching profiles / signing out stops active polling immediately
        document.getElementById('query').value = 'signout case';
        document.getElementById('searchBtn').click();
        await waitFor(() => !document.getElementById('discoverBtn').disabled, 'ready for signout discovery');
        document.getElementById('discoverBtn').click();
        await waitFor(() => window.__poll78 >= 1, 'started polling #78');
        await connect('token-b');
        const pollsAtSwitch = window.__poll78;
        await sleep(90);
        assert(window.__poll78 === pollsAtSwitch, 'polling continued after profile switch');
        assert(document.getElementById('discoveryBar').hidden, 'discoveryBar not reset on profile switch');

        document.body.dataset.discoveryAudit = 'pass';
      } catch (err) {
        document.body.dataset.discoveryAudit = 'fail: ' + err.message;
      }
    })();
    """

    html = UI_PATH.read_text(encoding="utf-8")
    html = html.replace("<script>", f"<script>{mock_fetch}</script><script>", 1)
    html = html.replace("</body>", f"<script>{harness}</script></body>")
    page_path = tmp_path / "discovery-ui-audit.html"
    page_path.write_text(html, encoding="utf-8")
    dom_path = tmp_path / "dom.html"
    err_path = tmp_path / "chrome_err.log"
    cmd = [
        _chrome(), "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-first-run", "--disable-breakpad", "--disable-crash-reporter",
        "--disable-background-networking", "--disable-component-update", "--disable-sync",
        "--host-resolver-rules=MAP * 0.0.0.0", f"--user-data-dir={tmp_path / 'chrome-prof'}",
        "--virtual-time-budget=10000", "--dump-dom", page_path.as_uri(),
    ]
    with dom_path.open("w", encoding="utf-8") as out_f, err_path.open("w", encoding="utf-8") as err_f:
        completed = subprocess.run(
            cmd, stdin=subprocess.DEVNULL, stdout=out_f, stderr=err_f, timeout=30, check=False
        )
    dom_out = dom_path.read_text(encoding="utf-8", errors="replace")
    err_out = err_path.read_text(encoding="utf-8", errors="replace")
    assert completed.returncode == 0, err_out
    assert 'data-discovery-audit="pass"' in dom_out, dom_out[-3000:]
