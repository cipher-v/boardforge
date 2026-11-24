# BoardForge

BoardForge is an evaluation-first job intelligence backend with assisted connector discovery, built around a simple idea:

> job search systems should understand changing preferences, track changing job boards, and explain recommendations with evidence.

It is intentionally not another generic RAG demo or auto-apply bot.

## What it does

- Accepts PDF/DOCX/TXT resumes and extracts skills, roles, projects, experience and evidence snippets
- Lets candidates review inferred skills and confirm target roles before those roles become hard constraints
- Returns recommendations from the current index immediately, then refreshes after optional background source discovery
- Detects public ATS providers from career URLs
- Fetches live jobs from Greenhouse, Lever and Ashby
- Falls back to JSON-LD JobPosting extraction for generic career pages
- Normalizes jobs into one schema
- Tracks sync runs and job discovery, update, closure and reopening events in SQLite
- Stores persistent candidate preferences and explicit feedback
- Normalizes INR salary ranges and conservative geography metadata
- Filters hard constraints before ranking and rejects unknown geography when required
- Produces structured evidence-backed reasons and risk flags
- Persists job embeddings by model and content hash so unchanged jobs are not re-embedded
- Exposes sync, preference, search, feedback and agent APIs
- Includes ranking, agent, preference and resume-extraction benchmarks plus CI tests
- Optionally uses an OpenAI-compatible model for structured preference extraction

## Why this exists

A normal recommender can retrieve semantically similar jobs and still fail badly:

- recommend an unpaid role to a paid-only candidate
- recommend a US-only role to an India-only candidate
- ignore a 3+ year requirement for a fresher
- repeatedly show a role the candidate rejected
- keep recommending a role that already closed
- invent a reason that is not supported by the job metadata

BoardForge treats those as first-class engineering failures.

## Architecture

```text
resume ----------------> evidence-backed candidate profile
                              |
                              v
career URL -----------> source discovery queue
   |
   v
source detector
   |
   +--> Greenhouse connector
   +--> Lever connector
   +--> Ashby connector
   +--> generic JSON-LD fallback
   |
   v
canonical Job schema
   |
   v
temporal SQLite index
(sync runs / missing observations / lifecycle events / content hash)
   |
   +----------------------+
   |                      |
preferences + feedback    |
   |                      |
   +-----------> ranker <-+
                  |
                  v
       structured evidence + risks
                  |
                  v
             API / tool-using agent
```

See [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -e ".[dev]"
pytest -q
uvicorn boardforge.api:app --reload
```

Open `http://127.0.0.1:8000` for the local product demo or `/docs` for the API.

### PostgreSQL + pgvector stack

```bash
docker compose up --build
```

This starts three services: the FastAPI application, a PostgreSQL 16 + pgvector
database, and an independent discovery/source-refresh worker. Migrations are
versioned under `boardforge/migrations` and applied when a PostgreSQL store starts.
SQLite remains the zero-configuration test/local fallback when `DATABASE_URL` is unset.

PostgreSQL uses a GIN full-text index for lexical candidate generation and an
HNSW pgvector index for semantic candidates. Only a bounded candidate set reaches
the evidence reranker; the application no longer needs to load the full job index.

Populate or refresh missing job embeddings with:

```bash
python -m boardforge.cli embed-index --batch-size 64 --max-jobs 10000
```

The included browser UI is intentionally a single-user local demo. It stores the
extracted profile, preferences and feedback in SQLite, never the raw resume, and
provides a **Delete my data** action. Add real authentication and per-user
authorization before exposing it as a public multi-user service.

API deployments can enable profile isolation with a JSON token map and a separate
admin token. Authentication is disabled when these variables are empty:

```text
BOARDFORGE_PROFILE_TOKENS={"candidate-secret-token":"candidate-profile-id"}
BOARDFORGE_ADMIN_TOKEN=admin-secret-token
```

Protected requests use `Authorization: Bearer <token>`. For a consumer product,
replace static tokens with an external OIDC provider while keeping the same
profile-ownership boundary.

## Optional background web discovery

Search always returns from the current index first. With Tavily configured,
BoardForge queues a deduplicated discovery request, records source provenance,
discovers Greenhouse/Lever/Ashby boards with one Basic search request restricted
to supported ATS domains, syncs those boards after the response, and lets the UI
poll for a refreshed result set.

```text
BOARDFORGE_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=...
BOARDFORGE_DISCOVERY_TTL_SECONDS=900
```

Brave remains an optional fallback provider:

```text
BOARDFORGE_SEARCH_PROVIDER=brave
BOARDFORGE_BRAVE_API_KEY=...
```

Do not commit real API keys. Keep them in the local shell, a local `.env` file,
or the deployment platform's secret manager.

The FastAPI demo executes queued work in-process. Request state is durable and a
separate process can drain/recover pending work with:

```bash
python -m boardforge.cli discovery-worker --limit 10
```

A production deployment should run this worker independently with scheduling,
retry policy and telemetry.

## Live-source CLI demo

```bash
python -m boardforge.cli sync https://job-boards.greenhouse.io/skildai-careers
python -m boardforge.cli sync https://jobs.lever.co/trackmind

python -m boardforge.cli prefs "I want AI/ML, computer vision or backend roles in India. Avoid hardware-heavy roles and unpaid internships."

python -m boardforge.cli search "Python LLM retrieval roles" --top-k 10
```

## API

### Sync a career page

```http
POST /sync
```

```json
{"url":"https://job-boards.greenhouse.io/skildai-careers"}
```

The response reports how many roles were inserted, updated, unchanged or closed.

### Update durable preferences

```http
POST /preferences
```

```json
{
  "profile_id":"tanish",
  "text":"I only want India-eligible roles. Avoid unpaid and hardware-heavy positions."
}
```

### Search with memory

```http
POST /search
```

```json
{
  "profile_id":"tanish",
  "query":"LLM agents retrieval backend",
  "top_k":10
}
```

### Record feedback

```http
POST /feedback
```

```json
{
  "profile_id":"tanish",
  "job_id":"greenhouse:example:123",
  "verdict":"reject",
  "note":"Too hardware-heavy"
}
```

Rejected jobs are suppressed in later searches.

## Optional structured LLM extraction

Without an API key, BoardForge uses a deterministic preference extractor.

Set these environment variables to enable structured model output:

```text
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
```

The LLM is intentionally kept at the probabilistic boundary. Ingestion, persistence, hard constraints, state tracking and evaluation remain deterministic and testable.

## Temporal intelligence

The same source is synced repeatedly. BoardForge records first-seen time, last observation, content changes, disappearance from the source, and active/closed state. Jobs close only after two consecutive successful observations in which they are absent. Empty or abnormally shrunken source responses are rejected rather than interpreted as mass closures.

This lets the system answer a more useful question than “what jobs exist?”:

> what changed since the last sync?

## Evaluation

Run `python -m boardforge.cli benchmark` for the 120-case adversarial ranking suite. Run `python -m boardforge.cli compare --markdown-out results/ranking.md` for lexical → constrained → evidence → feedback ablations, `python -m boardforge.cli agent-eval` for the 40-case production-fallback tool planner, `python -m boardforge.cli preference-eval` for natural-language preference extraction, and `python -m boardforge.cli resume-eval` for resume profile extraction. These datasets are synthetic regression/evaluation fixtures, not claims of real-user product quality.

See [docs/evaluation.md](docs/evaluation.md).

## Current ATS support

| Source | Method |
|---|---|
| Greenhouse | Public job-board API |
| Lever | Public postings API |
| Ashby | Public job-board API |
| Generic career page | JSON-LD JobPosting extraction |

## Next major milestone

Planned production milestones:

1. persistent hybrid full-text/vector retrieval
2. real-user labels to complement the packaged 120-case ranking and 40-case agent-tool benchmarks
3. an external discovery/sync worker with retries and operational telemetry
4. automatic validation of learned manifests against browser-visible counts

The project explicitly excludes authentication bypass and protected/private API access.

## Interview story

> I wanted to build a career agent on top of reliable job-search infrastructure rather than hide retrieval behind an LLM. BoardForge turns a resume into editable evidence, keeps inferred roles soft until the candidate confirms them, returns indexed results immediately, and discovers new ATS sources in the background. Deterministic constraints, temporal job state, feedback memory and versioned evals keep the agent measurable rather than anecdotal.


## Learn an unknown public career site from a browser HAR

BoardForge can infer a reusable connector manifest from a public browser network capture.

1. Open the public careers page in your browser.
2. Open DevTools -> Network.
3. Reload the page and export the network log as HAR.
4. Run:

```bash
python -m boardforge.cli learn-har careers.har --out connector-manifests.json
```

BoardForge scans JSON/XHR responses for job-like record sets, scores candidate endpoints, infers the response path and field mapping, and writes reusable manifests.

Inspect the generated manifest, then sync it:

```bash
python -m boardforge.cli sync-manifest connector-manifests.json --index 0
```

Only public GET endpoints are executed by learned manifests. HAR entries with authentication, cookies, API keys or sensitive query parameters are ignored. Private/local network targets and redirects are rejected.
