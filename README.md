# BoardForge

BoardForge is an evaluation-first job intelligence backend with assisted connector discovery, built around a simple idea:

> job search systems should understand changing preferences, track changing job boards, and explain recommendations with evidence.

It is intentionally not another generic RAG demo or auto-apply bot.

## What it does

- Detects public ATS providers from career URLs
- Fetches live jobs from Greenhouse, Lever and Ashby
- Falls back to JSON-LD JobPosting extraction for generic career pages
- Normalizes jobs into one schema
- Tracks sync runs and job discovery, update, closure and reopening events in SQLite
- Stores persistent candidate preferences and explicit feedback
- Normalizes INR salary ranges and conservative geography metadata
- Filters hard constraints before ranking and rejects unknown geography when required
- Produces structured evidence-backed reasons and risk flags
- Exposes sync, preference, search, feedback and agent APIs
- Includes an executable starter benchmark, ranking/stability metrics and CI tests
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
career URL
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
             API / agent
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

Open `http://127.0.0.1:8000/docs`.

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

Run `python -m boardforge.cli benchmark` for the 120-case adversarial ranking suite. Run `python -m boardforge.cli compare --markdown-out results/ranking.md` for lexical → constrained → evidence → feedback ablations, and `python -m boardforge.cli agent-eval` for the 40-case tool-selection suite. These datasets are synthetic regression/evaluation fixtures, not claims of real-user product quality.

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
3. scheduled source synchronization and operational telemetry
4. automatic validation of learned manifests against browser-visible counts

The project explicitly excludes authentication bypass and protected/private API access.

## Interview story

> I wanted to build job-search infrastructure rather than another matching demo. BoardForge ingests live ATS data, records lifecycle events, refuses unsafe mass-closure syncs, remembers candidate constraints, and returns structured evidence for every match. HAR-assisted discovery produces reusable declarative connectors without executing generated code. A reproducible benchmark makes ranking changes measurable rather than anecdotal.


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
