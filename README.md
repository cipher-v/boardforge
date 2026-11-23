# BoardForge

BoardForge is a self-learning job intelligence backend built around a simple idea:

> job search systems should understand changing preferences, track changing job boards, and explain recommendations with evidence.

It is intentionally not another generic RAG demo or auto-apply bot.

## What it does

- Detects public ATS providers from career URLs
- Fetches live jobs from Greenhouse, Lever and Ashby
- Falls back to JSON-LD JobPosting extraction for generic career pages
- Normalizes jobs into one schema
- Tracks first-seen, last-seen, updates and closures in SQLite
- Stores persistent candidate preferences and explicit feedback
- Filters hard constraints before ranking
- Produces evidence-backed reasons and risk flags
- Exposes sync, preference, search, feedback and agent APIs
- Includes ranking/stability evaluation utilities and CI tests
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
(first_seen / last_seen / active / closed / content hash)
   |
   +----------------------+
   |                      |
preferences + feedback    |
   |                      |
   +-----------> ranker <-+
                  |
                  v
      reasons + risks + violations
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

The same source is synced repeatedly. BoardForge records first-seen time, last observation, content changes, disappearance from the source, and active/closed state.

This lets the system answer a more useful question than “what jobs exist?”:

> what changed since the last sync?

## Evaluation

Current utilities include Precision@K, Recall@K, NDCG@K, hard-constraint violation rate, explanation evidence coverage, and Jaccard@K for paraphrase stability.

See [docs/evaluation.md](docs/evaluation.md).

## Current ATS support

| Source | Method |
|---|---|
| Greenhouse | Public job-board API |
| Lever | Public postings API |
| Ashby | Public job-board API |
| Generic career page | JSON-LD JobPosting extraction |

## Next major milestone

Unknown-site learning:

1. open a public careers page in an instrumented browser
2. observe public XHR/fetch traffic
3. identify the job-list endpoint and response schema
4. generate a reusable connector manifest
5. validate connector output against the browser-visible job count
6. reuse the connector without browser automation on future syncs

The project explicitly excludes authentication bypass and protected/private API access.

## Interview story

> I wanted to build job-search infrastructure rather than another matching demo. I started with live ATS ingestion and a temporal index so the system knows when jobs appear, change or close. Candidate preferences and feedback are persistent, hard constraints are deterministic, and ranking produces evidence plus risk flags. The next step is learning connectors for unknown public career sites from browser-observed network traffic and evaluating the whole pipeline for freshness, constraint violations, ranking quality and stability.
