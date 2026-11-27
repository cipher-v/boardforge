# Architecture

## Deployed beta snapshot

The deployed application uses Cloud Run, Neon PostgreSQL/pgvector, Vertex AI and a one-shot maintenance job. Cloud Scheduler is configured for twice-daily execution. Private invitation tokens isolate profiles; persistent beta quotas and the cumulative estimated AI budget constrain paid operations. Production discovery is dispatched to the maintenance job. Production settings are intentionally tighter than general configuration examples below. See [results](results.md) for observed validation and limitations.

BoardForge separates deterministic infrastructure from probabilistic model behavior.

## Data plane

1. **Source detection** maps a career URL to an official ATS connector (`Greenhouse`, `Lever`, `Ashby`, or declarative JSON-LD/learned manifests).
2. **Connector** fetches public job-board postings after DNS/IP SSRF validation.
3. **Normalizer** converts source-specific payloads to the canonical `Job` schema (`enrich.py`), prioritizing structured location/title workplace signals over company boilerplate.
4. **Temporal store** records sync runs, first/last observation, content hashes, and lifecycle events (`discovered`, `updated`, `closed`, `reopened`).
5. **Preference memory** stores durable candidate preferences and explicit save/reject feedback, isolated per authenticated profile.
6. **Candidate profile** stores resume-derived fields with supporting snippets and confirmation status.
7. **Ranker** enforces deterministic hard constraints first, then scores lexical/semantic relevance and confidence evidence.
8. **API & UI** expose profile management, preference viewing/resetting, search, agent orchestration, feedback, and explicit on-demand source discovery.

When `DATABASE_URL` is PostgreSQL, candidate generation runs inside PostgreSQL: full-text `ts_rank_cd` over explicit search intent (`lexical_query_text`) and HNSW `pgvector` similarity (`768`-d) produce a bounded candidate union, followed by deterministic constraint enforcement and evidence reranking. SQLite implements the same store contract for local development and isolated tests.

Resume-inferred role families remain soft evidence until the user confirms `target_roles` or saves explicit role preferences. Conversational turn corrections in the search bar (including `"Changed my mind, show me..."`) act as transient search overlays unless the user explicitly asks to persist preferences (`"from now on"`, `"save my preferences"`, `"anymore"`, or the Explicit Preferences panel).

## On-demand discovery lifecycle

Interactive search never blocks on web search or external ATS scraping, and ordinary low-match searches do **not** automatically spend discovery or LLM quota:

```text
User clicks "Discover more opportunities" (POST /discover/{profile_id})
   OR explicit conditional agent turn calls discover_job_sources (POST /agent)
                     |
                     v
   Deduplicated durable row in discovery_requests (TTL-cached by fingerprint)
                     |
                     +--> Local / Docker: coalesced in-process bounded drain
                     +--> Cloud Run: on-demand Cloud Run Job (:run) dispatch
                                             |
                                             v
                     Tavily domain-restricted search -> canonical ATS boards (<= 6)
                                             |
                                             v
                     Provenance recorded -> bounded concurrent ATS sync -> embedding backfill
                                             |
                                             v
         Browser polls GET /discovery/{request_id} -> refreshes via POST /search
```

- **Bounded execution, no always-on worker required:** Discovery requests are persisted in `discovery_requests` and claimed atomically (`claim_discoveries`). Locally (`BOARDFORGE_DISCOVERY_DISPATCH_MODE=local`), the API schedules a coalesced single-flight drain cycle in-process. On Cloud Run (`BOARDFORGE_DISCOVERY_DISPATCH_MODE=cloud_run_job`), the API triggers the one-shot `boardforge-maintenance` Cloud Run Job (`python -m boardforge.worker --once`) via the Cloud Run Jobs v2 `:run` endpoint with a dispatch cooldown.
- **Loop-safe UI refresh:** When `GET /discovery/{request_id}` reports `completed`, the browser refreshes results via `POST /search` (never re-invoking `/discover` or `/agent` discovery), preventing polling-induced discovery loops.

## Why temporal indexing matters

A job index is not a static dataset. Roles appear, change, get reposted, and close. BoardForge records discovery, update, closure, and reopening events. A job must be absent from two consecutive successful syncs before it closes; empty fetches and abnormal source shrinkage (`< 25%` of `>= 10` active jobs) are rejected.

## Probabilistic boundary

Vertex AI is the supported deployment provider (`gemini-embedding-001` for `768`-d embeddings and `gemini-3.8-flash` for relevance verification and conditional agent reasoning). OpenAI-dependent preference extraction is disabled by default (`BOARDFORGE_ENABLE_LLM_PREFERENCES=0`), so preference updates use deterministic extraction.

The optional **Relevance Verifier** (`BOARDFORGE_ENABLE_RELEVANCE_VERIFIER=1`) runs after retrieval, hard-constraint filtering, rejection memory, and deterministic confidence checks, over a bounded shortlist:
- Validated `direct` responsibility evidence can confirm a relevance-uncertain candidate; `adjacent` becomes an `alternative`; `irrelevant` is excluded; `uncertain`, malformed, timed-out, or unsupported output remains `needs_verification`.
- It **cannot** waive deterministic hard constraints, restore rejected jobs, reopen no-vacancy postings, or treat unknown salary/experience metadata as verified eligibility.
- Judgments are cached by job-content hash, normalized intent, canonical preferences, model, and prompt version.

## Intelligent agent boundary

The optional **Intelligent Agent** (`BOARDFORGE_ENABLE_INTELLIGENT_AGENT=1`) wraps `JobAgent` with bounded orchestration:
- **Deterministic fast path:** Straightforward single-intent searches execute `search_indexed_jobs` directly with `0` agent LLM calls and `0` automatic discovery calls.
- **Conditional & discovery turns:** Multi-step turns (conditional alternatives, explicit discovery requests, or combined preference/feedback + search turns) invoke Vertex Gemini with typed tool declarations (`search_indexed_jobs`, `discover_job_sources`, `synchronize_trusted_source`, `update_preferences`, `retrieve_search_feedback`, `record_job_feedback`).
- **Deterministic executor enforcement:** `AgentToolExecutor` (not the model) enforces authorization: preference writes require explicit persistence intent plus a verbatim quote from the current user turn; `discover_job_sources` is rejected before an indexed search or when `recommended_count >= min_sufficient_results`; `synchronize_trusted_source` accepts only boards already in `discovered_sources`; and once `discover_job_sources` queues a request, the turn stops cleanly (`discovery_queued`) without spending an extra reasoning call, returning `discovery` (including `request_id`) directly to the browser.
- Every paid model, embedding, and Tavily call reserves quota in the atomic `llm_usage_reservations` ledger before network execution.

## Connector discovery safety

Offline HAR learning (`boardforge learn-har`) inspects user-captured public XHR/fetch traffic to generate declarative `GET`-only connector manifests (`sync-manifest`). Private/loopback IPs, redirects to non-public hosts, credential-bearing URLs, and authenticated HAR entries are rejected.
