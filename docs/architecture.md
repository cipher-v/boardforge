# Architecture

BoardForge separates deterministic infrastructure from probabilistic model behavior.

## Data plane

1. **Source detection** maps a career URL to an ATS connector.
2. **Connector** fetches public job-board data.
3. **Normalizer** converts source-specific payloads to the canonical `Job` schema.
4. **Temporal store** records sync runs, first/last observation, content hashes and lifecycle events.
5. **Preference memory** stores durable candidate preferences and explicit feedback.
6. **Candidate profile** stores resume-derived fields with supporting snippets and confidence.
7. **Ranker** applies hard constraints first, then relevance and preference evidence.
8. **API/agent** exposes sync, profile review, preference, search and feedback tools.

When `DATABASE_URL` is PostgreSQL, candidate generation stays inside the database:
GIN full-text retrieval and HNSW pgvector similarity produce a bounded candidate
set, followed by deterministic constraint enforcement and evidence reranking.
SQLite implements the same store contract as a local/test fallback.

Resume-inferred role families are soft evidence. Only roles explicitly saved as a
preference or confirmed as `target_roles` become hard title-family constraints.
This prevents an extraction mistake from silently deleting otherwise useful jobs.

## Online discovery lifecycle

Interactive search never waits for web search or ATS ingestion:

```text
search/resume request -> rank current index -> response
                    \-> durable deduplicated discovery request
                                             |
                                             v
                              web search -> canonical boards
                                             |
                                             v
                                provenance -> ATS sync
                                             |
                                             v
                                    UI polls and refreshes
```

Discovery fingerprints query, role and geography context. Repeated equivalent
requests reuse recent work for a configurable TTL. Each source records the query
that discovered it and its last sync status. The demo uses FastAPI background
tasks; production should move execution to a separate retrying worker.

When embeddings are enabled, job vectors are cached in the store by stable job ID, embedding model and content hash. Queries are embedded per request; unchanged job documents reuse their stored vectors, while changed documents are re-embedded automatically.

## Why temporal indexing matters

A job index is not a static dataset. Roles appear, change, get reposted and close. BoardForge records discovery, update, closure and reopening events. A job must be absent from two consecutive successful syncs before it closes; empty and abnormal source shrinkage is rejected.

## Probabilistic boundary

LLM use is optional and narrow:
- structured preference extraction
- bidirectional relevance verification over a bounded, eligible candidate pool

The optional relevance verifier runs after retrieval, deterministic hard
constraints, rejection memory, and confidence enforcement but before final
top-K selection. It receives eligible `recommended` and `needs_verification`
candidates from a bounded pool. A supported direct judgment can promote a
relevance-uncertain row; adjacent becomes a separately identified alternative;
irrelevant is omitted; uncertain, malformed, missing, timed-out, or unsupported
remains `needs_verification`. It cannot restore a rejected job, reopen a
confirmed closed vacancy, resolve unknown eligibility, or waive a constraint.
Job descriptions are untrusted JSON data, never prompt instructions.

Judgments are cached by job-content hash, normalized intent, canonical
preferences, provider model, and prompt version. The provider boundary is a
small structured-batch protocol over a reusable generation interface supporting
schemas and future tool declarations. The primary adapter uses the official
Google Gen AI SDK in Vertex mode with ADC; the OpenAI-compatible/Groq adapter
remains available, while tests inject mock providers.

Core ingestion, state tracking, constraint enforcement and evaluation remain deterministic and testable.

## Intelligent agent boundary

The optional intelligent agent extends `JobAgent`; it does not replace retrieval,
ranking, relevance verification, or persistent state. Gemini through Vertex is
the primary reasoning provider. Groq with `openai/gpt-oss-120b` remains an
explicitly selected alternative, with no automatic provider switching.

The reasoning model can request only typed BoardForge tools:

- read profile, preferences, and previous feedback;
- search the existing index;
- queue targeted discovery after an insufficient indexed search;
- synchronize an exact source already present in the trusted-source registry;
- persist an explicitly authorized preference change; and
- record or read job feedback.

The executor, not the model, authorizes each action. Preference writes require
an exact quote from the current user turn. Discovery is rejected before indexed
search and after sufficient verified results. Source sync cannot accept an
arbitrary URL. Exact duplicate calls are rejected. Job descriptions and all tool
results remain marked as untrusted data. The model has no code-execution,
credential, job-creation, or job-application capability.

Each turn has independent iteration, tool-call, token, time, and cumulative-cost
limits. Reservations enter the same persistent atomic usage ledger as relevance,
embedding, and discovery calls, including retry capacity. The final user-facing
job summary is rendered from bounded tool output rather than unverified model
prose. If agent reasoning is disabled or unavailable, the existing deterministic
planner remains the fallback.

## Connector discovery safety

Unknown-site learning:
- open a public careers page in an instrumented browser
- observe public XHR/fetch requests
- identify stable job-list endpoints
- generate a connector manifest
- validate it against the browser-visible job count
- reuse the connector without browser automation on future syncs

Learned manifests are declarative and limited to public GET endpoints. Private/local targets, redirects, credential-bearing URLs, authenticated HAR entries and sensitive query parameters are rejected. No authentication bypass or protected API access is part of the design.

## Next architecture milestone

- persistent full-text and vector indexes
- pgvector candidate retrieval over the existing content-hash-keyed embedding cache
- an external queued worker, retries and operational metrics
- consented real-user agent traces and human-reviewed tool-use evaluation
