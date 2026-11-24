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
- later: reranking and explanation generation

Core ingestion, state tracking, constraint enforcement and evaluation remain deterministic and testable.

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
- larger ranking and tool-use evaluation datasets
