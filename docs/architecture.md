# Architecture

BoardForge separates deterministic infrastructure from probabilistic model behavior.

## Data plane

1. **Source detection** maps a career URL to an ATS connector.
2. **Connector** fetches public job-board data.
3. **Normalizer** converts source-specific payloads to the canonical `Job` schema.
4. **Temporal store** records first-seen, last-seen, content hash and active/closed state.
5. **Preference memory** stores durable candidate preferences and explicit feedback.
6. **Ranker** applies hard constraints first, then relevance and preference evidence.
7. **API/agent** exposes sync, preference update, search and feedback tools.

## Why temporal indexing matters

A job index is not a static dataset. Roles appear, change, get reposted and close. BoardForge stores state transitions so stale listings can be detected instead of repeatedly recommended.

## Probabilistic boundary

LLM use is optional and narrow:
- structured preference extraction
- later: reranking and explanation generation

Core ingestion, state tracking, constraint enforcement and evaluation remain deterministic and testable.

## Next architecture milestone

Unknown-site learning:
- open a public careers page in an instrumented browser
- observe public XHR/fetch requests
- identify stable job-list endpoints
- generate a connector manifest
- validate it against the browser-visible job count
- reuse the connector without browser automation on future syncs

No authentication bypass or protected API access is part of the design.
