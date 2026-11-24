# BoardForge engineering log

This is the running implementation journal for substantial BoardForge tasks.
Future work should add a dated section describing the request, problems found,
decisions, changes, verification, and anything still requiring human action.

The goal is to preserve both the technical reasoning and a plain-language
explanation of why each change was made. It is not a changelog: routine file
lists belong in Git history; this document records engineering evidence and
trade-offs.

## 2026-11-24 — Independent holdout and production-retrieval evaluation

### Requested outcome

Create an independent real-job holdout, provisional evidence-backed labels, and
an evaluation path through the production PostgreSQL retrieval code. Stop
before the paid Vertex run, preserve India v1.1 unchanged, and keep all work
uncommitted for review.

### Dataset decisions

- The source manifest contains ten targeted official ATS boards: Anthropic,
  Cloudflare, Datadog, Figma, Roblox, Palantir, Notion, Linear, Ramp, and
  Perplexity. It uses only existing Greenhouse, Lever, and Ashby connectors and
  excludes the development companies Coinbase, Jumio, and Twilio.
- Collection is deterministic: each company receives an equal quota and jobs
  are selected by a category round-robin with stable title/location/ID ordering.
  Collection refuses to overwrite an existing snapshot unless `--force` is
  explicit.
- The frozen corpus contains 100 jobs from ten companies. It stores full job
  descriptions, official application and board URLs, stable IDs, collection
  timestamps, category/seniority metadata, and a canonical SHA-256 checksum.
- The 36 intents cover broad and specialized roles, task paraphrases, changing
  preferences, manager-versus-IC intent, hard geography/work-policy/experience
  constraints, ambiguous requests, nontechnical distractors, and two intended
  no-suitable-job cases.
- Labels are generated from explicit per-intent description/title criteria and
  explicitly declared structured hard constraints. They were generated before
  any ranking evaluation and import neither production query parsing nor the
  confidence policy. Each of the 3,600 decisions is one of
  direct, adjacent, irrelevant, or forbidden.
- Labels are deliberately marked `automated_provisional_not_human_verified`.
  The Markdown review export includes intent notes and evidence snippets for
  every non-irrelevant decision. Human corrections must create a new version;
  they must not mutate the frozen provisional file.

### Evaluation infrastructure

- Added a guarded evaluation runner for lexical-only production retrieval,
  hybrid PostgreSQL retrieval, and exhaustive corpus reranking.
- The runner refuses database names that do not contain `holdout`, `eval`, or
  `test`, and refuses an evaluation database containing unrelated active jobs.
- Document and query embeddings share a persistent, task-aware cache keyed by
  model, task type, and text hash. Provider batch calls, provider prediction
  requests, and cache hits are recorded.
- Metrics include candidate recall, final precision/recall/NDCG,
  recommendation precision/recall/coverage, abstention, forbidden-result rate,
  hard-constraint violations, p50/p95 latency, and embedding usage.
- `PostgresStore.explain_candidate_jobs` executes the exact candidate SQL under
  `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` and reports only indexes named by
  executed plan nodes. It does not infer usage from index existence.

### Issues encountered and resolution

- The environment supplied `E:\mnt\data\matcheval`, but that path did not exist;
  the live repository was `E:\boardforge`. Work continued on the verified
  `feat/boardforge-v2` checkout.
- System Python lacked project dependencies. All generation and tests were
  rerun with the repository `.venv`.
- The first category classifier searched entire descriptions, so corporate
  boilerplate made sales roles look like AI jobs. The classifier was corrected
  to prioritize role-bearing title evidence and use a restricted technical-only
  description fallback. The corpus was recollected once before labels were
  created or evaluation was run.
- Managed filesystem permissions initially blocked generated labels and the
  test SQLite database. The same commands succeeded with scoped repository
  write permission; application code was not changed to work around the
  environment.
- A real local `boardforge_holdout_eval` database was created separately from
  the application database. With the 100 frozen rows and synthetic (non-paid)
  vectors, PostgreSQL used `idx_jobs_status` and `job_embeddings_pkey`; it did
  **not** choose the GIN or HNSW indexes. This is the honest result for this
  small corpus/current query shape, not an index-usage claim. The full runner
  preserves the raw plans so the paid evaluation can recheck the actual plans.

### Verification and stop condition

- Frozen corpus checksum:
  `d1f7a5bdbf65dceea31b6c5f67143a4c633c20fdaff88b5748b4837524e139e3`.
- Provisional-label checksum:
  `95166916abf9fcc1d221df3f94e06bde8c4a65dd952b98872305b4d1e31bb020`.
- Final full suite with a dedicated `boardforge_test` pgvector database: 130
  passed with one existing Starlette deprecation warning. The production
  EXPLAIN path was additionally exercised manually against the isolated local
  holdout database.
- Planned paid usage with an empty cache: 100 document + 36 query = **136
  Vertex prediction requests**. No Vertex evaluation was executed, in
  accordance with the requested stop point.
- Remaining human action: review `docs/holdout-v1-label-review.md`, record
  corrections in a new human-verified label version, approve the 136-request
  Vertex run, and then interpret retrieval/ranking metrics without tuning
  `confidence.py` against this holdout.

## 2026-11-24 — Deterministic ranking reliability milestone

### Requested outcome

Reduce false confident recommendations without damaging retrieval recall. The
starting frozen India evaluation contained 29 jobs and 24 queries (dataset
v1.1): Top-1 1.0, Recall@10 0.9315, NDCG@10 0.9406, recommended-irrelevant rate
0.3042, and hard-constraint violation rate 0.0.

### Problems found

1. Confidence enforcement lived in `hybrid_rank_jobs`, after the query embedding
   succeeded. Lexical-only searches, missing document vectors, disabled
   embeddings, and provider outages therefore used weaker confidence rules.
2. Query evidence used a set intersection over the entire description. Words
   from unrelated paragraphs could combine into apparently strong evidence.
3. The query embedding was requested in `retrieve_and_rank_jobs`; after failure,
   `hybrid_rank_jobs` attempted the same request again.
4. Role-only filtering removed potentially useful uncertain candidates instead
   of retaining them with an explicit verification status.
5. Evaluation measured false recommendations but did not expose the precision,
   recall, coverage, or abstention trade-off of the confidence policy.

### Design decisions

- Confidence is a deterministic policy over structured job/query evidence. An
  embedding may change ordering, but it may not upgrade unsupported relevance.
- Query intent is separated into role, domain, skill, and task evidence.
- Description evidence must co-occur inside a short token window. Scattered
  terms across a long JD count as retrieval hints, not verification.
- Broad words such as “platform”, “data”, and “services” cannot establish
  specific intent when the distinctive query terms are absent.
- Leadership and individual-contributor titles are checked independently of
  semantic similarity. “Engineering manager” and “software engineer” are not
  treated as interchangeable.
- Uncertain candidates remain in the result set as `needs_verification`. This
  separates retrieval recall from recommendation precision instead of hiding
  ambiguity by deleting results.
- No company, job ID, or evaluation-case-specific rule is used. The policy uses
  reusable occupational categories and evidence geometry.

### Implementation

- Added `boardforge/confidence.py` as the shared policy and intent assessment
  module.
- Applied the policy inside deterministic `rank_jobs`, before any semantic
  branch can return.
- Removed the semantic-only confidence implementation.
- Added a `query_embedding_attempted` signal so a failed initial request cannot
  be retried accidentally during hybrid rescoring.
- Added categorical `query_domain`, `query_skill`, and `query_task` evidence.
- Added recommended precision/recall, recommendation coverage, abstention rate,
  and no-confident-recommendation count/rate to the real evaluation report.

### Issues encountered and resolution

- Existing tests expected role-mismatched jobs to disappear. The milestone
  requires preserving *relevant* uncertain candidates, not retaining candidates
  with no evidence for an explicitly requested role. An initial implementation
  kept all role mismatches and degraded exact role searches and benchmark
  precision. The explicit role gate was restored; uncertainty is preserved
  inside the plausible role/domain candidate set.
- A salary-confidence test had one candidate with verified salary but no query
  evidence and another with query evidence but unknown salary. Both are now
  correctly uncertain for different reasons; the test checks preservation and
  status rather than forcing a misleading winner.
- A dedicated real-eval fixture was added where a labeled relevant candidate is
  retained as uncertain. It verifies zero recommendation coverage, full
  abstention, and one query with no confident recommendation without confusing
  that behavior with a clear explicit-role mismatch.
- The first evaluation command passed the corpus as a positional argument, but
  the CLI requires `--corpus`. Argument parsing stopped before provider setup,
  so no paid request occurred. The corrected command was then run once.

### Verification

- Focused confidence and evaluation regressions: 38 passed during development.
- Final full suite: 124 passed; one existing Starlette `TestClient` deprecation
  warning remains.
- One Vertex run was made over the frozen 29-job/24-query corpus. No discovery
  or bulk index backfill was run.

| Metric | Before | After |
|---|---:|---:|
| Top-1 accuracy | 1.0000 | 1.0000 |
| Top-1 direct accuracy | 0.9583 | 1.0000 |
| Recall@10 | 0.9315 | 0.9315 |
| NDCG@10 | 0.9406 | 0.9366 |
| Graded NDCG@10 | 0.9375 | 0.9408 |
| Recommended irrelevant rate | 0.3042 | 0.1542 |
| Hard-constraint violation rate | 0.0000 | 0.0000 |

New confidence metrics: recommended precision 0.8458, recommended recall
0.7755, recommendation coverage 0.4412, abstention rate 0.5588, and 0/24
queries with no confident recommendation.

The recommended-irrelevant rate fell by 49.3% relative while retrieval recall
and Top-1 accuracy were preserved. Binary NDCG decreased by 0.0040, while graded
NDCG and direct Top-1 accuracy improved.

### Remaining measured failure clusters

- Distinctive domain terms can still be outweighed by broad words such as
  “platform”, “services”, and “operations” in security and data queries.
- Some title families remain too close: account management versus support
  engineering, task-focused IC work versus management, and enterprise
  architecture versus implementation.
- Eight queries still contained at least one irrelevant confident
  recommendation: biometrics data pipelines, data engineer/Airflow, Segment
  data pipelines, storage/streaming platform, security platform backend,
  security incident response, programmable messaging support, and technical
  account management.
- Four additional failure rows contained evaluation-forbidden retrieved
  candidates but no false confident recommendation. These are retrieval/ranking
  issues, not hard-constraint violations.

A stricter distinctive-domain/title-family pass was prototyped locally after
the paid run, but deliberately not retained: changing status can change ranking,
so shipping it without another frozen Vertex evaluation would make the reported
metrics inaccurate. It is recorded here as the next measured iteration rather
than presented as an unverified improvement.

### Remaining external work

- A real Vertex evaluation requires Application Default Credentials, the GCP
  project, and paid API access. It should be run once after local tests pass to
  avoid repeated paid calls.
