# BoardForge engineering log

This is the running implementation journal for substantial BoardForge tasks.
Future work should add a dated section describing the request, problems found,
decisions, changes, verification, and anything still requiring human action.

The goal is to preserve both the technical reasoning and a plain-language
explanation of why each change was made. It is not a changelog: routine file
lists belong in Git history; this document records engineering evidence and
trade-offs.

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
