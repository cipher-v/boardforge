# Evaluation

BoardForge evaluates the system at six layers.

## Candidate profile extraction
- skill precision / recall / F1
- inferred-role precision / recall / F1
- evidence-snippet coverage
- candidate review and correction behavior

## Ingestion
- connector success rate
- normalized field completeness
- duplicate rate
- active-to-closed transition accuracy
- sync latency

## Ranking
- Precision@K
- precision among returned results
- Recall@K
- NDCG@K
- forbidden-result rate
- top-1 relevance
- rejection-memory compliance

## Stability
- paraphrase families across equivalent intents
- ranking drift after preference updates
- deterministic regression cases

## Explanations
- structured evidence coverage
- unsupported-claim risk
- missing-metadata warnings

## Agent behavior
- task completion and constraint preservation
- appropriate tool selection and unnecessary calls
- bounded stopping behavior and state correctness
- missing or unauthorized preference and feedback writes
- provider calls, token use, latency and estimated cost

The deterministic benchmark and the production fallback share the same planner. Integration tests execute planned tools against a temporary store and verify preference and feedback state changes, not only tool names.

## Ranking benchmark

```bash
python -m boardforge.cli benchmark
```

The default dataset is `boardforge/data/benchmark_v2.json`.

It contains **120 adversarial ranking cases** built from **24 failure families × 5 paraphrases**. Families cover:
- India vs non-India geography
- unknown geography
- location restrictions
- remote-only constraints
- salary floors
- experience ceilings
- unpaid roles
- hardware avoidance
- rejection memory
- AI/ML, CV, backend, software, QA, data engineering and data science role families
- mixed-role preferences

The dataset is synthetic and deterministic. It is designed for regression, ablation and failure analysis, not as a substitute for real-user relevance labels.

## Ablation / comparison runner

```bash
python -m boardforge.cli compare --markdown-out results/ranking.md
```

This compares:

1. `lexical_only`
2. `constrained_lexical`
3. `full_evidence`
4. `full_evidence_feedback`

This makes the value of hard constraints, structured role evidence and rejection memory measurable rather than anecdotal.

Optional embedding comparison:

```bash
set BOARDFORGE_ENABLE_EMBEDDINGS=1
set OPENAI_API_KEY=...
python -m boardforge.cli compare-embeddings
```

Embedding evaluation is opt-in because it makes external model calls.

## Agent tool-selection benchmark

```bash
python -m boardforge.cli agent-eval
```

The packaged `boardforge/data/agent_tools.json` dataset contains **40 tool-selection cases** covering:
- search-only requests
- preference updates
- feedback writes
- preference update + search
- feedback + search
- natural conversational variants

The deterministic policy is the production no-key fallback as well as the baseline, preventing evaluation/production planner drift.

Optional LLM planner evaluation:

```bash
set OPENAI_API_KEY=...
python -m boardforge.cli agent-eval-llm
```

The LLM evaluation is a planning-only benchmark: it selects tools without executing them, so it does not mutate user state. Production tool-loop behavior is tested separately.

The intelligent-agent milestone adds the frozen mocked manifest
`boardforge/data/agent_scenarios_v1.json`. Its twelve deterministic scenarios
cover straightforward and conditional searches, explicit preference changes,
insufficient results and discovery, negative feedback, contradictory constraints,
no suitable jobs, provider failure, budget exhaustion, prompt injection, and
repeated calls. Tests execute typed tools against isolated stores and score task
completion, constraints, tool choice, unnecessary calls, stopping, state, and
resource accounting. This suite uses no live model, embedding, discovery, or ATS
requests; a controlled live-agent evaluation remains separately authorized.

## Preference extraction benchmark

```bash
python -m boardforge.cli preference-eval
```

The packaged 20-case suite checks natural-language additions, removals, negation, aliases, geography changes, salary relaxation, experience ceilings and work-policy changes. It compares extracted preference fields against versioned expected state.

## Resume profile extraction benchmark

```bash
python -m boardforge.cli resume-eval
```

The packaged versioned suite measures skill and role extraction from synthetic
resume snippets. It is a deterministic regression set, not evidence of real-world
resume quality. The next step is a consented, de-identified labeled set containing
layout variation, ambiguous skills, career transitions and extraction corrections.

## Metric definitions

- `Precision@K` uses the conventional denominator `K`.
- `precision_among_returned` divides by the number of results actually returned.
- `Recall@K` may be below 1.0 when a case has more than K relevant jobs; interpret it alongside precision among returned and empty-result rate.

## Interpretation

High synthetic benchmark scores are useful only when paired with:
- explicit dataset versioning
- failure-family breakdowns
- real live-source syncs
- honest disclosure that the labels are synthetic
- later validation on real user feedback

The next evaluation milestone is a real-user labeled set collected from deployed BoardForge sessions.

## Independent production-retrieval holdout

The v1 holdout is separate from the 29-job India development benchmark:

- `boardforge/data/holdout_corpus_v1.json`: 100 frozen jobs from ten official
  Greenhouse, Lever, and Ashby boards.
- `boardforge/data/holdout_labels_v1_provisional.json`: 36 intents and complete
  four-way labels. These are automated provisional labels, not human truth.
- `docs/holdout-v1-label-review.md`: evidence export for human review.

The regex-provisional labels were subsequently audited semantically. Use
`boardforge/data/holdout_labels_v1_1_ai_audited.json` for review/evaluation and
read `docs/holdout-v1.1-ai-audit.md` for every correction and unresolved case.
The v1.1 dataset is **AI-audited, not human-verified**; the original provisional
file remains available for disagreement analysis.

Plan paid usage without contacting Vertex or PostgreSQL:

```bash
python -m boardforge.cli holdout-eval
```

The current empty-cache plan is 100 document plus 36 query embeddings (136
Vertex prediction requests). Lexical, hybrid, and exhaustive variants reuse the
same persistent cache.

After human label review, create a separate PostgreSQL database (for example
`boardforge_holdout_eval`) and explicitly execute:

```bash
python -m boardforge.cli holdout-eval \
  --labels boardforge/data/holdout_labels_v1_1_ai_audited.json \
  --execute-vertex \
  --database-url postgresql://boardforge:boardforge@localhost:5432/boardforge_holdout_eval
```

The database-name guard refuses the normal application database. The report
contains actual `EXPLAIN ANALYZE` plans and names GIN/HNSW only if PostgreSQL
used those indexes in the executed plan.

The controlled v1.1 run is stored locally in:

- `results/holdout-eval-v1_1-ai-audited.json` (complete per-query rows and raw
  execution plans), and
- `results/holdout-eval-v1_1-ai-audited.md` (readable comparison).

The run made 136 successful Vertex requests with no retries. Do not rerun it to
reproduce summary tables: regenerate the Markdown from the cached JSON rows.

## Optional LLM relevance-verifier comparison

The verifier evaluation consumes the saved hybrid per-query candidate sets, so
deterministic and LLM-verified paths begin with identical retrieval output. It
reports both precision deltas and recovery of relevant candidates that the
deterministic policy marked `needs_verification`.
The default command is plan-only and makes no provider call:

```bash
python -m boardforge.cli relevance-eval
```

Current empty-cache plan: 285 candidate judgments in 58 calls with batches of
up to eight, approximately 388,445 input and 34,200 output tokens (422,645
total), plus a maximum configured 7,424 thinking tokens. At the currently
configured Gemini 3.8 Flash introductory rates, the pre-run ceiling is about
$0.4474. The estimate is deliberately conservative and depends on the configured
description cap. It includes eligible recommended and uncertain rows; protected
eligibility rows are not sent.
The v1.1 labels remain **AI-audited, not human-verified**.

An authorized execution would require all of the following (do not run it for
planning):

```text
BOARDFORGE_RELEVANCE_PROVIDER=vertex
GOOGLE_CLOUD_PROJECT=boardforge-ai-2026
BOARDFORGE_RELEVANCE_VERTEX_LOCATION=global
BOARDFORGE_RELEVANCE_MODEL=gemini-3.8-flash
BOARDFORGE_RELEVANCE_BATCH_SIZE=8
BOARDFORGE_RELEVANCE_MAX_CANDIDATES=10
python -m boardforge.cli relevance-eval --execute --database-url <isolated-evaluation-postgres-url>
```

The evaluation cap is ten because it reuses the already frozen top-10 hybrid
pools. Production may use the separately configured 20-candidate pool.

One live eight-candidate smoke batch measured 8,505 input, 1,148 output and zero
thinking tokens. Scaling the plan by the observed input/output estimation error
produces a central full-run estimate of about 337,942 input and 40,898 output
tokens, or $0.4068. Allow roughly $0.45–$0.50 for candidate-mix variance and
thinking. This is measured extrapolation, not a guarantee; the full evaluation
has not been run.
