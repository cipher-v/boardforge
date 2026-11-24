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
- exact tool-sequence accuracy
- tool-set precision / recall / F1
- unnecessary tool calls
- missing preference updates
- missing feedback writes

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
