# Measured evaluation results

## Controlled retrieval holdout

100 frozen postings from 10 official boards, 36 intents, 3,600 AI-audited (not human-verified) relevance judgments. 32 queries had relevant jobs; four did not.

| Metric | Lexical | Hybrid | Exhaustive |
|---|---:|---:|---:|
| Candidate recall | 0.6681 | 0.9437 | 1.0000 |
| Precision@10 | 0.2406 | 0.3312 | 0.3250 |
| Recall@10 | 0.3791 | 0.5607 | 0.5528 |
| NDCG@10 | 0.4210 | 0.5771 | 0.5720 |
| Graded NDCG@10 | 0.4192 | 0.5857 | 0.5825 |
| Warm latency p50 (ms) | 73.09 | 143.75 | 150.85 |
| Warm latency p95 (ms) | 118.60 | 328.16 | 546.74 |

Aggregates for 32 relevant-result queries. Warm latency excludes embedding generation, candidate-metric queries and EXPLAIN overhead. 136 Vertex embedding requests completed with zero retries. PostgreSQL did not select GIN or HNSW in the measured 100-job execution plans; this is not evidence of large-scale ANN performance. Seven eligibility cases remained unresolved.

## Relevance verifier comparison

All 36 queries completed with identical retrieval candidate sets.

| Metric | Deterministic hybrid | LLM verified |
|---|---:|---:|
| Candidate recall | 0.9437 | 0.9437 |
| Precision@10 | 0.3312 | 0.3281 |
| Recall@10 | 0.5607 | 0.5563 |
| NDCG@10 | 0.5771 | 0.6243 |
| Graded NDCG@10 | 0.5857 | 0.6438 |
| Recommended precision | 0.4789 | 0.9583 |
| Recommended recall | 0.2944 | 0.1991 |
| Recommended irrelevant rate | 0.5211 | 0.0417 |
| False recommendations | 74 | 2 |
| Hard-constraint violation rate | 0 | 0 |

32 relevant-result queries. The verifier recovered 13 of 38 recoverable relevant recommendations but reduced recommendation recall and coverage. On the four no-relevant-result queries, false recommendations decreased from eight to five; one query still had false recommendations and forbidden-result rate increased from 0.025 to 0.05. The offline verifier configuration differs from the tighter production beta configuration.

### Resource usage

57 provider calls and HTTP attempts; 486,532 input tokens, 20,403 output tokens, zero thinking tokens and retries; estimated model cost USD 0.44141025. Offline experiment costs are distinct from production budget accounting. The report's zero verifier latency placeholders are not measured latency.

## Synthetic ranking ablation

120 adversarial cases, 26 jobs.

| Variant | P@5 | Recall@5 | NDCG@5 | Forbidden |
|---|---:|---:|---:|---:|
| Lexical only | 0.330 | 0.671 | 0.599 | 0.217 |
| Constrained lexical | 0.915 | 0.932 | 0.938 | 0.017 |
| Full evidence | 0.915 | 0.932 | 0.941 | 0.017 |
| Full evidence + feedback | 0.932 | 0.943 | 0.952 | 0.000 |

Synthetic regression evidence, not real-user relevance measurement.

## Live agent experiments

November 25: optimized Vertex agent latency fell from 45.11 to 33.19 seconds, but the execution was incomplete due to the iteration limit. The subsequent experimental run failed before searching due to provider/model configuration. Neither counts as a successful end-to-end live-agent benchmark.

## Production validation

November 28 snapshot: 1,097 active jobs, 447 current embeddings. One manually triggered maintenance cycle synchronized 25 sources with zero source failures, fetched 1,008 postings, inserted 11, processed 16 embeddings and took 65.59 seconds of worker processing time. Scheduled execution had not yet been independently verified at this snapshot.

Recorded full regression: 333 passed, 4 skipped (before the welcome-panel change). Subsequent PostgreSQL integration run: five passed. Concurrent quota test: five accepted and fifteen denied requests at a temporary per-user limit of five. These are operational measurements, not external user adoption evidence.

## Evidence and interpretation

Local source reports: `results/holdout-eval-v1_1-ai-audited.json`, `results/holdout-eval-v1_1-ai-audited.md`, `results/relevance-verifier-eval-v3.json`, `results/ranking.md`, `results/agent-live-smoke-vertex-v3-report.md`, `results/agent-live-smoke-vertex-v4.json`. Raw reports remain local pending privacy review.

Candidate recall measures candidate-pool coverage, recommendation precision measures relevance among recommended results, and Precision@K uses K as its denominator. Human evaluation and real-world feedback remain necessary.

## Supporting artifacts

- [Controlled holdout evaluation report](evidence/holdout-eval-v1_1-ai-audited.md)
- [Synthetic ranking ablation report](evidence/ranking.md)
- [Frozen holdout corpus](../boardforge/data/holdout_corpus_v1.json)
- [AI-audited relevance labels](../boardforge/data/holdout_labels_v1_1_ai_audited.json)
