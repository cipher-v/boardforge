# Evaluation

BoardForge evaluates the system at four layers.

## Ingestion
- connector success rate
- normalized field completeness
- duplicate rate
- active-to-closed transition accuracy
- sync latency

## Ranking
- Precision@K
- Recall@K
- NDCG@K
- hard-constraint violation rate
- rejection-memory compliance

## Stability
- Jaccard@K across paraphrases
- ranking drift after preference updates
- deterministic regression cases

## Explanations
- evidence coverage
- unsupported claim rate
- missing-metadata warnings

## Executable starter benchmark

```bash
python -m boardforge.cli benchmark
```

The versioned `boardforge/data/benchmark.json` fixture currently covers six deterministic ranking cases. It is intentionally a regression baseline, not statistically meaningful product evidence.

## Expansion plan

The production-facing benchmark should include at least 100 labeled profiles plus adversarial cases: negation, conflicting preferences, missing locations, remote ambiguity, experience ranges, salary normalization, unpaid internships, stale jobs and reposts. Agent evaluation will separately score tool selection, argument correctness and unnecessary tool calls.
