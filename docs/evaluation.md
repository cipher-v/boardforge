# Evaluation plan

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

The benchmark dataset should include adversarial cases: negation, conflicting preferences, missing locations, remote ambiguity, experience ranges, unpaid internships, stale jobs and reposts.
