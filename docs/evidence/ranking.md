# BoardForge ranking benchmark

Dataset: v2.0 — 120 cases / 26 jobs

| Variant | P@5 | Recall@5 | NDCG@5 | Forbidden | Top-1 | Empty |
|---|---:|---:|---:|---:|---:|---:|
| lexical_only | 0.330 | 0.671 | 0.599 | 0.217 | 0.500 | 0.000 |
| constrained_lexical | 0.915 | 0.932 | 0.938 | 0.017 | 0.900 | 0.000 |
| full_evidence | 0.915 | 0.932 | 0.941 | 0.017 | 0.908 | 0.000 |
| full_evidence_feedback | 0.932 | 0.943 | 0.952 | 0.000 | 0.925 | 0.000 |

> Synthetic adversarial benchmark. Useful for regression and ablation evidence; not a substitute for real-user relevance labels.
