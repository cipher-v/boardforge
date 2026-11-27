# BoardForge controlled holdout evaluation

> Labels: **AI-audited, not human-verified**.

- Corpus jobs: 100; queries: 36 (32 with relevant jobs, 4 with none)
- Corpus checksum: `d1f7a5bdbf65dceea31b6c5f67143a4c633c20fdaff88b5748b4837524e139e3`
- Label checksum: `c8cfc4ea4043bf9db48820c26efac3d63488231ca6664f3afd7688954b9b843c`

## Queries with relevant jobs

| Metric | Lexical | Hybrid PostgreSQL | Exhaustive reference |
|---|---:|---:|---:|
| Candidate recall | 0.6681 | 0.9437 | 1.0 |
| Precision@10 | 0.2406 | 0.3312 | 0.325 |
| Recall@10 | 0.3791 | 0.5607 | 0.5528 |
| NDCG@10 | 0.421 | 0.5771 | 0.572 |
| Graded NDCG@10 | 0.4192 | 0.5857 | 0.5825 |
| Recommended precision | 0.4087 | 0.4789 | 0.4626 |
| Recommended recall | 0.2035 | 0.2944 | 0.2944 |
| Recommendation coverage | 0.5044 | 0.5591 | 0.5787 |
| Abstention rate | 0.4956 | 0.4409 | 0.4213 |
| Recommended irrelevant rate | 0.5913 | 0.5211 | 0.5374 |
| Forbidden-result rate | 0.0219 | 0.0236 | 0.0236 |
| Hard-constraint violation rate | 0.0 | 0.0 | 0.0 |
| Warm latency p50 (ms) | 73.09 | 143.75 | 150.85 |
| Warm latency p95 (ms) | 118.6 | 328.16 | 546.74 |

## Queries with no relevant jobs

Recall and NDCG are N/A and are excluded from conventional aggregates.

| Metric | Lexical | Hybrid PostgreSQL | Exhaustive reference |
|---|---:|---:|---:|
| False recommendations | 9 | 8 | 11 |
| Queries with false recommendations | 2 | 2 | 2 |
| Recommendation coverage | 0.225 | 0.2 | 0.275 |
| Abstention rate | 0.775 | 0.8 | 0.725 |
| Recommended irrelevant rate | 1.0 | 1.0 | 1.0 |
| Forbidden-result rate | 0.0 | 0.025 | 0.025 |
| Hard-constraint violation rate | 0.0 | 0.0 | 0.0 |
| Warm latency p50 (ms) | 109.52 | 257.44 | 508.78 |
| Warm latency p95 (ms) | 128.9 | 268.27 | 580.18 |

## Embeddings and latency phases

- Vertex prediction texts: 136 (100 documents + 36 queries).
- Actual HTTP attempts: 136; additional retry attempts: 0.
- Embedding-cache hits: 36.
- Cold document embedding: 27827.86 ms; cold query embedding: 12683.6 ms; PostgreSQL embedding writes: 5733.47 ms.
- Warm retrieval p50/p95 values exclude embedding creation and EXPLAIN overhead.

## Actual PostgreSQL plans

- **lexical**: 36 queries explained; GIN used by 0; HNSW used by 0; indexes observed: `{}`.
- **hybrid**: 36 queries explained; GIN used by 0; HNSW used by 0; indexes observed: `{"job_embeddings_pkey": 1, "jobs_pkey": 25}`.

## False recommendations on no-relevant-result queries

- **lexical / technical_recruiter**: `ashby:notion:3979a6f5-2abb-447b-bd3a-0565b06bae24`, `ashby:notion:e9bf7ffa-9cf2-4360-a2e4-e9fa66859927`, `lever:palantir:10dfc8bc-99ad-4ca2-ab76-853cb90a92c2`, `ashby:notion:3fba1c39-c5cb-47d7-9ad2-1cec4d7e9d0c`, `lever:palantir:ab7e3425-81d5-4705-a7b5-cd60c8a45cdb`, `ashby:notion:30a5c739-185f-4a55-8a63-85974d687444`, `greenhouse:figma:6233040004`, `greenhouse:figma:5707966004`
- **lexical / legal_counsel**: `ashby:perplexity:fb1815de-07d9-4933-8746-979dbcb00870`
- **hybrid / technical_recruiter**: `lever:palantir:ab7e3425-81d5-4705-a7b5-cd60c8a45cdb`, `lever:palantir:10dfc8bc-99ad-4ca2-ab76-853cb90a92c2`, `ashby:notion:3fba1c39-c5cb-47d7-9ad2-1cec4d7e9d0c`, `ashby:notion:3979a6f5-2abb-447b-bd3a-0565b06bae24`, `ashby:notion:e9bf7ffa-9cf2-4360-a2e4-e9fa66859927`, `greenhouse:cloudflare:8402154`, `ashby:notion:30a5c739-185f-4a55-8a63-85974d687444`
- **hybrid / legal_counsel**: `ashby:perplexity:fb1815de-07d9-4933-8746-979dbcb00870`
- **exhaustive / technical_recruiter**: `lever:palantir:ab7e3425-81d5-4705-a7b5-cd60c8a45cdb`, `lever:palantir:10dfc8bc-99ad-4ca2-ab76-853cb90a92c2`, `ashby:notion:3fba1c39-c5cb-47d7-9ad2-1cec4d7e9d0c`, `ashby:notion:3979a6f5-2abb-447b-bd3a-0565b06bae24`, `ashby:notion:e9bf7ffa-9cf2-4360-a2e4-e9fa66859927`, `ashby:notion:4967b1cf-a1f7-417b-b650-feebbff767a3`, `ashby:notion:f1f9e19d-cbf3-49eb-9824-d04adf2e3d75`, `greenhouse:cloudflare:8402154`, `ashby:notion:11e56782-ee28-4025-ba0e-76222b0bfcea`, `ashby:notion:59f2246d-9cb5-4e97-879e-46c902dc276a`
- **exhaustive / legal_counsel**: `ashby:perplexity:fb1815de-07d9-4933-8746-979dbcb00870`

## Unresolved eligibility

- `india_backend_only` / `greenhouse:cloudflare:8401112` (direct): Board location is only 'Hybrid'; India eligibility is not stated. Outcomes: `{"exhaustive": {"recommended": false, "retrieved": false}, "hybrid": {"recommended": false, "retrieved": false}, "lexical": {"recommended": false, "retrieved": false}}`
- `india_backend_only` / `greenhouse:cloudflare:8105716` (direct): Board location is only 'Hybrid'; India eligibility is not stated. Outcomes: `{"exhaustive": {"recommended": false, "retrieved": false}, "hybrid": {"recommended": false, "retrieved": false}, "lexical": {"recommended": false, "retrieved": false}}`
- `remote_support_only` / `greenhouse:figma:6104028004` (adjacent): The posting location is São Paulo but the work policy is not stated. Outcomes: `{"exhaustive": {"recommended": false, "retrieved": false}, "hybrid": {"recommended": false, "retrieved": false}, "lexical": {"recommended": false, "retrieved": false}}`
- `junior_max_two_years` / `greenhouse:cloudflare:8401112` (adjacent): No numeric experience requirement is stated in structured data. Outcomes: `{"exhaustive": {"recommended": false, "retrieved": false}, "hybrid": {"recommended": false, "retrieved": false}, "lexical": {"recommended": false, "retrieved": false}}`
- `junior_max_two_years` / `lever:palantir:10dfc8bc-99ad-4ca2-ab76-853cb90a92c2` (adjacent): No numeric experience requirement is stated. Outcomes: `{"exhaustive": {"recommended": false, "retrieved": true}, "hybrid": {"recommended": false, "retrieved": true}, "lexical": {"recommended": false, "retrieved": true}}`
- `junior_max_two_years` / `lever:palantir:ab7e3425-81d5-4705-a7b5-cd60c8a45cdb` (adjacent): No numeric experience requirement is stated. Outcomes: `{"exhaustive": {"recommended": false, "retrieved": true}, "hybrid": {"recommended": false, "retrieved": true}, "lexical": {"recommended": false, "retrieved": true}}`
- `junior_max_two_years` / `ashby:perplexity:94ccf41e-d3e1-41aa-9569-c3bcbffc4184` (adjacent): No numeric experience requirement is stated. Outcomes: `{"exhaustive": {"recommended": false, "retrieved": true}, "hybrid": {"recommended": false, "retrieved": true}, "lexical": {"recommended": false, "retrieved": true}}`

Complete per-query metrics and raw `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` plans are stored in the machine-readable JSON report.
