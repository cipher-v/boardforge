# GCP deployment

BoardForge deploys as two Cloud Run resources sharing one image and service account:

- **`boardforge-api` (Cloud Run Service):** serves the web workspace, authenticated API, Vertex hybrid retrieval (`gemini-embedding-001`), Vertex relevance verification (`gemini-3.8-flash`), and bounded Vertex intelligent-agent orchestration.
- **`boardforge-maintenance` (one-shot Cloud Run Job):** drains queued Tavily discovery requests (`python -m boardforge.worker --once`), refreshes due official ATS boards, and backfills missing pgvector document embeddings. There is **no always-running worker** in production.

## Secrets

Create the required Secret Manager entries before deploying (never commit credentials to repository files):

```powershell
"postgresql://..." | gcloud secrets create boardforge-database-url --data-file=-
"tvly-..." | gcloud secrets create boardforge-tavily-key --data-file=-
```

## Deploy

```powershell
.\deploy\gcp\deploy.ps1 -ProjectId YOUR_PROJECT_ID -Region us-central1
```

Add `-PublicApi` only when exposing the web UI publicly; when `-PublicApi` is set, configure `BOARDFORGE_PROFILE_TOKENS` and `BOARDFORGE_ADMIN_TOKEN` secrets or front the service with Cloud Run IAM / IAP.

## Feature Gates & On-Demand Worker Dispatch

`deploy.ps1` explicitly configures Vertex as the single supported deployment provider:

- `BOARDFORGE_ENABLE_EMBEDDINGS=1` (`BOARDFORGE_EMBEDDING_PROVIDER=vertex`, `gemini-embedding-001`, 768-d)
- `BOARDFORGE_ENABLE_RELEVANCE_VERIFIER=1` (`BOARDFORGE_RELEVANCE_PROVIDER=vertex`, `gemini-3.8-flash`)
- `BOARDFORGE_ENABLE_INTELLIGENT_AGENT=1` (`BOARDFORGE_AGENT_PROVIDER=vertex`, `gemini-3.8-flash`)
- `BOARDFORGE_ENABLE_LLM_PREFERENCES=0` (OpenAI-dependent preference extraction stays disabled)
- `BOARDFORGE_DISCOVERY_DISPATCH_MODE=cloud_run_job` (`BOARDFORGE_MAINTENANCE_JOB=boardforge-maintenance`)

Any individual gate can be turned off at deploy time via `-EnableEmbeddings:$false`, `-EnableRelevanceVerifier:$false`, or `-EnableIntelligentAgent:$false`.

When a user clicks **Discover more opportunities** (`POST /discover/{profile_id}`) or an explicit conditional agent turn queues `discover_job_sources`, the API persists a deduplicated `'queued'` row in PostgreSQL and dispatches `boardforge-maintenance` on demand via the Cloud Run Jobs v2 `:run` API (scoped via `roles/run.developer` on `boardforge-maintenance` only, with a per-instance dispatch cooldown and `--parallelism 1 --tasks 1`).

## Optional 12-Hour Scheduled Maintenance (Cloud Scheduler)

To keep registered ATS boards fresh without an always-running worker, pass `-EnableScheduledMaintenance` to create or update an OAuth-authenticated Cloud Scheduler HTTP trigger (`0 */12 * * *` UTC by default) targeting the existing `boardforge-maintenance` Cloud Run Job `:run` endpoint:

```powershell
.\deploy\gcp\deploy.ps1 -ProjectId YOUR_PROJECT_ID -Region us-central1 -EnableScheduledMaintenance
```

Each scheduled run executes `python -m boardforge.worker --once`, refreshing due boards in least-recently-attempted order without starvation and skipping embedding generation for unchanged jobs. Unrefreshed active jobs older than `BOARDFORGE_STALE_JOB_MAX_AGE_DAYS` (default `30` days) are hidden from candidate search without deleting stored records.

You can also run a manual cycle at any time:

```powershell
gcloud run jobs execute boardforge-maintenance --region us-central1 --wait
```
