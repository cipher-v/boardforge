# GCP deployment

BoardForge deploys as two resources sharing the same image:

- a Cloud Run service for the API;
- a one-shot Cloud Run Job for discovery, source refresh, and embedding backfill.

Create two Secret Manager secrets before deployment:

```powershell
"postgresql://..." | gcloud secrets create boardforge-database-url --data-file=-
"tvly-..." | gcloud secrets create boardforge-tavily-key --data-file=-
```

Then deploy from the repository root:

```powershell
.\deploy\gcp\deploy.ps1 -ProjectId YOUR_PROJECT_ID -Region us-central1
```

Add `-PublicApi` only for an intentional public demo. If the API is public,
configure `BOARDFORGE_PROFILE_TOKENS` and `BOARDFORGE_ADMIN_TOKEN` as secrets or
use Cloud Run/IAP authentication.

The script creates a least-purpose runtime service account, grants Vertex AI
User and Secret Manager Secret Accessor, builds the image, and deploys both
resources. It does not create or modify the external PostgreSQL database.

For a manual maintenance run:

```powershell
gcloud run jobs execute boardforge-maintenance --region us-central1 --wait
```

In production, invoke that job from Cloud Scheduler at the desired interval.
The job is deliberately one-shot; the infinite local worker is intended for
Docker Compose, not a Cloud Run service.
