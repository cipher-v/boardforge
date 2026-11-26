param(
    [Parameter(Mandatory = $true)] [string] $ProjectId,
    [string] $Region = "us-central1",
    [string] $ServiceName = "boardforge-api",
    [string] $JobName = "boardforge-maintenance",
    [string] $Repository = "boardforge",
    [string] $ServiceAccountName = "boardforge-runtime",
    [string] $DatabaseSecret = "boardforge-database-url",
    [string] $TavilySecret = "boardforge-tavily-key",
    [string] $RelevanceModel = "gemini-3.8-flash",
    [string] $RelevanceLocation = "global",
    [bool] $EnableEmbeddings = $true,
    [bool] $EnableRelevanceVerifier = $true,
    [bool] $EnableIntelligentAgent = $true,
    [switch] $EnableScheduledMaintenance,
    [string] $MaintenanceSchedule = "0 */12 * * *",
    [string] $MaintenanceTimeZone = "Etc/UTC",
    [string] $SchedulerJobName = "",
    [int] $StaleJobMaxAgeDays = 30,
    [string] $QuarantinedSources = "lever:jobtwine",
    [switch] $PublicApi
)

$ErrorActionPreference = "Stop"
$serviceAccount = "$ServiceAccountName@$ProjectId.iam.gserviceaccount.com"
$image = "$Region-docker.pkg.dev/$ProjectId/$Repository/boardforge:latest"

gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com aiplatform.googleapis.com secretmanager.googleapis.com --project $ProjectId

gcloud artifacts repositories describe $Repository --location $Region --project $ProjectId 2>$null
if ($LASTEXITCODE -ne 0) {
    gcloud artifacts repositories create $Repository --repository-format docker --location $Region --project $ProjectId
}

gcloud iam service-accounts describe $serviceAccount --project $ProjectId 2>$null
if ($LASTEXITCODE -ne 0) {
    gcloud iam service-accounts create $ServiceAccountName --display-name "BoardForge runtime" --project $ProjectId
}

gcloud projects add-iam-policy-binding $ProjectId --member "serviceAccount:$serviceAccount" --role roles/aiplatform.user --condition=None
gcloud projects add-iam-policy-binding $ProjectId --member "serviceAccount:$serviceAccount" --role roles/secretmanager.secretAccessor --condition=None

gcloud builds submit --tag $image --project $ProjectId

$embeddingsFlag = if ($EnableEmbeddings) { "1" } else { "0" }
$relevanceFlag = if ($EnableRelevanceVerifier) { "1" } else { "0" }
$agentFlag = if ($EnableIntelligentAgent) { "1" } else { "0" }

$workerRuntimeEnv = @(
    "BOARDFORGE_SEARCH_PROVIDER=tavily",
    "BOARDFORGE_DISCOVERY_MAX_BOARDS=6",
    "BOARDFORGE_DISCOVERY_MAX_RESULTS=15",
    "BOARDFORGE_DISCOVERY_SYNC_CONCURRENCY=3",
    "BOARDFORGE_DISCOVERY_REQUESTS_PER_CYCLE=5",
    "BOARDFORGE_SOURCE_SYNC_LIMIT=10",
    "BOARDFORGE_SOURCE_SYNC_CONCURRENCY=4",
    "BOARDFORGE_STALE_JOB_MAX_AGE_DAYS=$StaleJobMaxAgeDays",
    "BOARDFORGE_QUARANTINED_SOURCES=$QuarantinedSources",
    "BOARDFORGE_ENABLE_EMBEDDINGS=$embeddingsFlag",
    "BOARDFORGE_EMBEDDING_PROVIDER=vertex",
    "BOARDFORGE_EMBEDDING_MODEL=gemini-embedding-001",
    "BOARDFORGE_EMBEDDING_DIMENSIONS=768",
    "BOARDFORGE_EMBEDDING_BATCH_SIZE=32",
    "BOARDFORGE_EMBEDDING_JOBS_PER_CYCLE=64",
    "GOOGLE_CLOUD_PROJECT=$ProjectId",
    "GOOGLE_CLOUD_LOCATION=$Region",
    "BOARDFORGE_CUMULATIVE_SPEND_LIMIT_USD=2.0",
    "BOARDFORGE_TAVILY_DAILY_LIMIT=900"
) -join ","

$apiRuntimeEnv = @(
    $workerRuntimeEnv,
    "BOARDFORGE_DISCOVERY_DISPATCH_MODE=cloud_run_job",
    "BOARDFORGE_MAINTENANCE_JOB=$JobName",
    "BOARDFORGE_JOB_DISPATCH_COOLDOWN_SECONDS=15",
    "BOARDFORGE_ENABLE_LLM_PREFERENCES=0",
    "BOARDFORGE_ENABLE_RELEVANCE_VERIFIER=$relevanceFlag",
    "BOARDFORGE_RELEVANCE_PROVIDER=vertex",
    "BOARDFORGE_RELEVANCE_MODEL=$RelevanceModel",
    "BOARDFORGE_RELEVANCE_VERTEX_LOCATION=$RelevanceLocation",
    "BOARDFORGE_RELEVANCE_INPUT_USD_PER_MILLION=0.75",
    "BOARDFORGE_RELEVANCE_OUTPUT_USD_PER_MILLION=3.75",
    "BOARDFORGE_RELEVANCE_MAX_CANDIDATES=12",
    "BOARDFORGE_RELEVANCE_BATCH_SIZE=8",
    "BOARDFORGE_RELEVANCE_MAX_RETRIES=2",
    "BOARDFORGE_RELEVANCE_CONCURRENCY=2",
    "BOARDFORGE_RELEVANCE_MAX_OUTPUT_TOKENS=1536",
    "BOARDFORGE_RELEVANCE_LOW_THINKING_HEADROOM_TOKENS=256",
    "BOARDFORGE_RELEVANCE_THINKING_LEVEL=LOW",
    "BOARDFORGE_RELEVANCE_THINKING_BUDGET=128",
    "BOARDFORGE_RELEVANCE_MAX_CALLS_PER_SEARCH=3",
    "BOARDFORGE_RELEVANCE_MAX_TOKENS_PER_SEARCH=60000",
    "BOARDFORGE_ENABLE_INTELLIGENT_AGENT=$agentFlag",
    "BOARDFORGE_AGENT_PROVIDER=vertex",
    "BOARDFORGE_AGENT_MAX_ITERATIONS=4",
    "BOARDFORGE_AGENT_MAX_TOOL_CALLS=8",
    "BOARDFORGE_AGENT_MIN_SUFFICIENT_RESULTS=3",
    "BOARDFORGE_LLM_CUMULATIVE_COST_LIMIT_USD=2.0"
) -join ","

$secrets = "DATABASE_URL=$DatabaseSecret`:latest,TAVILY_API_KEY=$TavilySecret`:latest"
$access = if ($PublicApi) { "--allow-unauthenticated" } else { "--no-allow-unauthenticated" }

gcloud run jobs deploy $JobName --image $image --region $Region --project $ProjectId --service-account $serviceAccount --command python --args=-m,boardforge.worker,--once --parallelism 1 --tasks 1 --set-env-vars $workerRuntimeEnv --set-secrets $secrets --max-retries 2 --task-timeout 15m

# Grant least-privilege job execution permission on the maintenance job so the API service can dispatch it on demand.
gcloud run jobs add-iam-policy-binding $JobName --region $Region --project $ProjectId --member "serviceAccount:$serviceAccount" --role roles/run.developer

gcloud run deploy $ServiceName --image $image --region $Region --project $ProjectId --service-account $serviceAccount --port 8080 --set-env-vars $apiRuntimeEnv --set-secrets $secrets $access

if ($EnableScheduledMaintenance) {
    $resolvedSchedulerJob = if ($SchedulerJobName) { $SchedulerJobName } else { "$JobName-12h" }
    $jobRunUri = "https://run.googleapis.com/v2/projects/$ProjectId/locations/$Region/jobs/$JobName`:run"
    gcloud services enable cloudscheduler.googleapis.com --project $ProjectId
    gcloud scheduler jobs describe $resolvedSchedulerJob --location $Region --project $ProjectId 2>$null
    if ($LASTEXITCODE -eq 0) {
        gcloud scheduler jobs update http $resolvedSchedulerJob --location $Region --project $ProjectId --schedule $MaintenanceSchedule --time-zone $MaintenanceTimeZone --uri $jobRunUri --http-method POST --oauth-service-account-email $serviceAccount --oauth-token-scope "https://www.googleapis.com/auth/cloud-platform"
    } else {
        gcloud scheduler jobs create http $resolvedSchedulerJob --location $Region --project $ProjectId --schedule $MaintenanceSchedule --time-zone $MaintenanceTimeZone --uri $jobRunUri --http-method POST --oauth-service-account-email $serviceAccount --oauth-token-scope "https://www.googleapis.com/auth/cloud-platform" --description "Invoke BoardForge one-shot maintenance job every 12 hours"
    }
    Write-Host "Configured optional Cloud Scheduler job $resolvedSchedulerJob ($MaintenanceSchedule $MaintenanceTimeZone) -> $JobName."
}

Write-Host "Deployed $ServiceName and on-demand Cloud Run Job $JobName."
Write-Host "Manual maintenance check: gcloud run jobs execute $JobName --region $Region --project $ProjectId --wait"
