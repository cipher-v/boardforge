param(
    [Parameter(Mandatory = $true)] [string] $ProjectId,
    [string] $Region = "us-central1",
    [string] $ServiceName = "boardforge-api",
    [string] $JobName = "boardforge-maintenance",
    [string] $Repository = "boardforge",
    [string] $ServiceAccountName = "boardforge-runtime",
    [string] $DatabaseSecret = "boardforge-database-url",
    [string] $TavilySecret = "boardforge-tavily-key",
    [string] $RelevanceSecret = "boardforge-relevance-key",
    [ValidateSet("vertex", "groq")] [string] $RelevanceProvider = "vertex",
    [string] $RelevanceModel = "gemini-3.8-flash",
    [string] $RelevanceLocation = "global",
    [switch] $EnableRelevanceVerifier,
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

$relevanceEnabled = if ($EnableRelevanceVerifier) { "1" } else { "0" }
$workerRuntimeEnv = "BOARDFORGE_SEARCH_PROVIDER=tavily,BOARDFORGE_ENABLE_EMBEDDINGS=1,BOARDFORGE_EMBEDDING_PROVIDER=vertex,BOARDFORGE_EMBEDDING_MODEL=gemini-embedding-001,BOARDFORGE_EMBEDDING_DIMENSIONS=768,GOOGLE_CLOUD_PROJECT=$ProjectId,GOOGLE_CLOUD_LOCATION=$Region"
$apiRuntimeEnv = "$workerRuntimeEnv,BOARDFORGE_ENABLE_RELEVANCE_VERIFIER=$relevanceEnabled,BOARDFORGE_RELEVANCE_PROVIDER=$RelevanceProvider,BOARDFORGE_RELEVANCE_MODEL=$RelevanceModel,BOARDFORGE_RELEVANCE_VERTEX_LOCATION=$RelevanceLocation,BOARDFORGE_RELEVANCE_INPUT_USD_PER_MILLION=0.75,BOARDFORGE_RELEVANCE_OUTPUT_USD_PER_MILLION=3.75,BOARDFORGE_RELEVANCE_MAX_CANDIDATES=20,BOARDFORGE_RELEVANCE_BATCH_SIZE=8,BOARDFORGE_RELEVANCE_MAX_RETRIES=2,BOARDFORGE_RELEVANCE_CONCURRENCY=2,BOARDFORGE_RELEVANCE_MAX_OUTPUT_TOKENS=1536,BOARDFORGE_RELEVANCE_LOW_THINKING_HEADROOM_TOKENS=256,BOARDFORGE_RELEVANCE_THINKING_LEVEL=LOW,BOARDFORGE_RELEVANCE_THINKING_BUDGET=128,BOARDFORGE_RELEVANCE_MAX_CALLS_PER_SEARCH=3,BOARDFORGE_RELEVANCE_MAX_TOKENS_PER_SEARCH=60000,BOARDFORGE_LLM_CUMULATIVE_COST_LIMIT_USD=2.0,BOARDFORGE_CUMULATIVE_SPEND_LIMIT_USD=2.0"
$workerSecrets = "DATABASE_URL=$DatabaseSecret`:latest,TAVILY_API_KEY=$TavilySecret`:latest"
$apiSecrets = $workerSecrets
if ($EnableRelevanceVerifier -and $RelevanceProvider -eq "groq") {
    $apiSecrets = "$apiSecrets,BOARDFORGE_RELEVANCE_API_KEY=$RelevanceSecret`:latest"
    $apiRuntimeEnv = "$apiRuntimeEnv,BOARDFORGE_GROQ_RELEVANCE_MODEL=openai/gpt-oss-120b,BOARDFORGE_RELEVANCE_BATCH_SIZE=4,BOARDFORGE_GROQ_INPUT_USD_PER_MILLION=0.15,BOARDFORGE_GROQ_OUTPUT_USD_PER_MILLION=0.60,BOARDFORGE_GROQ_MAX_DESCRIPTION_CHARS=12000,BOARDFORGE_GROQ_MAX_OUTPUT_TOKENS=1280,BOARDFORGE_GROQ_MAX_REQUEST_TOKENS=7600,BOARDFORGE_GROQ_TPM_LIMIT=8000,BOARDFORGE_GROQ_TPD_LIMIT=200000"
}
$access = if ($PublicApi) { "--allow-unauthenticated" } else { "--no-allow-unauthenticated" }

gcloud run deploy $ServiceName --image $image --region $Region --project $ProjectId --service-account $serviceAccount --port 8080 --set-env-vars $apiRuntimeEnv --set-secrets $apiSecrets $access

gcloud run jobs deploy $JobName --image $image --region $Region --project $ProjectId --service-account $serviceAccount --command python --args=-m,boardforge.worker,--once --set-env-vars $workerRuntimeEnv --set-secrets $workerSecrets --max-retries 3 --task-timeout 30m

Write-Host "Deployed $ServiceName and $JobName. Run maintenance with:"
Write-Host "gcloud run jobs execute $JobName --region $Region --project $ProjectId --wait"
