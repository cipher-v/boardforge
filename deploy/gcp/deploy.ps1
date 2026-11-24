param(
    [Parameter(Mandatory = $true)] [string] $ProjectId,
    [string] $Region = "us-central1",
    [string] $ServiceName = "boardforge-api",
    [string] $JobName = "boardforge-maintenance",
    [string] $Repository = "boardforge",
    [string] $ServiceAccountName = "boardforge-runtime",
    [string] $DatabaseSecret = "boardforge-database-url",
    [string] $TavilySecret = "boardforge-tavily-key",
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

$runtimeEnv = "BOARDFORGE_SEARCH_PROVIDER=tavily,BOARDFORGE_ENABLE_EMBEDDINGS=1,BOARDFORGE_EMBEDDING_PROVIDER=vertex,BOARDFORGE_EMBEDDING_MODEL=gemini-embedding-001,BOARDFORGE_EMBEDDING_DIMENSIONS=768,GOOGLE_CLOUD_PROJECT=$ProjectId,GOOGLE_CLOUD_LOCATION=$Region"
$secrets = "DATABASE_URL=$DatabaseSecret`:latest,TAVILY_API_KEY=$TavilySecret`:latest"
$access = if ($PublicApi) { "--allow-unauthenticated" } else { "--no-allow-unauthenticated" }

gcloud run deploy $ServiceName --image $image --region $Region --project $ProjectId --service-account $serviceAccount --port 8080 --set-env-vars $runtimeEnv --set-secrets $secrets $access

gcloud run jobs deploy $JobName --image $image --region $Region --project $ProjectId --service-account $serviceAccount --command python --args=-m,boardforge.worker,--once --set-env-vars $runtimeEnv --set-secrets $secrets --max-retries 3 --task-timeout 30m

Write-Host "Deployed $ServiceName and $JobName. Run maintenance with:"
Write-Host "gcloud run jobs execute $JobName --region $Region --project $ProjectId --wait"
