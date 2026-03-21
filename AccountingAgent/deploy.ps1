# Deploy to GCP Cloud Run from Windows
# Usage: .\deploy.ps1

$ErrorActionPreference = "Stop"

$PROJECT_ID = "ainm26osl-764"
$REGION = "europe-west1"
$SERVICE_NAME = "accounting-agent"
$IMAGE = "europe-west1-docker.pkg.dev/$PROJECT_ID/accounting-agent/app:latest"

Write-Host "=== Building and deploying to Cloud Run ===" -ForegroundColor Cyan

function Invoke-Gcloud {
  param([string[]]$Args)
  $oldPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  & gcloud @Args 2>$null
  $code = $LASTEXITCODE
  $ErrorActionPreference = $oldPreference
  return $code
}

# Enable APIs
Write-Host "Enabling GCP APIs..."
if ((Invoke-Gcloud @("services","enable","run.googleapis.com","artifactregistry.googleapis.com","aiplatform.googleapis.com","--project=$PROJECT_ID","--quiet")) -ne 0) {
  Write-Host "WARNING: gcloud services enable reported an error. Continuing..." -ForegroundColor Yellow
}

# Create Artifact Registry repo
Write-Host "Setting up Artifact Registry..."
Invoke-Gcloud @("artifacts","repositories","create","accounting-agent","--repository-format=docker","--location=$REGION","--project=$PROJECT_ID") | Out-Null
Invoke-Gcloud @("auth","configure-docker","$REGION-docker.pkg.dev","--quiet") | Out-Null

# Build and push
Write-Host "Building Docker image..." -ForegroundColor Yellow
docker build -t $IMAGE .
Write-Host "Pushing to Artifact Registry..." -ForegroundColor Yellow
docker push $IMAGE

# Deploy to Cloud Run
Write-Host "Deploying to Cloud Run..." -ForegroundColor Yellow
& gcloud run deploy $SERVICE_NAME `
  --image=$IMAGE `
  --region=$REGION `
  --project=$PROJECT_ID `
  --platform=managed `
  --allow-unauthenticated `
  --timeout=300 `
  --memory=1Gi `
  --cpu=2 `
  --concurrency=1 `
  --min-instances=1 `
  --max-instances=10 `
  --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID,GCP_REGION=$REGION" `
  --quiet

# Get URL
$URL = & gcloud run services describe $SERVICE_NAME --region=$REGION --project=$PROJECT_ID --format="value(status.url)"

Write-Host ""
Write-Host "=== Deployed! ===" -ForegroundColor Green
Write-Host "Endpoint URL: $URL/solve"
Write-Host "Health check: $URL/health"
Write-Host ""
Write-Host "Submit this URL to the competition: $URL/solve" -ForegroundColor Cyan
