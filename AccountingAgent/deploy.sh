#!/bin/bash
# Deploy to GCP Cloud Run
# Usage: bash deploy.sh

set -euo pipefail

PROJECT_ID="ainm26osl-764"
REGION="europe-west1"
SERVICE_NAME="accounting-agent"
IMAGE="europe-west1-docker.pkg.dev/${PROJECT_ID}/accounting-agent/app:latest"

echo "=== Building and deploying to Cloud Run ==="

# Ensure APIs are enabled
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  --project="${PROJECT_ID}" --quiet 2>/dev/null || true

# Create Artifact Registry repo (idempotent)
gcloud artifacts repositories create accounting-agent \
  --repository-format=docker \
  --location="${REGION}" \
  --project="${PROJECT_ID}" 2>/dev/null || true

# Configure Docker auth
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet 2>/dev/null || true

# Build and push
echo "Building Docker image..."
docker build -t "${IMAGE}" .
echo "Pushing to Artifact Registry..."
docker push "${IMAGE}"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --platform=managed \
  --allow-unauthenticated \
  --timeout=300 \
  --memory=1Gi \
  --cpu=2 \
  --min-instances=1 \
  --max-instances=10 \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION}" \
  --quiet

# Get the URL
URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)")

echo ""
echo "=== Deployed! ==="
echo "Endpoint URL: ${URL}/solve"
echo "Health check: ${URL}/health"
echo ""
echo "Submit this URL to the competition: ${URL}/solve"
