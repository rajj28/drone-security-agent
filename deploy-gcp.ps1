# Deploy to Google Cloud Run (PowerShell)
# Run: .\deploy-gcp.ps1

$ErrorActionPreference = "Stop"

$PROJECT_ID = ( gcloud config get-value project 2>$null).Trim()
$REGION = "us-central1"

Write-Host "🚀 Deploying to Google Cloud Project: $PROJECT_ID" -ForegroundColor Green
Write-Host "Region: $REGION" -ForegroundColor Green
Write-Host ""

# Check if .env file exists
if (-not (Test-Path .env)) {
    Write-Host "❌ .env file not found!" -ForegroundColor Red
    exit 1
}

# Load environment variables from .env
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^#][^=]*)=(.*)$') {
        $name = $matches[1]
        $value = $matches[2]
        Set-Item -Path "Env:$name" -Value $value
    }
}

# Verify required variables
if (-not $env:GEMINI_API_KEY -or -not $env:GROQ_API_KEY -or -not $env:PINECONE_API_KEY) {
    Write-Host "❌ Missing required API keys in .env file" -ForegroundColor Red
    Write-Host "Required: GEMINI_API_KEY, GROQ_API_KEY, PINECONE_API_KEY" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Environment variables loaded" -ForegroundColor Green
Write-Host ""

# Enable required APIs
Write-Host "🔧 Enabling required APIs..." -ForegroundColor Yellow
 gcloud services enable cloudbuild.googleapis.com
 gcloud services enable run.googleapis.com
Write-Host "✅ APIs enabled" -ForegroundColor Green
Write-Host ""

# Build and push API container
Write-Host "📦 Building API container..." -ForegroundColor Yellow
 docker build -t "gcr.io/$PROJECT_ID/drone-security-api:latest" -f Dockerfile.railway .
docker push "gcr.io/$PROJECT_ID/drone-security-api:latest"
Write-Host "✅ API container pushed" -ForegroundColor Green
Write-Host ""

# Build and push Dashboard container
Write-Host "📦 Building Dashboard container..." -ForegroundColor Yellow
docker build -t "gcr.io/$PROJECT_ID/drone-security-dashboard:latest" -f Dockerfile.dashboard .
docker push "gcr.io/$PROJECT_ID/drone-security-dashboard:latest"
Write-Host "✅ Dashboard container pushed" -ForegroundColor Green
Write-Host ""

# Deploy API to Cloud Run
Write-Host "🚀 Deploying API to Cloud Run..." -ForegroundColor Yellow
$envVars = "GEMINI_API_KEY=$env:GEMINI_API_KEY,GROQ_API_KEY=$env:GROQ_API_KEY,PINECONE_API_KEY=$env:PINECONE_API_KEY,USE_CLOUD_ANALYZER=true,AGENT_LLM_PROVIDER=groq,MAX_FRAMES=10"

 gcloud run deploy drone-security-api `
    --image "gcr.io/$PROJECT_ID/drone-security-api:latest" `
    --region $REGION `
    --platform managed `
    --allow-unauthenticated `
    --set-env-vars $envVars `
    --memory 2Gi `
    --cpu 2 `
    --timeout 300 `
    --max-instances 5

$API_URL = ( gcloud run services describe drone-security-api --region $REGION --format 'value(status.url)' 2>$null).Trim()
Write-Host "✅ API deployed at: $API_URL" -ForegroundColor Green
Write-Host ""

# Deploy Dashboard to Cloud Run
Write-Host "🚀 Deploying Dashboard to Cloud Run..." -ForegroundColor Yellow
$dashboardEnvVars = "API_URL=$API_URL,GEMINI_API_KEY=$env:GEMINI_API_KEY,GROQ_API_KEY=$env:GROQ_API_KEY,PINECONE_API_KEY=$env:PINECONE_API_KEY"

gcloud run deploy drone-security-dashboard `
    --image "gcr.io/$PROJECT_ID/drone-security-dashboard:latest" `
    --region $REGION `
    --platform managed `
    --allow-unauthenticated `
    --set-env-vars $dashboardEnvVars `
    --memory 1Gi `
    --cpu 1 `
    --timeout 300

$DASHBOARD_URL = ( gcloud run services describe drone-security-dashboard --region $REGION --format 'value(status.url)' 2>$null).Trim()
Write-Host "✅ Dashboard deployed at: $DASHBOARD_URL" -ForegroundColor Green
Write-Host ""

Write-Host "🎉 Deployment Complete!" -ForegroundColor Green -BackgroundColor Black
Write-Host ""
Write-Host "API URL: $API_URL" -ForegroundColor Cyan
Write-Host "Dashboard URL: $DASHBOARD_URL" -ForegroundColor Cyan
Write-Host ""
$healthUrl = "$API_URL/health"
Write-Host "Health Check: $healthUrl" -ForegroundColor Cyan
