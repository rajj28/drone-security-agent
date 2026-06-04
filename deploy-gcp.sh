#!/bin/bash
# Deploy to Google Cloud Run
# Prerequisites: gcloud CLI installed and authenticated

set -e

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
REGION="us-central1"

echo "🚀 Deploying to Google Cloud Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Check if .env file exists with required variables
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    exit 1
fi

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Verify required variables
if [ -z "$GEMINI_API_KEY" ] || [ -z "$GROQ_API_KEY" ] || [ -z "$PINECONE_API_KEY" ]; then
    echo "❌ Missing required API keys in .env file"
    echo "Required: GEMINI_API_KEY, GROQ_API_KEY, PINECONE_API_KEY"
    exit 1
fi

echo "✅ Environment variables loaded"
echo ""

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
#gcloud services enable run.googleapis.com
# Already enabled - skip to save time
echo "✅ APIs enabled"
echo ""

# Build and push API container
echo "📦 Building API container..."
docker build -t gcr.io/$PROJECT_ID/drone-security-api:latest -f Dockerfile.railway .
docker push gcr.io/$PROJECT_ID/drone-security-api:latest
echo "✅ API container pushed"
echo ""

# Build and push Dashboard container
echo "📦 Building Dashboard container..."
docker build -t gcr.io/$PROJECT_ID/drone-security-dashboard:latest -f Dockerfile.dashboard .
docker push gcr.io/$PROJECT_ID/drone-security-dashboard:latest
echo "✅ Dashboard container pushed"
echo ""

# Deploy API to Cloud Run
echo "🚀 Deploying API to Cloud Run..."
gcloud run deploy drone-security-api \
    --image gcr.io/$PROJECT_ID/drone-security-api:latest \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --set-env-vars "GEMINI_API_KEY=$GEMINI_API_KEY,GROQ_API_KEY=$GROQ_API_KEY,PINECONE_API_KEY=$PINECONE_API_KEY,USE_CLOUD_ANALYZER=true,AGENT_LLM_PROVIDER=groq,MAX_FRAMES=10" \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 5

API_URL=$(gcloud run services describe drone-security-api --region $REGION --format 'value(status.url)')
echo "✅ API deployed at: $API_URL"
echo ""

# Deploy Dashboard to Cloud Run
echo "🚀 Deploying Dashboard to Cloud Run..."
gcloud run deploy drone-security-dashboard \
    --image gcr.io/$PROJECT_ID/drone-security-dashboard:latest \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --set-env-vars "API_URL=$API_URL,GEMINI_API_KEY=$GEMINI_API_KEY,GROQ_API_KEY=$GROQ_API_KEY,PINECONE_API_KEY=$PINECONE_API_KEY" \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300

DASHBOARD_URL=$(gcloud run services describe drone-security-dashboard --region $REGION --format 'value(status.url)')
echo "✅ Dashboard deployed at: $DASHBOARD_URL"
echo ""

echo "🎉 Deployment Complete!"
echo ""
echo "API URL: $API_URL"
echo "Dashboard URL: $DASHBOARD_URL"
echo ""
echo "Health Check: $API_URL/health"
