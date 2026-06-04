# Google Cloud Platform Deployment Guide

Deploy the Drone Security Agent to Google Cloud Run using your $300 free credit.

## Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
- GCP project created
- `.env` file with API keys

## Quick Start

### 1. Authenticate & Set Project

```bash
# Login to Google Cloud
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Verify
 gcloud config get-value project
```

### 2. Enable APIs

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
```

### 3. Deploy

```bash
# Make script executable (on Linux/Mac)
chmod +x deploy-gcp.sh

# Run deployment
./deploy-gcp.sh
```

## Manual Deployment Steps

### Build Containers

```bash
# API
 docker build -t gcr.io/YOUR_PROJECT_ID/drone-security-api:latest -f Dockerfile.railway .
docker push gcr.io/YOUR_PROJECT_ID/drone-security-api:latest

# Dashboard
docker build -t gcr.io/YOUR_PROJECT_ID/drone-security-dashboard:latest -f Dockerfile.dashboard .
docker push gcr.io/YOUR_PROJECT_ID/drone-security-dashboard:latest
```

### Deploy API

```bash
 gcloud run deploy drone-security-api \
  --image gcr.io/YOUR_PROJECT_ID/drone-security-api:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "GEMINI_API_KEY=your_key,GROQ_API_KEY=your_key,PINECONE_API_KEY=your_key,USE_CLOUD_ANALYZER=true,AGENT_LLM_PROVIDER=groq" \
  --memory 2Gi \
  --cpu 2
```

### Deploy Dashboard

```bash
# Get API URL first
API_URL=$( gcloud run services describe drone-security-api --region us-central1 --format 'value(status.url)')

# Deploy dashboard
gcloud run deploy drone-security-dashboard \
  --image gcr.io/YOUR_PROJECT_ID/drone-security-dashboard:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "API_URL=$API_URL,GEMINI_API_KEY=your_key" \
  --memory 1Gi
```

## View Logs

```bash
# API logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=drone-security-api" --limit=50

# Dashboard logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=drone-security-dashboard" --limit=50
```

## Update Environment Variables

```bashn# Update API
gcloud run services update drone-security-api \
  --set-env-vars "NEW_VAR=value"

# Update Dashboard
gcloud run services update drone-security-dashboard \
  --set-env-vars "NEW_VAR=value"
```

## Cost Estimate (with $300 credit)

| Service | Memory | CPU | Est. Monthly Cost |
|---------|--------|-----|-------------------|
| API | 2GB | 2 | ~$50-80 |
| Dashboard | 1GB | 1 | ~$20-30 |
| **Total** | | | **~$70-110/month** |

**With $300 credit: ~2.5-4 months free!**

## Troubleshooting

### Permission Denied on Docker Push

```bash
# Configure Docker to use gcloud credentials
gcloud auth configure-docker
```

### Build Fails

```bash
# Check Cloud Build logs
gcloud builds list
gcloud builds log BUILD_ID
```

### Service Not Found

```bash
# List services
gcloud run services list

# Describe specific service
gcloud run services describe drone-security-api --region us-central1
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User          │────▶│  Cloud Run       │────▶│  Cloud Run      │
│   Browser       │     │  Dashboard       │     │  API            │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                      │
                                                      ▼
                                               ┌──────────────┐
                                               │  External    │
                                               │  APIs        │
                                               │  (Gemini,    │
                                               │   Groq,      │
                                               │   Pinecone)  │
                                               └──────────────┘
```

## Support

- [Cloud Run Docs](https://cloud.google.com/run/docs)
- [Cloud Build Docs](https://cloud.google.com/build/docs)
- [GCP Pricing Calculator](https://cloud.google.com/products/calculator)
