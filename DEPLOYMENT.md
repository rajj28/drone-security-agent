# Deployment Guide for Drone Security Analyst Agent

## Overview

This guide covers multiple deployment options for the Drone Security Analyst Agent, including Docker deployment, cloud deployment, and local development setup.

## Prerequisites

- Docker and Docker Compose installed
- OpenAI API key
- Pinecone API key and index created
- Python 3.11+ (for local development)

## Quick Start with Docker

### 1. Environment Setup

Copy the environment template and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your actual API keys:
```
OPENAI_API_KEY=your_actual_openai_key
PINECONE_API_KEY=your_actual_pinecone_key
PINECONE_INDEX_NAME=your_pinecone_index_name
```

### 2. Build and Run

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

### 3. Access the Services

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **Health Check**: http://localhost:8000/health

## Deployment Options

### Option 1: Docker Compose (Recommended for Production)

Deploy both API and dashboard services:

```bash
# Production deployment
docker-compose -f docker-compose.yml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 2: Docker Single Service (API Only)

If you only need the API backend:

```bash
# Build the image
docker build -t drone-security-agent .

# Run the container
docker run -d \
  --name drone-security-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -e OPENAI_API_KEY=your_key \
  -e PINECONE_API_KEY=your_key \
  drone-security-agent
```

### Option 3: Local Development

For development and testing:

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your keys

# Run the pipeline components
python src/frame_extractor.py
python src/telemetry_generator.py
python src/vision_analyzer.py
python src/pinecone_indexer.py
python src/alert_engine.py
python src/summarizer.py

# Start API server
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start dashboard
streamlit run demo/dashboard.py
```

## Cloud Deployment

### AWS ECS/EKS

1. Push the Docker image to ECR:
```bash
# Build and tag
docker build -t drone-security-agent .
docker tag drone-security-agent:latest <aws-account-id>.dkr.ecr.<region>.amazonaws.com/drone-security-agent:latest

# Push to ECR
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws-account-id>.dkr.ecr.<region>.amazonaws.com
docker push <aws-account-id>.dkr.ecr.<region>.amazonaws.com/drone-security-agent:latest
```

2. Deploy using ECS Task Definition or Kubernetes manifests.

### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT-ID/drone-security-agent

# Deploy to Cloud Run
gcloud run deploy drone-security-agent --image gcr.io/PROJECT-ID/drone-security-agent --platform managed
```

### Azure Container Instances

```bash
# Build and push to ACR
az acr build --registry <registry-name> --image drone-security-agent .

# Deploy to ACI
az container create \
  --resource-group <resource-group> \
  --name drone-security-agent \
  --image <registry-name>.azurecr.io/drone-security-agent \
  --cpu 2 --memory 4 \
  --ports 8000
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4 Vision |
| `PINECONE_API_KEY` | Yes | Pinecone API key for vector search |
| `PINECONE_INDEX_NAME` | Yes | Name of your Pinecone index |
| `OPENAI_MODEL` | No | OpenAI model (default: gpt-4o) |
| `OPENAI_TEMPERATURE` | No | LLM temperature (default: 0.1) |

## Data Persistence

The application uses the following volume mounts:
- `./data:/app/data` - Input video files and extracted frames
- `./outputs:/app/outputs` - Analysis results, alerts, and session data

Ensure these directories are backed up in production.

## Monitoring and Health Checks

### Health Endpoints

- `GET /health` - Basic health check
- `GET /frames` - List available frames
- `GET /alerts` - Get all alerts

### Logs

```bash
# Docker Compose logs
docker-compose logs -f drone-security-api
docker-compose logs -f drone-security-dashboard

# Individual container logs
docker logs -f <container-name>
```

## Scaling Considerations

1. **API Scaling**: Deploy multiple instances behind a load balancer
2. **Database**: Consider external PostgreSQL for production data persistence
3. **File Storage**: Use S3 or similar for video and frame storage
4. **Caching**: Add Redis for session management and caching

## Security Notes

- Never commit `.env` files to version control
- Use HTTPS in production (add SSL termination)
- Implement authentication for production APIs
- Regularly rotate API keys
- Monitor API usage and costs

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8000 and 8501 are available
2. **API key errors**: Verify keys in `.env` are correct
3. **Memory issues**: Increase Docker memory allocation for large video files
4. **Pinecone connection**: Ensure index exists and is accessible

### Debug Commands

```bash
# Check container status
docker-compose ps

# Access container shell
docker-compose exec drone-security-api bash

# Test API locally
curl http://localhost:8000/health
```

## Free/Cost-Effective Deployment Options

### Option 1: Local Machine (100% Free)

Deploy on your own computer for zero cloud costs:

```bash
# 1. Clone repository
git clone <your-repo-url>
cd drone-security-agent

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start services
$env:USE_CLOUD_ANALYZER="true"
uvicorn src.api:app --host 0.0.0.0 --port 8000

# In another terminal:
streamlit run demo/dashboard.py --server.port 8501
```

**Access:**
- Dashboard: http://localhost:8501
- API: http://localhost:8000

---

### Option 2: Google Colab (Free Tier)

Run entirely in Google Colab with free GPU:

```python
# In a Colab notebook:
!git clone https://github.com/your-repo/drone-security-agent.git
%cd drone-security-agent
!pip install -r requirements.txt

# Set API keys
import os
os.environ['OPENAI_API_KEY'] = 'your-key'
os.environ['PINECONE_API_KEY'] = 'your-key'

# Start API with ngrok for public access
!pip install pyngrok
from pyngrok import ngrok
ngrok.set_auth_token('your-ngrok-token')
public_url = ngrok.connect(8000)
print(f"Public API URL: {public_url}")

# Run API
!uvicorn src.api:app --host 0.0.0.0 --port 8000
```

**Note:** Ngrok free tier provides temporary public URLs.

---

### Option 3: GitHub Codespaces (Free Tier)

Use GitHub Codespaces for development:

1. Push code to GitHub repository
2. Open repository in GitHub Codespaces
3. Run setup commands in terminal
4. Forward ports 8000 and 8501

---

## Completely Free Cloud Deployment (Public URL for Judges)

### Option 1: Render.com (Free Tier - RECOMMENDED)

**100% FREE** - Gets you a public URL instantly. App sleeps after 15 min of inactivity but wakes up on next request (just takes 10-15 seconds).

```bash
# 1. Sign up at https://render.com (free, no credit card)
# 2. Connect your GitHub repo
# 3. Create render.yaml in your project root:
```

**render.yaml:**
```yaml
services:
  - type: web
    name: drone-security-api
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: OPENAI_API_KEY
        sync: false  # Set in Render dashboard
      - key: PINECONE_API_KEY
        sync: false
      - key: USE_CLOUD_ANALYZER
        value: true

  - type: web
    name: drone-security-dashboard
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run demo/dashboard.py --server.port $PORT
    envVars:
      - key: API_URL
        value: https://drone-security-api.onrender.com
```

**Access after deploy:**
- API: `https://drone-security-api.onrender.com`
- Dashboard: `https://drone-security-dashboard.onrender.com`

---

### Option 2: Ngrok (Instant Public URL - FREE)

**Perfect for demo day** - Expose your local server to internet instantly.

```bash
# 1. Sign up free at https://ngrok.com
# 2. Download ngrok
# 3. Authenticate
ngrok config add-authtoken YOUR_TOKEN

# 4. In Terminal 1 - Start your local API
$env:USE_CLOUD_ANALYZER="true"
venv\Scripts\uvicorn src.api:app --host 0.0.0.0 --port 8000

# 5. In Terminal 2 - Start ngrok tunnel
ngrok http 8000

# 6. Get public URL (e.g., https://abc123.ngrok.io)
# Share this URL with judges!
```

**Pros:** Instant, free, uses your local machine (fast)
**Cons:** URL changes each time (unless you pay $5/month for reserved domain)

---

### Option 3: Google Colab + Ngrok (FREE)

Deploy entirely in Google Colab - free GPU and public URL:

```python
# In Google Colab notebook:
!git clone https://github.com/your-username/drone-security-agent.git
%cd drone-security-agent
!pip install -r requirements.txt pyngrok

import os
os.environ['OPENAI_API_KEY'] = 'your-key'
os.environ['PINECONE_API_KEY'] = 'your-key'

# Get public URL
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"Public URL for judges: {public_url}")

# Start server
!uvicorn src.api:app --host 0.0.0.0 --port 8000
```

**URL stays active as long as Colab notebook is running.**

---

### Option 4: PythonAnywhere (FREE)

Free Python hosting with always-on web apps:

1. Sign up at https://www.pythonanywhere.com (free tier)
2. Upload code via web interface or Git
3. Set up virtualenv and install requirements
4. Configure web app with WSGI
5. Get free URL: `yourname.pythonanywhere.com`

**Limitations:** 512 MB storage, 100 seconds CPU/day (fine for demos)

---

## Quick Summary for Judges Demo

| Service | Cost | Setup Time | URL Stability | Best For |
|---------|------|------------|---------------|----------|
| **Render** | $0 | 10 min | Permanent | Production demo |
| **Ngrok** | $0 | 2 min | Temporary (changes) | Quick demos |
| **Google Colab** | $0 | 15 min | Session-based | GPU demos |
| **PythonAnywhere** | $0 | 20 min | Permanent | Always-on demo |

**RECOMMENDATION:** Use **Render.com** for a permanent free URL judges can access anytime, or **Ngrok** for instant one-time demos.

---

### Option 5: Railway (Free $5/month credit)

Similar to Render but with $5 free credit monthly:

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and deploy
railway login
railway init
railway up
```

---

### Cost Optimization Tips

| Service | Free Tier | Cost After |
|---------|-----------|------------|
| **OpenAI GPT-4o** | $5 credit | ~$0.005/image |
| **Pinecone** | 1 index free | $70/month |
| **Hugging Face** | Free API | Free |
| **Ngrok** | 1 tunnel | $5/month |
| **Railway** | $5 credit | Pay-as-you-go |
| **Render** | 1 web service | $7/month |

**To Minimize Costs:**
1. Use local deployment for development
2. Limit frame extraction (MAX_FRAMES=20)
3. Use GPT-4o-mini instead of GPT-4o
4. Cache analysis results
5. Process videos in batches

---

## Support

For deployment issues:
1. Check the logs for error messages
2. Verify all environment variables are set
3. Ensure API keys have proper permissions
4. Consult the main README.md for detailed setup instructions
