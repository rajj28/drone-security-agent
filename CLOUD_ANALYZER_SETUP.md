# Cloud-Enhanced Analyzer Setup Guide

## Overview
The Cloud-Enhanced Analyzer uses **Hugging Face's free Inference API** to run CLIP and BLIP models in the cloud, while keeping GPT-4o analysis local. This gives you enhanced analysis without:
- Local GPU requirements
- Model loading delays
- Memory issues
- Timeout problems

## Benefits
- **Free tier**: 30,000 requests/month on Hugging Face
- **Fast inference**: Cloud GPUs (no local model loading)
- **Reliable**: Professional API infrastructure
- **Scalable**: No local resource constraints

## Setup Steps

### 1. Get Hugging Face API Token (FREE)

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: "drone-security-analyzer"
4. Role: "read"
5. Click "Generate token"
6. Copy the token (starts with `hf_...`)

### 2. Set Environment Variable

**Windows PowerShell:**
```powershell
$env:HF_API_TOKEN="your_token_here"
$env:USE_CLOUD_ANALYZER="true"
```

**Windows Command Prompt:**
```cmd
set HF_API_TOKEN=your_token_here
set USE_CLOUD_ANALYZER=true
```

**Permanent (Windows System Settings):**
1. Open "System Properties" → "Environment Variables"
2. Add new User Variable:
   - Name: `HF_API_TOKEN`
   - Value: `your_token_here`
3. Add another:
   - Name: `USE_CLOUD_ANALYZER`
   - Value: `true`

### 3. Start the Server

```powershell
$env:HF_API_TOKEN="hf_your_token_here"
$env:USE_CLOUD_ANALYZER="true"
venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000
```

### 4. Verify It's Working

You should see in the console:
```
Analyzer Configuration:
  Ultimate Analyzer (Local CLIP+BLIP+GPT-4o): False
  Cloud Analyzer (HF CLIP+BLIP + Local GPT-4o): True
  BLIP Analyzer: False
  CLIP Analyzer: False
  Standard GPT-4o Vision: False
```

When analyzing frames, you'll see:
```
Using Cloud-Enhanced Analyzer (HF CLIP + BLIP + Local GPT-4o)...
Calling Hugging Face CLIP API...
Calling Hugging Face BLIP API...
Running local GPT-4o analysis with cloud context...
Cloud analysis saved for frame_001
```

## How It Works

1. **CLIP (Cloud)**: Visual similarity matching for threat detection
   - Model: `openai/clip-vit-base-patch32`
   - Detects: suspicious poses, concealed objects, unusual behavior
   
2. **BLIP (Cloud)**: Image captioning for scene understanding
   - Model: `Salesforce/blip-image-captioning-base`
   - Provides: detailed descriptions, object identification
   
3. **GPT-4o (Local)**: Deep security analysis
   - Uses cloud context (CLIP + BLIP results) for better analysis
   - Provides: threat assessment, reasoning, recommendations

## Free Tier Limits

- **30,000 requests/month** on Hugging Face Inference API
- At ~50 frames per video: **600 videos/month** free!
- Request latency: ~1-3 seconds per API call

## Troubleshooting

### "No HF_API_TOKEN set"
- Token not configured. Follow Step 2 above.

### API rate limit errors
- You've hit the free tier limit. Wait until next month or upgrade.

### "Cloud Analyzer failed"
- Check your internet connection
- Verify token is valid at https://huggingface.co/settings/tokens
- Check if models are loading at https://huggingface.co/openai/clip-vit-base-patch32

### Slow analysis
- Each frame makes 2-3 API calls (CLIP, BLIP, GPT-4o)
- Expected time: ~5-10 seconds per frame (faster than local models)

## Comparison: Local vs Cloud vs Standard

| Feature | Local Ultimate | Cloud Enhanced | Standard GPT-4o |
|---------|---------------|----------------|-----------------|
| GPU Required | Yes (8GB+) | No | No |
| Initial Load | 2-3 minutes | Instant | Instant |
| Per Frame | 30-60s | 5-10s | 5-10s |
| CLIP + BLIP | Local | Cloud | |
| Cost | Free (GPU power) | Free (30k req/mo) | OpenAI API cost |
| Reliability | Moderate | High | High |

## Recommendation

**Use Cloud Enhanced Analyzer when:**
- You don't have a powerful GPU
- You want fast, reliable enhanced analysis
- You stay within 30k requests/month

**Use Standard GPT-4o when:**
- You want fastest processing
- You don't need CLIP/BLIP enhancement
- You have API rate limits

## Support

- Hugging Face API docs: https://huggingface.co/docs/api-inference
- Get help: https://discuss.huggingface.co/
- Token issues: https://huggingface.co/settings/tokens
