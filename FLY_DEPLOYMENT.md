# Deploying to Fly.io

This guide deploys the Drone Security Analyst Agent (FastAPI backend + built React dashboard in one container) to Fly.io.

## Why Fly.io for this app

Cloud Run throttles a container's CPU to near-zero as soon as the HTTP response is sent — but this app does its heavy work (frame extraction, vision analysis) in a **background task after the upload response returns**, so on Cloud Run the pipeline runs on starved CPU unless you pay for "CPU always allocated". Fly.io machines are real VMs: full CPU for the whole pipeline, and `auto_stop_machines = "suspend"` resumes in about a second instead of a multi-second cold start.

## Prerequisites

- A [Fly.io account](https://fly.io/app/sign-up) (requires a payment card; a small app like this typically costs a few dollars per month, and suspended machines cost almost nothing while idle)
- `flyctl` CLI:

```powershell
# Windows (PowerShell)
pwsh -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

```bash
# macOS / Linux
curl -L https://fly.io/install.sh | sh
```

Then log in:

```bash
fly auth login
```

## 1. Build the frontend

The Dockerfile copies the pre-built dashboard from `frontend/dist/`, so build it first:

```bash
cd frontend
npm install
npm run build
cd ..
```

## 2. Create the app

`fly.toml` is already in the repo. Create the app (pick a globally unique name, or keep the default and let Fly suffix it):

```bash
fly apps create drone-security-agent
```

If you choose a different name, update `app = "..."` in `fly.toml`.

The default region is `bom` (Mumbai). Change `primary_region` in `fly.toml` if your users are elsewhere (`fly platform regions` lists all options).

## 3. Set secrets

Secrets replace the `.env` file in production. Required:

```bash
fly secrets set GEMINI_API_KEY="your-gemini-key" PINECONE_API_KEY="your-pinecone-key"
```

Optional (set the ones you use):

```bash
fly secrets set MONGODB_URI="mongodb+srv://..." GROQ_API_KEY="..." NVIDIA_API_KEY="..." HF_API_TOKEN="..." GEMINI_API_KEY_2="..." GEMINI_API_KEY_3="..."
```

Note: without `MONGODB_URI`, sessions live only in memory/local disk and are lost when the machine restarts or suspends. Set it for persistent sessions (MongoDB Atlas free tier works).

## 4. Deploy

```bash
fly deploy
```

Fly builds the Docker image on a remote builder (no local Docker needed) and starts the machine. When it finishes:

```bash
fly open          # open the dashboard in your browser
fly logs          # tail application logs
fly status        # machine state and health checks
```

## 5. Verify

- `https://<your-app>.fly.dev/health` returns `{"status": ...}`
- `https://<your-app>.fly.dev/docs` shows the API docs
- Upload the sample video from the dashboard and watch the pipeline complete

## Performance tuning

| Goal | Command / setting |
|------|-------------------|
| Zero wake-up latency (always warm) | `min_machines_running = 1` in `fly.toml` |
| More CPU for faster pipeline | `fly scale vm shared-cpu-4x` or `performance-1x` |
| More memory | `fly scale memory 4096` |
| Second region | `fly regions add sin` |

The default config (`shared-cpu-2x`, 2GB, suspend-on-idle) is the best cost/performance starting point: idle costs are minimal and resume is ~1s.

## Ongoing deploys

After any code change:

```bash
cd frontend && npm run build && cd ..
fly deploy
```

## Troubleshooting

- **Out-of-memory during video processing** — `fly scale memory 4096`, or reduce `MAX_FRAMES`.
- **Health check failing on deploy** — the app needs a few seconds to import Python deps; the `grace_period = "30s"` in `fly.toml` covers this. Check `fly logs` for import errors (usually a missing secret).
- **Sessions disappear after idle** — set `MONGODB_URI` (see step 3); local disk on Fly machines is ephemeral unless you attach a volume.
- **Machine never stops (billing)** — the dashboard polls `/health` every 30s while a browser tab is open; close the tab and the machine suspends after a few minutes of no traffic.
