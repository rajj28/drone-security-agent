# DiagramGPT Prompt — Drone Security Analyst Agent Architecture

Paste the prompt below into https://www.eraser.io/diagramgpt to generate the diagram.

---

## Prompt

```
Create a professional cloud architecture diagram for an AI-powered Drone Security Surveillance System with the following components and data flows:

TITLE: "Drone Security Analyst Agent — System Architecture"

--- USER LAYER ---
- "Security Operator" (user icon) connects to "React Dashboard" via HTTPS (mobile-responsive, works on phones)
- "React Dashboard" (React 19 + TypeScript + Vite 8) is a single-page app served from Fly.io. It has these modules: Video Upload (with "Enable Cloud Enhancers" toggle), Live Capture (phone camera / drone stream), Frame Analysis Viewer, Alert Center, Semantic Search, Session Summary, AI Security Agent (Q&A), Agentic Orchestration View, Debug Panel

--- CLOUD PLATFORM (Fly.io Machines, shared-cpu-4x / 2GB, suspend-on-idle) ---
- Single container (Python 3.10, pre-built React dist copied in)
- "FastAPI Backend" (port 8080, Uvicorn) serves both the REST API and the React static frontend
- FastAPI has these endpoint groups: /upload-video, /live/* (live capture start/frame/status/stop), /health, /frames, /alerts, /search, /qa, /sessions, /processing-status/{id}
- "Background Task Pipeline" is triggered by POST /upload-video (or live capture stop) and runs asynchronously inside the same container
- A "Keep-Alive Heartbeat" task runs every 60 seconds during processing to keep session status fresh
- "Pipeline Lock (threading)" serializes concurrent video processing
- "Cancellation module" lets the operator abort a running pipeline

--- AI PROCESSING PIPELINE (6 stages, runs inside Background Task) ---
Stage 1: "Intelligent Frame Extractor" — uses FFmpeg + OpenCV. Supports 4 strategies: uniform, motion_based, scene_change, hybrid. Outputs frame_*.jpg images (10-500 frames per video). Input: uploaded video (MP4/AVI/MOV/DAV/MKV/+10 formats)

Stage 2: "Telemetry Generator" — generates simulated drone GPS coordinates, altitude, heading, location tags (Main Gate, Garage, Perimeter North, Restricted Zone, etc.) for each frame

Stage 3: "Vision Analysis Engine" (concurrent, 5 workers) — Gemini is the primary vision model, with an optional verification ensemble:
  - Sub-component "Gemini 2.5 VLM — Stage 1" (neutral observation): structured JSON output with scene_type, people_count, person_features, objects, security_signals, threat_level
  - Sub-component "Gemini 2.5 VLM — Stage 2" (security deep-dive, conditional): triggered only when suspicious keywords detected in Stage 1. Detailed threat analysis with reasoning
  - Sub-component "Cloud Enhancers (optional, toggled per run from the dashboard)" — independent models that CROSS-CHECK Gemini's analysis to reduce hallucination:
      - "Hugging Face CLIP" (openai/clip-vit-base-patch32 via HF Inference API): zero-shot classification against 30 security behavior prompts, outputs an independent threat_score (0-100) and threat categories
      - "Hugging Face BLIP" (Salesforce/blip-image-captioning-base via HF Inference API): independent scene caption + security keyword extraction
  - Sub-component "Unified Context Manager": provides previous 5 frames' history for temporal awareness

Stage 4: "Alert Engine" — deterministic rule engine with 5 escalation rules:
  Rule 1: Unknown signals → MEDIUM
  Rule 2: After-hours activity → HIGH  
  Rule 3: Weapons/fire → CRITICAL
  Rule 4: Vehicle in restricted zone → MEDIUM
  Rule 5: Loitering → HIGH
  Severity levels: CLEAR, LOW, MEDIUM, HIGH, CRITICAL

Stage 5: "Person Tracker" — cross-frame re-identification by clothing color, body type, accessories, position

Stage 6: "Session Summarizer" — generates narrative security report using Gemini text generation from all frame analyses

--- VECTOR SEARCH & Q&A (parallel to pipeline completion) ---
- "Pinecone Indexer" upserts frame descriptions to Pinecone using integrated inference (llama-text-embed-v2, 768 dimensions, cosine metric). Each record has metadata: frame_id, timestamp, location, threat_level, people_count
- "Security Q&A Agent" (RAG pattern): receives user question → Pinecone semantic search (top-k=5) → retrieves relevant frame analyses + all suspicious frames → constructs prompt with session context + conversation history → calls Gemini for answer generation

--- EXTERNAL SERVICES ---
- "Google Gemini API" (Gemini 2.5 Pro + 2.5 Flash, round-robin across up to 3 API keys for rate-limit distribution). Connects to Vision Analysis Engine and Q&A Agent and Session Summarizer
- "Pinecone Vector DB" (index: flytbase, AWS us-east-1, 768 dim, cosine). Connects to Pinecone Indexer and Q&A Agent (search)
- "MongoDB Atlas" (document database). Stores session status, processing progress, frame metadata. GridFS for binary frame storage. Connects to FastAPI Backend for session persistence
- "Hugging Face Inference API" (free tier, cloud GPUs). Connects to the CLIP and BLIP Cloud Enhancer sub-components in Vision Analysis (active only when "Enable Cloud Enhancers" is toggled on)

--- STORAGE LAYERS ---
- "File System" (ephemeral container storage): data/sessions/{id}/extracted/ (frame JPGs), /telemetry/ (JSON), /analysis/ (JSON), /alerts/ (JSON), /session/ (person DB, QA log)
- "MongoDB Atlas": persistent session metadata + status + GridFS images
- "Pinecone": persistent vector embeddings for semantic search

--- DATA FLOWS (draw arrows) ---
1. Security Operator → React Dashboard (HTTPS, WebSocket polling every 15s)
2. React Dashboard → FastAPI Backend (REST API calls)
3. FastAPI Backend → Background Task Pipeline (on video upload)
4. Frame Extractor → Telemetry Generator → Vision Analysis Engine → Alert Engine → Person Tracker → Session Summarizer (sequential pipeline)
5. Vision Analysis Engine → Hugging Face Inference API (CLIP + BLIP cross-check calls, only when Cloud Enhancers enabled)
6. Vision Analysis Engine → Google Gemini API (VLM Stage 1 + Stage 2)
7. Vision Analysis Engine → File System (save analysis JSON)
8. Alert Engine → File System (save alerts JSON)
9. After pipeline: Pinecone Indexer → Pinecone Vector DB (upsert frame vectors)
10. FastAPI Backend → MongoDB Atlas (save/read session status)
11. Q&A Agent → Pinecone Vector DB (semantic search)
12. Q&A Agent → Google Gemini API (answer generation)
13. Q&A Agent → File System (read analysis + alerts for context)
14. FastAPI Backend → React Dashboard (JSON responses + static file serving)

--- DEPLOYMENT ---
- "fly deploy" builds the Docker image on Fly's remote Depot builder, pushes to the Fly registry, and rolls it out to the Fly.io Machine (rolling strategy, /health checks gate the rollout)

--- STYLING ---
- Use a dark/professional color scheme
- Group Google services together in a "Google Cloud Platform" boundary box
- Group AI models in an "AI Engine" boundary box
- Group external APIs in an "External Services" boundary box
- Show the pipeline stages as a numbered flow (1→2→3→4→5→6)
- Use appropriate icons: shield for security, brain for AI, database for storage, cloud for services
```
