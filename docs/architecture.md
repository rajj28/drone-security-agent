# Drone Security Analyst Agent - Architecture Design

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DRONE SECURITY ANALYST AGENT                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────────────┐
│   Video Input   │────→│  Frame Extractor │────→│   Vision Analyzer (GPT-4o)  │
│  (Simulated/    │     │                  │     │                             │
│   Real Drone)   │     │ • Extract frames │     │ • Scene description         │
│  (MP4 Input)    │     │ • Generate       │     │ • Object detection          │
└─────────────────┘     │   telemetry      │     │ • Behavior analysis         │
                        └──────────────────┘     │ • Alert generation          │
                                                   └─────────────┬─────────────┘
                                                                 │
                                ┌────────────────────────────────┼────────────────┐
                                │                                │                │
                                ▼                                ▼                ▼
                       ┌─────────────────┐          ┌──────────────────┐  ┌─────────────────┐
                       │  Alert Engine   │          │  Frame Indexer   │  │   Summarizer    │
                       │                 │          │  (Pinecone DB)   │  │                 │
                       │ • Severity      │          │                  │  │ • Hourly        │
                       │   classification│          │ • Vector store   │  │   summaries     │
                       │ • Immediate     │          │ • Metadata       │  │ • Pattern       │
                       │   notifications │          │   index          │  │   detection     │
                       └────────┬────────┘          └────────┬─────────┘  └────────┬────────┘
                                │                          │                     │
                                └──────────────────────────┼─────────────────────┘
                                                           │
                                                           ▼
                                              ┌─────────────────────────┐
                                              │   React Vite Console    │
                                              │                         │
                                              │ • Frame Analysis View   │
                                              │ • Alert Center          │
                                              │ • Semantic Search       │
                                              │ • Session Summaries     │
                                              └─────────────────────────┘
```

## Component Details

### 1. Frame Extractor (`src/frame_extractor.py` / `src/intelligent_frame_extractor.py`)
**Purpose**: Extract frames from drone video and generate telemetry

**Process**:
1. Read video file (MP4/AVI)
2. Extract frames at configurable intervals (hybrid strategy: motion + scene change + uniform)
3. Generate telemetry for each frame:
   - Timestamp (HH:MM:SS)
   - Location (gate, garage, perimeter)
   - Drone position metadata
4. Save to `data/extracted/`

**Key Classes**:
- `IntelligentFrameExtractor`: Motion-based + scene-change + uniform detection
- `TelemetryGenerator`: Creates realistic drone telemetry

### 2. Vision Analyzer (`src/vision_analyzer.py` / `src/cloud_enhanced_analyzer.py`)
**Purpose**: AI-powered frame analysis using Groq/Llama or Gemini Vision

**Process**:
1. Load frame image
2. Send to VLM with structured prompt
3. Extract JSON response:
   - Scene type (interior/exterior/retail/perimeter)
   - People count & features (clothing, actions, position)
   - Objects detected (vehicles, bags, phones, weapons)
   - Security signals (reaching, concealing, loitering, climbing)
4. Generate alerts if suspicious behavior detected

**Key Functions**:
- `analyze_frame()`: Main analysis pipeline
- `_normalize_analysis()`: Extract security signals from person actions
- `_extract_partial_json()`: Handle truncated API responses

**Security Detection Logic**:
```python
suspicious_actions = ["reaching", "concealing", "hiding", "grabbing", "palming", "climbing"]
for person in detected_people:
    for action in person.actions:
        if action in suspicious_actions:
            generate_security_signal(person.id, action)
```

### 3. Alert Engine (`src/alert_engine.py`)
**Purpose**: Generate and manage security alerts

**Alert Levels**:
- **LOW**: Monitor (e.g., "person browsing normally")
- **MEDIUM**: Investigate (e.g., "reaching behavior detected")
- **HIGH**: Immediate action (e.g., "unauthorized access at midnight")
- **CRITICAL**: Threat detected (e.g., weapon or fire hazard)

**Alert Format**:
```json
{
  "severity": "HIGH",
  "alert_type": "Suspicious behavior",
  "frame_id": "frame_017",
  "timestamp": "16:00:16",
  "location": "Garage",
  "message": "Person_2: reaching towards product",
  "recommended_action": "Review frame and continue monitoring"
}
```

### 4. Frame Indexer (`src/pinecone_indexer.py`)
**Purpose**: Store frame embeddings for semantic search

**Technology**: Pinecone Vector Database

**Process**:
1. Generate text embedding from frame description (OpenAI or Pinecone Integrated text-embedding)
2. Store vector with metadata:
   - frame_id
   - timestamp
   - scene_type
   - objects_detected
   - alert_level
3. Enable natural language queries

**Search Examples**:
- Query: "person in red shirt" → Vector similarity search
- Query: "truck at midnight" → Metadata + vector hybrid search

### 5. API Gateway (`src/api.py`)
**Purpose**: FastAPI backend for frontend communication

**Endpoints**:
- `POST /upload-video`: Upload and process video
- `GET /frames`: List all frames
- `GET /alerts`: Get all alerts
- `POST /search`: Semantic search frames
- `GET /session/summary`: Get session summary

### 6. React UI Console (`frontend/`)
**Purpose**: Modern, production-grade UI dashboard built with Vite + React + TypeScript + Glassmorphism CSS.

**Panels**:
1. **Frame Analysis**: View frame + AI analysis + telemetry charts + alerts
2. **Alert Center**: View all alerts by severity
3. **Semantic Search**: Natural language frame search
4. **Session Summary**: Aggregated security report and exporters
5. **Security Agent**: Conversation floating chatbot for security Q&A

## Data Flow

```
Video Upload → Frame Extraction → Vision Analysis → Alert Generation
                    ↓                      ↓              ↓
              Telemetry DB           Frame Index    Alert Store
                    ↓                      ↓              ↓
                    └──────────────────────┴──────────────┘
                                          ↓
                                 React Vite Dashboard
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vision AI | Llama Vision (Groq) / Gemini | Frame analysis, object detection |
| Embeddings | Llama integrated embedding | Vector representations |
| Vector DB | Pinecone | Frame indexing & search |
| Backend | FastAPI | API endpoints |
| Frontend | React + Vite (TypeScript) | Dashboard Console UI |
| Data Storage | MongoDB + local JSON | Telemetry, analysis, alerts |

## Design Decisions

### 1. Multi-Provider Vision Setup
- **Gemini / Groq**: To avoid rate limit 429 locks during active testing, the pipeline dynamically fails over or routes primary analysis to Groq VLM (`meta-llama/llama-4-scout-17b-16e-instruct`) and text agent Q&A to Llama 3.3.

### 2. Deterministic Rule + VLM Alerts
- Combining fast, rule-based heuristics in `src/alert_engine.py` (Layer 1) with conversational LLM safety validation (Layer 2) prevents conservative false negatives from either source.

### 3. Integrated Vector Inference
- Using Pinecone integrated inference eliminates local embedding generation latency, utilizing cloud-hosted text-embedding models directly through the DB upsert pipeline.

## Scalability Considerations

1. **Async Background Pipeline**: Frame analysis and visual indexing processes are delegated to FastAPI background workers.
2. **Rate Limit Defenses**: Adaptive throttle cooldown intervals prevent API locks.
3. **Session-Level Context Isolation**: Each session maintains isolated timeline stores.
