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
└─────────────────┘     │ • Generate       │     │ • Object detection          │
                        │   telemetry      │     │ • Behavior analysis         │
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
                                              │    Streamlit Dashboard  │
                                              │                         │
                                              │ • Frame Analysis View   │
                                              │ • Alert Center          │
                                              │ • Semantic Search       │
                                              │ • Session Summaries     │
                                              └─────────────────────────┘
```

## Component Details

### 1. Frame Extractor (`src/frame_extractor.py`)
**Purpose**: Extract frames from drone video and generate telemetry

**Process**:
1. Read video file (MP4/AVI)
2. Extract frames at configurable intervals (hybrid strategy: motion + scene change)
3. Generate telemetry for each frame:
   - Timestamp (HH:MM:SS)
   - Location (gate, garage, perimeter)
   - Drone position metadata
4. Save to `data/extracted/`

**Key Classes**:
- `IntelligentFrameExtractor`: Motion-based + scene-change detection
- `TelemetryGenerator`: Creates realistic drone telemetry

### 2. Vision Analyzer (`src/vision_analyzer.py`)
**Purpose**: AI-powered frame analysis using GPT-4o Vision

**Process**:
1. Load frame image
2. Send to GPT-4o with structured prompt
3. Extract JSON response:
   - Scene type (interior/exterior/retail)
   - People count & features (clothing, actions, position)
   - Objects detected (vehicles, bags, phones)
   - Security signals (reaching, concealing, loitering)
4. Generate alerts if suspicious behavior detected

**Key Functions**:
- `analyze_frame()`: Main analysis pipeline
- `_normalize_analysis()`: Extract security signals from person actions
- `_extract_partial_json()`: Handle truncated API responses

**Security Detection Logic**:
```python
suspicious_actions = ["reaching", "concealing", "hiding", "grabbing", "palming"]
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

**Alert Format**:
```json
{
  "severity": "MEDIUM",
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
1. Generate text embedding from frame description (OpenAI `text-embedding-3-large`)
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

### 6. Dashboard (`demo/dashboard.py`)
**Purpose**: Streamlit UI for visualization and interaction

**Tabs**:
1. **Frame Analysis**: View frame + AI analysis + alerts
2. **Alert Center**: View all alerts by severity
3. **Semantic Search**: Natural language frame search
4. **Session Summary**: Aggregated security report
5. **Security Agent**: Q&A about video content

## Data Flow

```
Video Upload → Frame Extraction → Vision Analysis → Alert Generation
                    ↓                      ↓              ↓
              Telemetry DB           Frame Index    Alert Store
                    ↓                      ↓              ↓
                    └──────────────────────┴──────────────┘
                                          ↓
                              Streamlit Dashboard
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vision AI | GPT-4o (OpenAI) | Frame analysis, object detection |
| Embeddings | OpenAI text-embedding-3-large | Vector representations |
| Vector DB | Pinecone | Frame indexing & search |
| Backend | FastAPI | API endpoints |
| Frontend | Streamlit | Dashboard UI |
| Data Storage | JSON files + Vector DB | Telemetry, analysis, alerts |

## Design Decisions

### 1. Why GPT-4o vs. CLIP/BLIP?
- **GPT-4o**: Best for complex scene understanding and behavior analysis
- **CLIP**: Good for object matching but limited scene context
- **BLIP**: Good for captioning but misses security nuances
- **Decision**: GPT-4o provides end-to-end analysis with reasoning

### 2. Why Pinecone vs. Local Index?
- **Pinecone**: Managed service, fast similarity search, metadata filtering
- **Local (FAISS)**: Free but requires management and scaling
- **Decision**: Pinecone for production-ready search with minimal setup

### 3. Why FastAPI + Streamlit?
- **FastAPI**: Async, auto-docs, easy integration
- **Streamlit**: Rapid UI development, perfect for demos
- **Decision**: Fast backend + quick frontend for assignment timeline

## Scalability Considerations

1. **Async Processing**: Frame analysis uses background tasks
2. **Batch Processing**: Multiple frames can be analyzed in parallel
3. **Incremental Indexing**: New frames added to index immediately
4. **Session Isolation**: Each video upload creates isolated session

## Security & Privacy

1. **No Face Recognition**: System focuses on behavior, not identity
2. **No Data Retention**: JSON files local, no external logging
3. **API Key Security**: Environment variables for all keys
