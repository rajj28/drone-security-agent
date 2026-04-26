# Drone Security Analyst Architecture

## Pipeline Overview

This prototype uses a staged pipeline so every frame can be traced from raw video to alert and retrieval.

```mermaid
flowchart TD
    A[Input .dav Video] --> B[Frame Extractor]
    B --> C[Telemetry Generator]
    B --> D[Vision Analyzer GPT-4o]
    C --> D
    D --> E[Alert Engine\nRule + LLM Reasoning]
    D --> F[Pinecone Indexer\nEmbeddings + Metadata]
    E --> G[Session Context + Incident Log]
    F --> H[Semantic Search]
    G --> I[Agent Reasoning Layer]
    H --> I
    I --> J[FastAPI Endpoints]
    I --> K[Streamlit Dashboard]
    G --> L[Summarizer + QA Agent]
```

## Pipeline Narrative

1. Frame extraction reads the DVR video and writes frame images to `data/extracted`.
2. Telemetry generation creates per-frame operational context such as location, timestamp, restricted-zone flag, and after-hours flag.
3. Vision analysis calls GPT-4o vision and emits a structured JSON record per frame with objects, people, vehicles, activity, and threat signals.
4. Alert engine applies deterministic rules first, then uses an LLM reasoning layer for validation/override context.
5. Pinecone indexing stores frame descriptions and metadata for cross-domain retrieval (time/object/semantic query).
6. Stateful agent consumes analysis + telemetry + alerts + retrieval context to produce recommendations.
7. FastAPI and Streamlit expose results for operators, while summarizer and QA produce shift-level intelligence outputs.

## Data Contracts

### Telemetry Contract
- `frame_id`
- `timestamp`
- `location`
- `drone`
- `is_after_hours`
- `is_restricted_zone`

### Vision Analysis Contract
- `frame_id`
- `timestamp`
- `location`
- `vlm_description`
- `scene_type`
- `objects_detected`
- `object_details`
- `people_count`
- `person_features`
- `vehicles_detected`
- `vehicle_details`
- `activity`
- `security_signals`
- `suspicious_elements`
- `threat_assessment`
- `confidence`
- `recommended_action`
- `alert_reasoning`
- `alert_priority_signals`

### Alert Contract
- `alert_id`
- `frame_id`
- `timestamp`
- `location`
- `alert_triggered`
- `severity`
- `alert_type`
- `message`
- `objects_involved`
- `rule_triggered`
- `llm_validated`
- `llm_reasoning`
- `recommended_action`
- `acknowledged`
- `auto_escalate`

### Session Context Contract
- `frames_analyzed`
- `total_alerts`
- `high_alerts`
- `medium_alerts`
- `low_alerts`
- `people_detected`
- `vehicles_detected`
- `locations_visited`
- `incidents`
- `agent_processed_frames`
- `alert_processed_frames`
- `running_narrative`

## Alert Flow

1. Vision analysis produces frame-level threat metadata.
2. Rule engine applies deterministic security policies.
3. LLM alert reasoning layer validates or adjusts severity for ambiguous cases.
4. Alert payload is persisted and rolled into session context and context summaries.
5. High alerts can be flagged for escalation.

## Memory Flow

1. Agent memory uses `ConversationSummaryBufferMemory` plus `outputs/session/agent_memory_log.json`.
2. Alert memory uses `ConversationSummaryBufferMemory` plus `outputs/session/alert_memory_log.json`.
3. Shared context timeline uses `outputs/session/context_summaries.json` for cumulative reasoning.

## Scalability Notes

- Frame-level artifacts are append-only JSON, enabling replay and auditing.
- Pinecone vector index enables semantic retrieval at larger history sizes.
- Reasoning layers consume only a rolling context window to bound token usage.
- Duplicate frame guards prevent accidental double counting during reruns.
