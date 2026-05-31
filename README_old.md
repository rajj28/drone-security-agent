# Drone Security Analyst Agent

A production-grade, evaluation-ready AI system for automated drone-based CCTV security analysis. Built for FlytBase job assignment.

---

## Feature Spec

### Value Proposition
The Drone Security Analyst Agent improves property security by continuously interpreting surveillance frames plus telemetry, detecting abnormal activity early, and presenting actionable alerts and searchable incident history to operators.

### Measurable Requirements
1. The system must process and persist at least 50 frame-level analysis records per monitoring run, each linked to telemetry context (`timestamp`, `location`, zone flags).
2. The system must generate deterministic rule-based alerts and save a structured alert artifact per frame, with severity and operator recommendation fields.
3. The system must provide frame-by-frame indexing with queryable retrieval by semantic text/object/time context through the indexing/search layer.

---

## Features
- Extracts frames from .dav CCTV video
- Analyzes each frame with GPT-4o Vision
- Generates and saves telemetry JSON per frame
- Indexes frame descriptions in Pinecone (semantic search)
- LangChain agent with tools, memory, and session context
- Two-layer alert engine (rule-based + LLM validation)
- Session summarizer and Q&A agent (BONUS)
- FastAPI backend and Streamlit dashboard
- All outputs saved as human-readable JSON for evaluation

---

## Project Structure
```
drone-security-agent/
├── src/
│   ├── config.py
│   ├── frame_extractor.py
│   ├── telemetry_generator.py
│   ├── vision_analyzer.py
│   ├── pinecone_indexer.py
│   ├── alert_engine.py
│   ├── agent.py
│   ├── summarizer.py
│   └── qa_agent.py
├── data/
│   ├── frames/
│   └── extracted/
├── outputs/
│   ├── telemetry/
│   ├── analysis/
│   ├── alerts/
│   ├── index/
│   └── session/
├── tests/
│   └── test_agent.py
├── demo/
│   └── dashboard.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Systerm Acrhitecture
<img width="1617" height="768" alt="image" src="https://github.com/user-attachments/assets/e44b54a6-f2f9-4dcb-952d-bcea89b0a950" />

---

## Setup
1. Clone repo and install requirements:
   ```sh
   pip install -r requirements.txt
   ```
2. Add your API keys and config to `.env` (see template in assignment).
3. Place your .dav video in `data/frames/`.
4. Run the pipeline:
   ```sh
   python src/frame_extractor.py
   python src/telemetry_generator.py
   python src/vision_analyzer.py
   python src/pinecone_indexer.py
   python src/alert_engine.py
   python src/summarizer.py
   python src/qa_agent.py  # (optional, for demo Q&A)
   ```
5. Start API:
   ```sh
   uvicorn src.api:app --reload
   ```
6. Start dashboard:
   ```sh
   streamlit run demo/dashboard.py
   ```
7. Run tests:
   ```sh
   pytest
   ```

---

## Evaluation Outputs
All outputs are saved as JSON in the `outputs/` directory. Judges can inspect these for every metric.

---

## AI Tools Used

### What AI Generated (Initial Drafts)
- Initial scaffolding for some module structures and prompt templates.
- Baseline alert-rule ideas and first-pass test/checklist suggestions.
- Early documentation drafts for setup and execution steps.

### What Was Customized by Me
- Final architecture decisions, pipeline wiring, and data contract definitions.
- Memory hardening design (`agent_processed_frames`, `alert_processed_frames`) and separation of memory logs.
- Enriched vision schema for person/vehicle/security signals and downstream reasoning integration.
- Alert reasoning flow, risk-driven fixes, and deterministic evaluation harness.
- Validation workflow, debugging, and final artifact curation for assignment compliance.

---

## Notes
- All code is production-quality, fully typed, and documented.
- All outputs are human-readable (indent=2).
- Progress and errors are logged with emojis.
- Bonus features (session summary, Q&A) are included.


