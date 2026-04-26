# Technical Report

## Problem Approach

I built a frame-first security intelligence pipeline for a docked drone context, where each frame is processed with telemetry context and converted into structured operational intelligence.

The implementation focuses on:
- deterministic artifact generation for auditability,
- layered alerting (rules first, LLM reasoning second),
- searchable frame indexing for cross-domain retrieval,
- memory-aware agent recommendations with context continuity.

## Assumptions

- Monitoring source is a fixed-property CCTV-like stream from a docked drone scenario.
- Per-frame reasoning is sufficient for a prototype, with temporal continuity approximated using session context and rolling summaries.
- Alert severity prioritizes safety and false-negative reduction over aggressive suppression.

## Tooling and Configuration Choices

- Vision + reasoning model: GPT-4o for multimodal frame interpretation and structured operational reasoning.
- Embeddings model: text-embedding-3-large (1024 dimensions) for semantic indexing.
- Indexing store: Pinecone Serverless for frame-level retrieval with metadata.
- Agent/memory framework: LangChain ConversationSummaryBufferMemory for bounded reasoning context.
- API/UI: FastAPI + Streamlit for operator-facing access.

## Results Snapshot

Representative outcomes in generated artifacts:
- Frame analysis artifacts with object/person/vehicle/security signals.
- Alert artifacts with severity, rule trigger, and recommendation fields.
- Index artifacts enabling semantic retrieval by object/time intent.
- Session artifacts including context summaries, memory logs, and one-line shift summary.

## What Could Be Improved With More Time

- Introduce true temporal video understanding across frame windows.
- Add stronger calibration and confidence validation for alert thresholds.
- Increase test depth for failure modes and adversarial edge-cases.
- Improve summarizer reliability with strict response schema enforcement.
- Add richer dashboard analytics and operator workflow controls.

## Submission Readiness Notes

- Core prototype pipeline is complete and functional.
- Cross-domain indexing requirement is implemented.
- Bonus capabilities (session summary and Q&A) are included.
- Remaining improvements are primarily documentation polish, test expansion, and report/video packaging.
