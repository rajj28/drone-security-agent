# Why This Architecture — Design Rationale & Interview Guide

This document explains every major design decision in the Drone Security Analyst
Agent, the alternatives that were rejected, and the trade-offs accepted. It is
written in first person so it can be spoken from directly in an interview.

---

## 30-Second Elevator Pitch

> "I built an agentic video-intelligence system: a drone or camera feed goes
> through a staged pipeline — intelligent frame extraction, Gemini VLM scene
> analysis with an optional CLIP+BLIP verification ensemble, vector indexing,
> and a hybrid rule+LLM alert engine — and then an agent layer sits on top
> with tools, memory, and self-validation, so an operator can ask
> natural-language questions about what happened and get grounded, cited
> answers. It runs in production on Fly.io, survives free-tier rate limits
> through multi-key rotation and throttling, and supports live capture from a
> phone or RTSP drone stream."

---

## 1. Staged Pipeline, Not an End-to-End Video Model

**The decision:** Video → Frame Extraction → Telemetry → VLM Analysis → Vector
Index → Alert Engine → Summary. Six explicit stages, each producing inspectable
JSON artifacts.

**Why:**
- **Explainability.** Security is a domain where "the model said so" is not an
  acceptable answer. Every alert traces back to a specific frame, a specific
  VLM description, and a specific rule or LLM validation step. An operator (or
  a court) can audit the chain.
- **Debuggability & recovery.** When a stage fails (a 429, a corrupt frame), I
  re-run *that stage*, not the whole pipeline. Each stage's output is
  checkpointed to disk/MongoDB, so the pipeline is resumable.
- **Cost control.** Vision-LLM calls are the expensive resource. A staged design
  lets me put the throttle (`API_MIN_INTERVAL_SEC`), retry, and provider
  fallback exactly at the stage boundary where the money is spent.
- **Independent evolution.** I migrated the vision provider twice during
  development (GPT-4o → Groq → Gemini Flash, which is now the production
  provider everywhere) without touching extraction, indexing, or alerting.
  That's the payoff of stage isolation.

**Rejected alternative:** a single multimodal call ("here's a video, find
threats"). Rejected because it's a black box, cost scales with video length not
information content, and long-video context degrades detection quality.

**Trade-off accepted:** more moving parts and inter-stage plumbing. Mitigated
with a unified session context object shared by all stages.

---

## 2. Intelligent Frame Extraction (the Cost-Aware Autonomy Argument)

**The decision:** Frames are selected by a hybrid strategy — motion detection +
scene-change detection + quality gating (blur/darkness rejection) — instead of
naive fixed-FPS sampling. Uniform sampling remains available as a strategy flag.

**Why this is the most "agentic" decision in the system:** an agent's core
skill is *deciding where to spend limited compute*. A 10-minute patrol video at
1 FPS is 600 VLM calls; 590 of them show an empty parking lot. Motion and
scene-change gating spends the API budget only on frames that carry signal.
This is the same principle as an LLM agent deciding which tool call is worth
making.

**Trade-off accepted:** a fast, subtle event could slip between motion
triggers. That's why the strategy is configurable per-run (`hybrid`, `motion`,
`uniform`) — incident clips that are *all* action use uniform; long patrols use
hybrid.

---

## 3. One Production Provider — Gemini Everywhere — Engineered for Reliability

**The decision:** Gemini is the single production LLM: Gemini Flash VLM for
vision analysis, Gemini text for alert validation, session summaries, and the
Q&A agent. Around that single provider I built a reliability layer: every call
is rate-limited by a global minimum-interval throttle (`API_MIN_INTERVAL_SEC`),
retried with exponential backoff + jitter, and load-shared **round-robin across
multiple Gemini API keys** to multiply quota. A flash-only mode
(`GEMINI_FLASH_ONLY`) prevents fallback calls from burning Pro-tier quota, and
if the API is fully unavailable an offline OpenCV-heuristic fallback keeps the
pipeline producing (degraded, clearly-flagged) output.

**Why one provider instead of a multi-vendor zoo:**
- **Consistency of judgment.** Threat assessment must be comparable across
  frames and sessions. Mixing vendors mid-pipeline means frame 3 and frame 4
  get scored by models with different calibrations — a real problem for a
  system whose output is a severity level.
- **One prompt-engineering surface.** The structured-JSON vision prompts and
  the alert-validation prompts are tuned against one model family's behavior.
  Every additional vendor doubles that tuning and testing burden.
- **Operational simplicity with the same resilience.** The failure mode that
  actually matters on a free/low tier is *rate limiting*, and multi-key
  rotation + throttling + backoff solves it without cross-vendor complexity.

**Deliberate escape hatch:** provider selection still lives behind env config
(`VISION_PROVIDER`, `AGENT_LLM_PROVIDER`), so swapping or A/B-testing another
provider is a config change, not a rewrite. I keep the abstraction; I don't pay
the multi-vendor operational cost until there's a reason to.

**Trade-off accepted:** single-vendor dependency. Mitigated by the abstraction
layer, the offline fallback, and — for *correctness* rather than availability —
the CLIP+BLIP verification ensemble (next section).

---

## 3b. Cloud Enhancers: CLIP + BLIP as a Verification Ensemble (Anti-Hallucination)

**The decision:** A production mode — toggled by an **"Enable Cloud
Enhancers"** switch in the dashboard — runs two additional, independent vision
models per frame via the HuggingFace Inference API alongside Gemini:
- **CLIP** (`openai/clip-vit-base-patch32`): zero-shot classification of the
  frame against a bank of security-behavior prompts, producing an independent
  threat score and categories.
- **BLIP** (`Salesforce/blip-image-captioning-base` + VQA): an independent
  scene caption and keyword extraction.

Their outputs are merged into the frame analysis next to Gemini's, so the alert
engine and the operator see *agreement or disagreement* between models.

**Why — this is my hallucination-mitigation architecture:**
- **Uncorrelated errors.** Gemini, CLIP, and BLIP are different architectures
  trained by different organizations on different data. When a generative VLM
  hallucinates ("person holding a weapon" that isn't there), a discriminative
  zero-shot classifier like CLIP is very unlikely to hallucinate the *same*
  thing. Agreement raises confidence; disagreement is a signal to downgrade or
  flag for human review.
- **Discriminative checks generative.** CLIP doesn't generate text at all — it
  scores fixed hypotheses. That makes it a cheap, fast verifier for exactly the
  class of error (confident fabrication) that generative models make.
- **Cost-tiered by design.** The ensemble is a per-run toggle, not always-on:
  demo/dev runs use Gemini alone (fast, cheap); production or high-stakes runs
  enable the enhancers and pay the extra latency for verified analysis. The
  toggle flows from the UI through the API (`use_cloud_enhancers`) into the
  vision stage — the same cost-aware-autonomy principle as frame gating.
- **Why HF Inference API instead of self-hosting:** no local GPU requirement,
  cloud-GPU latency, and a free tier — the models run *near* their weights
  instead of shipping a 4GB container.

**Trade-off accepted:** 2 extra API calls per frame and HF cold-start latency
when enabled. That's exactly why it's a toggle and not a default.

---

## 4. Hybrid Alert Engine: Deterministic Rules + LLM Validation

**The decision:** Alerts are generated in two phases. Phase 1 is rule-based
(`rule_based_alert`): explicit checks like behavioral-threat and
shoplifting-pattern predicates over the structured VLM output. Phase 2 is LLM
validation (`llm_validate_alert`): an LLM agent with conversation memory
reviews the candidate alert in context and confirms, escalates, or suppresses
it with a reasoning string.

**Why — and this is the answer I'd defend hardest:**
- **Pure rules are brittle.** "Person near fence after 22:00" misses every
  novel threat pattern.
- **Pure LLM is unaccountable and expensive.** LLM-only alerting hallucinates,
  can't guarantee that a defined critical condition *always* fires, and costs a
  call per frame per rule.
- **Rules-then-LLM gives the best of both:** the rules layer provides
  *guaranteed recall* on known-critical patterns at zero marginal cost, and the
  LLM layer provides *precision* (false-positive suppression) and novel-pattern
  reasoning. The LLM never gets to silently delete a critical rule hit — it
  annotates and can only escalate severity on those.

This mirrors how mature ML systems deploy models behind deterministic
guardrails, and it's directly transferable to agent safety: constrain with
hard rules, reason with the model.

---

## 5. The Agent Layer: Tools + Memory + Grounding (LangChain)

**The decision:** The Q&A "Security Agent" is a tool-using agent with:
- **Tools:** `analyze_frame`, `search_frames` (vector search),
  `get_telemetry`, `trigger_alert`, `get_session_context`,
  `query_event_history` — the agent *chooses* which to call per question.
- **Memory:** `ConversationSummaryBufferMemory` plus a persisted memory log and
  rolling context summaries, so follow-up questions ("what did *he* do next?")
  resolve against prior turns and prior sessions.
- **Grounding:** answers cite frame IDs; the UI makes citations clickable so
  the operator can jump to the evidence.

**Why:**
- **Tool use is the core agentic pattern.** The agent doesn't get the whole
  session dumped into its prompt — it *retrieves* what it needs. That keeps
  token cost flat as sessions grow and answers grounded in actual data.
- **Memory is what separates an agent from a chatbot.** Security review is
  inherently multi-turn ("show me the alerts" → "why was #2 medium?" →
  "compare with the earlier frame").
- **Citations kill hallucination at the UX level.** If the agent can't point
  to a frame, the operator knows to distrust the claim.

---

## 6. Multi-Agent Orchestration (Analysis / QA / Question Agents)

**The decision:** Higher-level reasoning is split across three specialized
Gemini-powered agents under an orchestrator: an Analysis Agent (pattern +
risk + temporal reasoning), a QA Agent (validation + confidence scoring of
other agents' outputs), and a Question Agent (NL search + Q&A routing).

**Why:**
- **Separation of concerns beats one mega-prompt.** A prompt that must analyze,
  self-check, and answer questions simultaneously does all three worse. Small
  focused prompts are also independently testable.
- **The QA agent is machine self-verification** — an agent checking another
  agent's work with a confidence score before it reaches the operator. This is
  the pattern behind every serious agentic deployment (reviewer/critic loops).
  Combined with the CLIP+BLIP ensemble at the perception layer, the system
  verifies itself at *two* levels: models cross-check perception, agents
  cross-check reasoning.
- **Right-sizing compute per role:** cheap discriminative models (CLIP) for
  verification, a fast VLM (Gemini Flash) for perception, and the heavier
  reasoning prompts only at the orchestration layer where depth pays for
  itself. Allocating model capacity per role is a cost/latency decision an
  agentic engineer must make constantly.

---

## 7. Vector Indexing (Pinecone) for Cross-Domain Search

**The decision:** Every frame's VLM description + telemetry is embedded and
indexed in Pinecone; semantic search is exposed both in the UI and as an agent
tool.

**Why:** the requirement was frames "queryable by time and object." Keyword
matching fails on paraphrase ("guy grabbing phones" vs "person handling
merchandise"). Embedding search makes the *agent's* retrieval robust too — the
`search_frames` tool is the same index, so one investment serves both the human
UI and the agent's grounding. Serverless Pinecone was chosen over self-hosted
FAISS because session data must survive stateless container restarts.

---

## 8. Persistence: MongoDB Sessions over Filesystem

**The decision:** Session state (frames, analyses, alerts, chat logs,
processing status) persists to MongoDB Atlas; the filesystem layout remains as
a local/dev fallback behind a flag (`USE_MONGO_CONTEXT`).

**Why:** cloud containers are ephemeral — Fly machines suspend, Cloud Run
scales to zero. Any state worth keeping must live outside the container. Also
enables multiple concurrent sessions and horizontal scale-out later (any
machine can serve any session). Document DB fits because pipeline artifacts are
already JSON documents with evolving schemas — no migration tax during rapid
iteration.

---

## 9. API Design: Async FastAPI + Background Pipelines + Cancellation

**The decision:** Upload returns immediately with a session ID; the pipeline
runs as a background task; the client polls `/processing-status/{id}`; a
heartbeat task keeps status fresh during long stages; and a cancellation module
lets the user abort a running pipeline cleanly.

**Why:** a 10-minute pipeline cannot live inside an HTTP request (timeouts,
disconnects, proxy limits). Status polling was chosen over WebSockets
deliberately: it's stateless, survives machine suspend/resume on Fly, and works
through every proxy. Cancellation matters because on free-tier rate limits an
accidental 100-frame run holds the API budget hostage — an agentic system
must be *interruptible*.

---

## 10. Live Capture: Browser-Push and Server-Pull, Two Deliberate Modes

**The decision:** two ingestion modes sharing one pipeline:
1. **Browser-push** — the phone's camera captures via `getUserMedia`, JS
   samples frames to canvas and POSTs JPEGs to `/live/{sid}/frame`.
2. **Server-pull** — the backend connects to an RTSP/HTTP drone-camera stream
   and samples frames itself.

**Why browser-push for phones:** the phone is behind NAT on a mobile network —
the server can't reach it, but it can always reach the server. Client-side
sampling (1 frame / N sec as JPEG) also uses ~1000x less bandwidth than
streaming video that would mostly be thrown away — the *pipeline* only wants
sparse frames anyway. Why server-pull for drones: real drone/IP cameras speak
RTSP and can't run JavaScript. Both modes converge on the same session format,
so everything downstream (analysis, alerts, chat) is ingestion-agnostic.

---

## 11. Deployment: Fly.io Machines over Cloud Run (a Migration Story)

**The decision:** deployed first on GCP Cloud Run, then migrated to Fly.io
Machines (shared-cpu-4x / 2GB, suspend-on-idle).

**Why the migration — real operational lessons:**
- **Cloud Run throttles CPU outside request handling.** My pipeline runs as a
  background task *after* the upload response — exactly the execution model
  Cloud Run starves. Video decoding at throttled CPU was unusable.
- **Cold starts:** Cloud Run scale-to-zero meant full container cold boots.
  Fly's `suspend` resumes a machine in ~1s with process state intact.
- **Cost predictability** on a personal budget: one always-suspendable machine
  beats per-request billing with minimum instances.

Being able to say "I deployed on X, measured, found these failure modes, and
migrated to Y" is worth more than any diagram.

---

## 12. Engineering for Failure (the Underrated Section)

Everything here exists because something broke in practice:
- **Retry with exponential backoff + jitter** on all API calls (`api_retry`).
- **Global min-interval throttle** between paid API calls, tunable per run.
- **Multi-key round-robin** for Gemini to multiply free-tier quota.
- **Offline vision fallback** (OpenCV heuristics) when every provider is down.
- **Frame quality gating** so blurred/dark frames don't waste VLM calls.
- **Resumable stages** (`--skip-extraction`, `--skip-vision`) for recovery.
- **42 pytest tests** covering the assignment requirements end-to-end, plus a
  "harsh testing" suite that attacks edge cases.

---

## Mapping to "Agentic AI Engineer" Competencies

| Agentic concept | Where it lives in this project |
|---|---|
| Tool use / function calling | Security Agent's 6 tools (search, telemetry, alerts…) |
| Memory (short + long term) | ConversationSummaryBufferMemory + persisted memory log + context summaries |
| Planning / decomposition | Staged pipeline; orchestrator routing across 3 agents |
| Self-verification | QA Agent validating Analysis Agent output with confidence scores; LLM validation over rule alerts |
| Grounding / anti-hallucination | CLIP+BLIP verification ensemble; frame-ID citations; vector retrieval instead of prompt-stuffing |
| Guardrails | Deterministic rule layer the LLM cannot override downward |
| Cost-aware autonomy | Motion-gated frame selection; cloud-enhancer toggle (pay for verification only when it matters); throttles |
| Resilience | Multi-key rotation, retries with backoff, offline degradation, cancellation |
| Production operation | Fly.io deployment, MongoDB persistence, status/heartbeat, mobile-responsive operator UI |

---

## Likely Interview Questions & Answers

**"Why not fine-tune a model instead of prompting?"**
No labeled drone-security dataset exists at useful scale, threat definitions
change faster than retraining cycles, and zero-shot VLMs already read scenes
well. Fine-tuning would be the right call later for the *embedding* model if
retrieval precision became the bottleneck — I'd know because retrieval quality
is measurable from the QA agent's confidence scores.

**"Why not YOLO/classical CV for detection?"**
I do use classical CV — where it's the right tool: motion/scene-change gating
and the offline fallback. But object detection ("person", "box") isn't threat
understanding ("person concealing merchandise behind the counter"). The VLM
gives semantic, describable, queryable scene understanding; YOLO gives bounding
boxes. And where I need a discriminative model's reliability, I use CLIP in the
cloud-enhancer ensemble — zero-shot, so it adapts to new threat categories by
editing a prompt list instead of retraining.

**"How do you handle hallucination?"**
Five layers: structured JSON output contracts on VLM calls; the **CLIP+BLIP
verification ensemble** — independent discriminative models cross-checking
Gemini's perception, togglable per run for production; the deterministic rule
layer that doesn't depend on LLM honesty; the QA agent scoring confidence on
generated analyses; and citations-to-frames in the chat UI so unverifiable
claims are visible as such. The ensemble is the key one: generative models
fabricate confidently, and the cheapest defense is a second, *architecturally
different* model that scores fixed hypotheses instead of generating.

**"How would you scale this to 100 drones?"**
The stages are already decoupled with checkpointed state, so the path is:
replace in-process background tasks with a work queue (e.g. Redis + workers),
scale vision workers horizontally (stateless — state is in MongoDB/Pinecone),
shard sessions by drone ID, and move alert fan-out to pub/sub. The design was
staged precisely so this migration is mechanical, not a rewrite.

**"What would you do differently?"**
Add an evaluation harness earlier — golden-set videos with labeled expected
alerts, scored per pipeline change. I tested requirements (42 tests) but
regression-testing *judgment quality* needs eval sets, and that's the first
thing I'd build with more time.

---

*System: Python/FastAPI · React/TS · Gemini Flash (vision + reasoning) ·
HF CLIP+BLIP cloud enhancers · LangChain/LangGraph · Pinecone · MongoDB ·
Fly.io · Live demo: https://drone-security-agent.fly.dev*
