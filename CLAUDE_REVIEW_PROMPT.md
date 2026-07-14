# Second Opinion Request: Drone Security Analyst Agent

## Context
I'm building a Drone Security Analyst Agent for a university assignment. The system processes drone surveillance videos to detect security threats in retail environments (shops, warehouses, parking lots).

## Current Architecture

### Tech Stack
- **Frame Extraction**: OpenCV with intelligent sampling (motion + scene change detection)
- **Vision Analysis**: Cloud-enhanced pipeline combining:
  - Hugging Face CLIP (cloud API) - visual threat scoring
  - Hugging Face BLIP (cloud API) - image captioning
  - OpenAI GPT-4o Vision (local) - deep analysis with cloud context
- **Vector Search**: Pinecone for frame indexing and semantic search
- **Backend**: FastAPI
- **Frontend**: Streamlit dashboard
- **Alert Engine**: Rule-based + LLM validation

### Detection Logic
1. Extract frames from uploaded video (max 20 frames)
2. Run cloud CLIP + BLIP in parallel for quick visual understanding
3. Feed CLIP/BLIP results as context to GPT-4o for deep analysis
4. Parse GPT-4o output for person actions and behaviors
5. Extract security signals from text: "reaching", "concealing", "loitering", etc.
6. Generate alerts if suspicious patterns detected

### Current Issue
- GPT-4o detection is **inconsistent** - same frame sometimes detects "reaching towards shelf" as suspicious, sometimes describes it as "examining products" (benign)
- Running frame 017 (mobile phone shop scene) through the pipeline yields inconsistent threat assessments
- Need 90%+ accuracy for assignment requirements

## My Questions

1. **Is GPT-4o Vision the right choice** for consistent security threat detection, or should I switch to:
   - Fine-tuned specialized models (YOLO + behavior classifier)
   - Open source LLaVA models (Groq API - free tier)
   - Gemini 1.5 Flash (cheaper, faster)
   - Hybrid: YOLO for detection + GPT-4o only for behavior analysis

2. **Prompt Engineering**: My current prompt is neutral to avoid OpenAI content policy refusals. Should I:
   - Make it explicitly security-focused (risk: content policy blocks)
   - Keep it neutral but add retry logic with stronger prompts
   - Use two-stage: neutral first, then targeted follow-up if needed

3. **Consistency Problem**: How to make detection deterministic:
   - Temperature = 0 (already set)
   - Few-shot prompting with examples
   - Structured output forcing (JSON mode)
   - Ensemble: run multiple times, majority vote

4. **Free Alternatives**: For a student assignment with $0 budget, what's the best completely free stack:
   - Option A: YOLOv8 (local) + Hugging Face BLIP (free API) + rule-based threats
   - Option B: LLaVA on Groq (free tier) for everything
   - Option C: MediaPipe pose detection (local) + simple heuristics
   - Option D: Roboflow pre-trained shoplifting models (free tier)

5. **Assignment Requirements Check** - Does my approach meet these:
   - Real-time video analysis with telemetry
   - Object/event detection and logging
   - Real-time security/safety alerts
   - Frame-by-frame indexing and search (Pinecone)
   - Cross-domain: Does "drone telemetry + vision AI" count as cross-domain?

## Expected Outputs (Assignment Examples)

**Log entry**: "Blue Ford F150 spotted at garage, 12:00. Confidence: 0.92"

**Alert**: 
```json
{
  "severity": "MEDIUM",
  "type": "Person loitering at main gate",
  "timestamp": "00:01",
  "location": "main_gate"
}
```

**Search query**: "show all truck events today" → [frame_005, frame_012, frame_034]

## Constraints
- Must be completable in 1 week (remaining time)
- Must work for demo video (retail theft scenario)
- Must detect: people, vehicles, suspicious behaviors (reaching, loitering)
- Must generate alerts with reasoning
- Budget: Prefer $0, can spend $10-20 on APIs if needed

## What I've Tried
1. Cloud CLIP + BLIP + GPT-4o pipeline working
2. Security signal extraction from person_features text
3. Alert generation with reasoning
4. Inconsistent detection of "reaching" as suspicious vs benign
5. Frame indexing works but semantic search needs improvement

## Request
Please review my approach and suggest:
1. **Critical fixes** needed for consistent threat detection
2. **Architecture changes** if current approach is fundamentally flawed
3. **Best free/cheap model combination** for 90%+ accuracy
4. **Quick wins** I can implement in 2-3 days

Be honest - if GPT-4o is wrong tool for this job, tell me what to switch to and why.
