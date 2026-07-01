# Production Pipeline Evaluation
- **Session**: `theft_newsflare_008`
- **Production ready**: True (5/5 checks)
- **Rate limit hits (import side)**: 0

## VLM understanding
- Pass: **True** (12/12 frames)

## Agent responses
- Pass: **True**
- Q: What suspicious activity happened in this session?… → hits: ['phone', 'steal', 'retail', 'shop', 'person', 'suspicious']
- Q: How many people were visible?… → hits: ['people']
- Q: Was there activity near a display case or counter?… → hits: ['display', 'counter', 'phone', 'shop']

## Pinecone
- Pass: **True**
- `person stealing phone` → score 0.4216 (frame_011)
- `suspicious activity in retail store` → score 0.4443 (frame_006)