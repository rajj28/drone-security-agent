#!/usr/bin/env python3
"""Re-normalize frame analyses and re-index Pinecone for a session."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)
session_id = sys.argv[1] if len(sys.argv) > 1 else "theft_newsflare_e2e_001"
os.environ["SESSION_ID"] = session_id

from src.session_bootstrap import apply_session_layout
from src.config import settings
from src.vision_analyzer import _normalize_analysis
from src.pinecone_indexer import index_frames, search_frames

apply_session_layout(session_id)
analysis_dir = settings.ANALYSIS_DIR
all_results = []
for path in sorted(analysis_dir.glob("frame_*_analysis.json")):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tel_path = settings.TELEMETRY_DIR / f"{data['frame_id']}_telemetry.json"
    tel = (
        json.loads(tel_path.read_text(encoding="utf-8"))
        if tel_path.exists()
        else {"frame_id": data["frame_id"]}
    )
    fixed = _normalize_analysis(data, tel)
    for key in ("frame_id", "timestamp", "location", "model_used", "processing_time_ms"):
        if key in data:
            fixed[key] = data[key]
    path.write_text(json.dumps(fixed, indent=2), encoding="utf-8")
    all_results.append(fixed)
    print(
        data["frame_id"],
        "vlm_chars=",
        len(fixed.get("vlm_description", "")),
        "people=",
        fixed.get("people_count"),
    )

(analysis_dir / "all_analysis.json").write_text(
    json.dumps(all_results, indent=2), encoding="utf-8"
)
log = index_frames()
sr = search_frames("person stealing phone from display", top_k=3)
top = sr["results"][0]["similarity_score"] if sr.get("results") else None
print("indexed:", log["total_indexed"], "top_score:", top)
