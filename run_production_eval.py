#!/usr/bin/env python3
"""
Production-style pipeline evaluation: full run + VLM/agent/Pinecone scoring.

Usage:
  python run_production_eval.py [--max-frames 5] [--session-id prod_eval_001]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "outputs" / "evaluation"

THEFT_KEYWORDS = (
    "phone",
    "mobile",
    "retail",
    "shop",
    "store",
    "steal",
    "theft",
    "snatch",
    "grab",
    "display",
    "counter",
    "person",
    "man",
    "people",
)
AGENT_QUESTIONS = [
    ("What suspicious activity happened in this session?", ["phone", "steal", "retail", "shop", "person", "suspicious"]),
    ("How many people were visible?", ["people", "person", "0", "one", "two", "three", "four", "five", "several", "multiple"]),
    ("Was there activity near a display case or counter?", ["display", "counter", "case", "phone", "shop"]),
]
PINECONE_QUERIES = [
    ("person stealing phone", 0.25),
    ("suspicious activity in retail store", 0.20),
]


def _score_vlm(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not analyses:
        return {"pass": False, "reason": "no analyses"}
    scored = []
    for a in analyses:
        if not a:
            continue
        desc = (a.get("vlm_description") or "").lower()
        bad = desc in ("", "scene analysis unavailable.")
        hits = [k for k in THEFT_KEYWORDS if k in desc]
        people = int(a.get("people_count") or 0)
        scored.append(
            {
                "frame_id": a.get("frame_id"),
                "desc_chars": len(desc),
                "keyword_hits": hits,
                "people_count": people,
                "threat": a.get("threat_level") or a.get("threat_assessment"),
                "model": a.get("model_used"),
                "parsing_failed": bool(a.get("_parsing_failed")),
                "understanding_ok": not bad and len(hits) >= 2,
            }
        )
    ok = sum(1 for s in scored if s["understanding_ok"])
    return {
        "pass": ok >= max(1, len(scored) * 0.6),
        "frames_ok": ok,
        "frames_total": len(scored),
        "per_frame": scored,
    }


def _score_agent(answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for q, expected_keywords in AGENT_QUESTIONS:
        match = next((a for a in answers if a.get("question") == q), None)
        if not match:
            results.append(
                {
                    "question": q,
                    "keyword_hits": [],
                    "answer_preview": "",
                    "ok": False,
                    "missing": True,
                }
            )
            continue
        text = (match.get("answer") or match.get("error") or "").lower()
        hits = [k for k in expected_keywords if k in text]
        results.append(
            {
                "question": q,
                "keyword_hits": hits,
                "answer_preview": text[:200],
                "ok": len(hits) >= 1 and "error" not in match,
            }
        )
    ok = sum(1 for r in results if r["ok"])
    return {"pass": ok >= 2, "scored": results}


def _score_pinecone(search_results: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for query, min_score in PINECONE_QUERIES:
        sr = search_results.get(query, {})
        top = sr.get("results", [{}])[0] if sr.get("results") else {}
        score = top.get("similarity_score") or 0
        rows.append(
            {
                "query": query,
                "top_score": score,
                "top_frame": top.get("frame_id"),
                "pass": score >= min_score,
            }
        )
    return {"pass": any(r["pass"] for r in rows), "queries": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--session-id", default="prod_eval_001")
    parser.add_argument("--skip-pipeline", action="store_true", help="Only evaluate existing session")
    args = parser.parse_args()

    video = ROOT / "Sneaky Thieves Caught Stealing Phones On Camera - Newsflare (1080p, h264) (1).mp4"
    if not video.exists():
        for p in ROOT.glob("*.mp4"):
            if "thieves" in p.name.lower() or "sneaky" in p.name.lower():
                video = p
                break

    report: Dict[str, Any] = {
        "eval_type": "production_pipeline",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "session_id": args.session_id,
        "video": str(video),
        "config": {},
    }

    env = os.environ.copy()
    env["SESSION_ID"] = args.session_id
    env["GEMINI_PREFER_FLASH"] = "true"
    env["GEMINI_FLASH_ONLY"] = env.get("GEMINI_FLASH_ONLY", "true")
    env["API_MIN_INTERVAL_SEC"] = env.get("API_MIN_INTERVAL_SEC", "15")
    env["USE_MONGO_CONTEXT"] = "false"
    report["config"] = {
        "gemini_prefer_flash": True,
        "api_min_interval_sec": env["API_MIN_INTERVAL_SEC"],
        "max_frames": args.max_frames,
    }

    pipeline_rc = 0
    if not args.skip_pipeline:
        cmd = [
            sys.executable,
            str(ROOT / "run_full_pipeline.py"),
            "--video",
            str(video),
            "--session-id",
            args.session_id,
            "--max-frames",
            str(args.max_frames),
            "--standard-vision",
            "--api-interval",
            env["API_MIN_INTERVAL_SEC"],
            "--with-pinecone",
        ]
        print("Running pipeline:", " ".join(cmd))
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=False)
        pipeline_rc = proc.returncode
        report["pipeline"] = {
            "exit_code": pipeline_rc,
            "elapsed_sec": round(time.time() - t0, 1),
        }

    # Load session artifacts
    sys.path.insert(0, str(ROOT))
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
    os.environ.update({k: v for k, v in env.items() if k.startswith(("SESSION", "GEMINI", "API_"))})
    from src.session_bootstrap import apply_session_layout
    from src.config import settings
    from src.api_retry import rate_limit_hits, reset_api_run_state
    from src.pinecone_indexer import search_frames

    apply_session_layout(args.session_id)
    session_root = settings.SESSION_DIR

    analyses_path = settings.ANALYSIS_DIR / "all_analysis.json"
    analyses = json.loads(analyses_path.read_text(encoding="utf-8")) if analyses_path.exists() else []

    pipeline_report_path = settings.OUTPUTS_DIR / "pipeline_run_report.json"
    pipeline_report = (
        json.loads(pipeline_report_path.read_text(encoding="utf-8"))
        if pipeline_report_path.exists()
        else {}
    )

    report["vlm"] = _score_vlm(analyses)
    agent_answers = pipeline_report.get("steps", {}).get("agent", {}).get("answers", [])
    report["agent"] = _score_agent(agent_answers)

    # Pinecone smoke (may already be in pipeline report)
    pinecone_search: Dict[str, Any] = {}
    for query, _ in PINECONE_QUERIES:
        try:
            pinecone_search[query] = search_frames(query, top_k=3)
        except Exception as exc:
            pinecone_search[query] = {"error": str(exc)}
    report["pinecone"] = _score_pinecone(pinecone_search)

    health = pipeline_report.get("health", {})
    report["context"] = {
        "pass": bool(health.get("session_context.json") and health.get("summary_count", 0) > 0),
        "health": health,
    }

    report["rate_limits"] = {
        "hits_this_eval_import": rate_limit_hits(),
        "mitigation": "GEMINI_PREFER_FLASH=true, API_MIN_INTERVAL_SEC>=12, fewer Pro retries",
    }

    checks = [
        report["vlm"].get("pass"),
        report["agent"].get("pass"),
        report["pinecone"].get("pass"),
        report["context"].get("pass"),
        pipeline_rc == 0 or args.skip_pipeline,
    ]
    report["production_ready"] = all(checks)
    report["checks_passed"] = sum(1 for c in checks if c)
    report["checks_total"] = len(checks)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_json = EVAL_DIR / "production_eval_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Production Pipeline Evaluation",
        f"- **Session**: `{args.session_id}`",
        f"- **Production ready**: {report['production_ready']} ({report['checks_passed']}/{report['checks_total']} checks)",
        f"- **Rate limit hits (import side)**: {report['rate_limits']['hits_this_eval_import']}",
        "",
        "## VLM understanding",
        f"- Pass: **{report['vlm']['pass']}** ({report['vlm'].get('frames_ok', 0)}/{report['vlm'].get('frames_total', 0)} frames)",
        "",
        "## Agent responses",
        f"- Pass: **{report['agent']['pass']}**",
    ]
    for row in report["agent"].get("scored", []):
        lines.append(f"- Q: {row['question'][:60]}… → hits: {row['keyword_hits']}")
    lines.extend(["", "## Pinecone", f"- Pass: **{report['pinecone']['pass']}**"])
    for row in report["pinecone"].get("queries", []):
        lines.append(f"- `{row['query']}` → score {row['top_score']} ({row['top_frame']})")

    (EVAL_DIR / "production_eval_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"PRODUCTION EVAL: {'PASS' if report['production_ready'] else 'NEEDS WORK'}")
    print(f"Report: {out_json}")
    print("=" * 60)
    return 0 if report["production_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
