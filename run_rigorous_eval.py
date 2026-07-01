#!/usr/bin/env python3
"""
Rigorous end-to-end pipeline evaluation on a specific video.
Produces outputs/evaluation/pipeline_eval_report.json and .md
"""
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from src.intelligent_frame_extractor import (
    IntelligentFrameExtractor,
    ExtractionStrategy,
)
from src.config import settings
from src.telemetry_generator import generate_telemetry
import src.api as api_module
from src.alert_engine import rule_based_alert


def test_quality_filter() -> Dict[str, Any]:
    extractor = IntelligentFrameExtractor()
    cases = [
        ("dark_frame", np.zeros((100, 100, 3), dtype=np.uint8)),
        ("bright_frame", np.ones((100, 100, 3), dtype=np.uint8) * 255),
        ("solid_color", np.ones((100, 100, 3), dtype=np.uint8) * 128),
    ]
    results = []
    for name, frame in cases:
        ok, reason = extractor._is_frame_quality_acceptable(frame)
        results.append({"case": name, "accepted": ok, "reason": reason})
    dark_rejected = not results[0]["accepted"]
    bright_rejected = not results[1]["accepted"]
    solid_rejected = not results[2]["accepted"]
    return {
        "cases": results,
        "pass": dark_rejected and bright_rejected and solid_rejected,
    }


def run_pipeline(video_path: Path, max_frames: int) -> Dict[str, Any]:
    session_id = f"eval_{int(time.time())}"
    session_dir = ROOT / "data" / "eval_sessions" / session_id
    extracted_dir = session_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    settings.VIDEO_FILE = video_path
    settings.EXTRACTED_DIR = extracted_dir
    settings.OUTPUTS_DIR = session_dir / "outputs"
    settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.TELEMETRY_DIR = session_dir / "telemetry"
    settings.TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    settings.ANALYSIS_DIR = session_dir / "analysis"
    settings.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    settings.SESSION_DIR = session_dir / "session"
    settings.SESSION_DIR.mkdir(parents=True, exist_ok=True)
    settings.ALERTS_DIR = session_dir / "alerts"
    settings.ALERTS_DIR.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "video": str(video_path),
        "session_id": session_id,
        "session_dir": str(session_dir),
        "max_frames": max_frames,
        "started_at": datetime.utcnow().isoformat() + "Z",
    }

    # Video metadata
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        report["video_meta"] = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_sec": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
        }
        cap.release()

    # Step 1: extraction
    t0 = time.time()
    extractor = IntelligentFrameExtractor()
    frames = extractor.extract_frames_intelligently(
        video_path,
        extracted_dir,
        strategy=ExtractionStrategy.HYBRID,
        max_total_frames=max_frames,
    )
    extractor.save_extraction_log(
        frames,
        video_path,
        ExtractionStrategy.HYBRID,
        output_path=settings.OUTPUTS_DIR / "extraction_log.json",
    )
    report["extraction"] = {
        "elapsed_sec": round(time.time() - t0, 2),
        "frames_extracted": len(frames),
        "frame_files": [f.filename for f in frames],
        "timestamps_sec": [round(f.timestamp, 2) for f in frames],
        "reasons": [getattr(f, "extraction_reason", "") for f in frames],
    }
    log_path = settings.OUTPUTS_DIR / "extraction_log.json"
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
        all_frames = log.get("frames", [])
        rejected = [x for x in all_frames if not x.get("accepted", True)]
        report["extraction"]["total_scanned"] = len(all_frames)
        report["extraction"]["rejected_count"] = len(rejected)
        report["extraction"]["rejection_reasons"] = {}
        for r in rejected:
            reason = r.get("reject_reason", "unknown")
            report["extraction"]["rejection_reasons"][reason] = (
                report["extraction"]["rejection_reasons"].get(reason, 0) + 1
            )

    if not frames:
        report["status"] = "FAILED"
        report["failure"] = "No frames extracted"
        return report

    # Step 2: telemetry
    t0 = time.time()
    meta_path = settings.OUTPUTS_DIR / "extraction_log.json"
    with open(meta_path, encoding="utf-8") as f:
        frame_meta = json.load(f)["frames"]
    generate_telemetry(frame_meta, output_dir=settings.TELEMETRY_DIR)

    # Vision analyzer uses get_latest_extracted_folder(); point it at this session
    api_module.get_latest_extracted_folder = lambda: settings.EXTRACTED_DIR
    from src.vision_analyzer import analyze_all_frames
    tel_path = settings.TELEMETRY_DIR / "all_telemetry.json"
    telemetry: List[Dict] = []
    if tel_path.exists():
        with open(tel_path, encoding="utf-8") as f:
            telemetry = json.load(f)
    report["telemetry"] = {
        "elapsed_sec": round(time.time() - t0, 2),
        "count": len(telemetry),
        "sample": telemetry[0] if telemetry else None,
    }

    # Step 3: vision
    t0 = time.time()
    try:
        analyses = analyze_all_frames()
    except Exception as e:
        report["vision"] = {"error": str(e), "elapsed_sec": round(time.time() - t0, 2)}
        report["status"] = "FAILED"
        return report

    report["vision"] = {
        "elapsed_sec": round(time.time() - t0, 2),
        "frames_analyzed": len(analyses),
    }

    # Step 4: per-frame metrics + alerts
    threat_counts: Dict[str, int] = {}
    people_counts: List[int] = []
    scene_types: List[str] = []
    frame_details: List[Dict[str, Any]] = []
    alerts_triggered = 0
    critical_high = 0

    for a in analyses:
        if not a:
            continue
        tl = str(a.get("threat_level", "UNKNOWN")).upper()
        threat_counts[tl] = threat_counts.get(tl, 0) + 1
        pc = a.get("people_count")
        if isinstance(pc, int):
            people_counts.append(pc)
        st = a.get("scene_type")
        if st:
            scene_types.append(str(st))
        alert = rule_based_alert(a, a.get("telemetry", {}))
        has_alert = bool(alert and alert.get("alert_triggered"))
        if has_alert:
            alerts_triggered += 1
        if tl in ("CRITICAL", "HIGH"):
            critical_high += 1
        frame_details.append({
            "frame_id": a.get("frame_id"),
            "threat_level": tl,
            "people_count": pc,
            "scene_type": st,
            "model_used": a.get("model_used"),
            "description_snippet": (a.get("description") or a.get("vlm_description") or "")[:200],
            "alert_triggered": has_alert,
            "alert_severity": alert.get("severity") if alert else None,
        })

    report["analysis_summary"] = {
        "threat_level_distribution": threat_counts,
        "people_count_min": min(people_counts) if people_counts else None,
        "people_count_max": max(people_counts) if people_counts else None,
        "people_count_avg": round(sum(people_counts) / len(people_counts), 2) if people_counts else None,
        "unique_scene_types": sorted(set(scene_types)),
        "alerts_triggered": alerts_triggered,
        "critical_or_high_frames": critical_high,
        "frame_details": frame_details,
    }

    # Theft-video expectations (heuristic scoring)
    has_retail = any("retail" in s.lower() or "shop" in s.lower() for s in scene_types)
    has_elevated_threat = critical_high > 0
    good_people_count = (max(people_counts) if people_counts else 0) >= 2
    report["scoring"] = {
        "quality_filter_pass": None,  # filled by caller
        "retail_scene_detected": has_retail,
        "elevated_threat_detected": has_elevated_threat,
        "multi_person_detected": good_people_count,
        "frames_with_alerts": alerts_triggered > 0,
    }
    score = sum(
        1
        for k in (
            "elevated_threat_detected",
            "multi_person_detected",
            "frames_with_alerts",
        )
        if report["scoring"].get(k)
    )
    if has_retail:
        score += 1
    report["scoring"]["heuristic_score_0_to_4"] = score
    report["scoring"]["overall_pass"] = score >= 3 and has_elevated_threat

    report["status"] = "COMPLETED"
    report["finished_at"] = datetime.utcnow().isoformat() + "Z"
    return report


def main():
    video_name = "Sneaky Thieves Caught Stealing Phones On Camera - Newsflare (1080p, h264) (1).mp4"
    video_path = ROOT / video_name
    if not video_path.exists():
        for p in ROOT.glob("*.mp4"):
            if "Sneaky" in p.name or "Thieves" in p.name:
                video_path = p
                break
    if not video_path.exists():
        print(f"Video not found: {video_name}")
        return 1

    max_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    print(f"Rigorous eval: {video_path.name} (max_frames={max_frames})")

    qf = test_quality_filter()
    print(f"Quality filter: {'PASS' if qf['pass'] else 'FAIL'}")

    report = run_pipeline(video_path, max_frames)
    report["quality_filter"] = qf
    if report.get("scoring"):
        report["scoring"]["quality_filter_pass"] = qf["pass"]

    out_dir = ROOT / "outputs" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "pipeline_eval_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown summary
    md_lines = [
        "# Pipeline Evaluation Report",
        f"- **Video**: {report.get('video')}",
        f"- **Session**: {report.get('session_id')}",
        f"- **Status**: {report.get('status')}",
        "",
        "## Quality Filter",
        f"- **PASS**: {qf['pass']}",
        "",
        "## Extraction",
    ]
    ex = report.get("extraction", {})
    md_lines.append(f"- Frames extracted: {ex.get('frames_extracted')}")
    md_lines.append(f"- Rejected: {ex.get('rejected_count', 'N/A')}")
    md_lines.append(f"- Rejection breakdown: {ex.get('rejection_reasons', {})}")
    md_lines.append("")
    md_lines.append("## Vision / Threats")
    summ = report.get("analysis_summary", {})
    md_lines.append(f"- Threat distribution: {summ.get('threat_level_distribution')}")
    md_lines.append(f"- People count range: {summ.get('people_count_min')}–{summ.get('people_count_max')} (avg {summ.get('people_count_avg')})")
    md_lines.append(f"- CRITICAL/HIGH frames: {summ.get('critical_or_high_frames')}")
    md_lines.append(f"- Alerts triggered: {summ.get('alerts_triggered')}")
    md_lines.append("")
    md_lines.append("## Scoring")
    sc = report.get("scoring", {})
    for k, v in sc.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.append("")
    md_lines.append("## Per-Frame Details")
    for fd in summ.get("frame_details", []):
        md_lines.append(
            f"- **{fd.get('frame_id')}**: {fd.get('threat_level')}, "
            f"people={fd.get('people_count')}, alert={fd.get('alert_triggered')}"
        )

    md_path = out_dir / "pipeline_eval_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")
    print(f"Status: {report.get('status')}")
    if sc := report.get("scoring"):
        print(f"Heuristic score: {sc.get('heuristic_score_0_to_4')}/4, pass={sc.get('overall_pass')}")
    return 0 if report.get("status") == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
