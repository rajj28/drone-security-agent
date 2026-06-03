"""
offline_vision_fallback.py — No-API vision placeholder when quota/rate limits block cloud calls.

Uses OpenCV heuristics only so the pipeline (context, alerts, agent structure) can still run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2


def analyze_frame_offline(
    frame_id: str,
    image_path: Path,
    telemetry: Dict[str, Any],
) -> Dict[str, Any]:
    """Rule-of-thumb analysis without OpenAI/HF (for dev when quota is zero)."""
    people_count = 0
    activity = "scene visible"
    threat = "MEDIUM"
    description = f"Offline analysis for {frame_id} at {telemetry.get('location', 'unknown')}."

    try:
        img = cv2.imread(str(image_path))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            # Edge density as proxy for activity / crowd clutter
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = float((edges > 0).sum()) / max(edges.size, 1)
            if edge_ratio > 0.08:
                people_count = 3
                activity = "busy scene with multiple subjects or motion"
                threat = "HIGH"
            elif edge_ratio > 0.04:
                people_count = 2
                activity = "moderate activity in frame"
                threat = "MEDIUM"
            else:
                people_count = 1
                activity = "low activity"
                threat = "LOW"
            description = (
                f"Offline heuristic: edge_density={edge_ratio:.3f}, "
                f"resolution={w}x{h}. Retail surveillance frame."
            )
    except Exception as exc:
        description = f"Offline fallback (cv2 error: {exc})"

    if telemetry.get("is_after_hours"):
        threat = "HIGH"
    if telemetry.get("is_restricted_zone"):
        threat = "HIGH"

    return {
        "frame_id": frame_id,
        "timestamp": telemetry.get("timestamp"),
        "location": telemetry.get("location"),
        "vlm_description": description,
        "scene_type": "retail",
        "people_count": people_count,
        "objects_detected": ["person"] if people_count else [],
        "activity": activity,
        "threat_level": threat,
        "threat_assessment": threat.lower(),
        "threat_type": "suspicious_behavior" if threat in ("HIGH", "CRITICAL") else "clear",
        "security_signals": ["offline_heuristic"],
        "alert_reasoning": "Generated without cloud APIs (quota/rate limit). Re-run with billing for real VLM.",
        "model_used": "offline-opencv-heuristic",
        "processing_time_ms": 0,
    }
