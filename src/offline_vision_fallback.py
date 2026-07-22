"""
offline_vision_fallback.py — No-API vision placeholder when quota/rate limits block cloud calls.

Uses OpenCV heuristics only so the pipeline (context, alerts, agent structure) can still run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2

from src.frame_preprocessor import assess_image_quality


def analyze_frame_offline(
    frame_id: str,
    image_path: Path,
    telemetry: Dict[str, Any],
    reason: str = "API quota/rate limit",
) -> Dict[str, Any]:
    """Degraded-mode placeholder when the real VLM call couldn't run at all.

    This is an OpenCV edge-density heuristic, NOT a security assessment — it has
    no idea what's in the frame beyond how visually cluttered it looks, so it
    cannot say a scene is actually CLEAR. To avoid manufacturing false
    reassurance during a Gemini outage, threat_level is always forced to at
    least MEDIUM ("needs human review") rather than reporting a confident
    verdict this heuristic has no basis for. `security_signals` always includes
    "analysis_degraded" so every downstream consumer (rules, alert LLM,
    dashboard) can tell this frame was never actually looked at by a real VLM.
    """
    people_count = 0
    activity = "scene visible (unanalyzed)"
    edge_severity = "MEDIUM"
    description = f"Offline analysis for {frame_id} at {telemetry.get('location', 'unknown')}."
    image_quality: Dict[str, Any] = {}

    try:
        image_quality = assess_image_quality(image_path)
        img = cv2.imread(str(image_path))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            # Edge density as a rough proxy for clutter/motion — NOT object or
            # threat recognition. Only used to pick how urgently this frame
            # should be queued for human review, never to declare it safe.
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = float((edges > 0).sum()) / max(edges.size, 1)
            if edge_ratio > 0.08:
                people_count = 3
                activity = "busy scene with multiple subjects or motion (unanalyzed)"
                edge_severity = "HIGH"
            elif edge_ratio > 0.04:
                people_count = 2
                activity = "moderate activity in frame (unanalyzed)"
                edge_severity = "MEDIUM"
            else:
                people_count = 1
                activity = "low visual activity (unanalyzed)"
                edge_severity = "MEDIUM"  # floor: never LOW/CLEAR for a frame nobody actually looked at
            description = (
                f"DEGRADED ANALYSIS — real VLM call failed ({reason}). This is only an "
                f"OpenCV edge-density heuristic (edge_density={edge_ratio:.3f}, "
                f"resolution={w}x{h}), not a real scene understanding. A human operator "
                f"must manually review this frame; no automated threat detection was performed."
            )
    except Exception as exc:
        description = f"DEGRADED ANALYSIS — real VLM call failed ({reason}); offline heuristic also errored (cv2: {exc})."

    threat = edge_severity
    if telemetry.get("is_after_hours"):
        threat = "HIGH"
    if telemetry.get("is_restricted_zone"):
        threat = "HIGH"

    return {
        "frame_id": frame_id,
        "timestamp": telemetry.get("timestamp"),
        "location": telemetry.get("location"),
        "vlm_description": description,
        "scene_type": "unknown",
        "people_count": people_count,
        "objects_detected": ["person"] if people_count else [],
        "activity": activity,
        "threat_level": threat,
        "threat_assessment": threat.lower(),
        "threat_type": "needs_review",
        "security_signals": ["analysis_degraded", "offline_heuristic"],
        "suspicious_elements": [f"Automated analysis unavailable ({reason}) — flagged for manual review."],
        "alert_reasoning": (
            f"Real VLM analysis failed ({reason}); this is an OpenCV heuristic placeholder, "
            f"not a genuine threat assessment. Manual review required."
        ),
        "recommended_action": "Manually review this frame — automated vision analysis did not run.",
        "model_used": "offline-opencv-heuristic",
        "processing_time_ms": 0,
        "confidence": 0.0,
        "image_quality": image_quality or None,
    }
