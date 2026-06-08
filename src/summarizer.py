"""
summarizer.py — Generates a comprehensive session summary and one-line summary using GPT-4o.

- Collects all analysis and alert JSONs
- Sends to GPT-4o for summary
- Saves outputs/session/session_summary.json
"""

import ast
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.gemini_client import generate_text
from src.config import settings

SUMMARY_PATH = settings.SESSION_DIR / "session_summary.json"

ONE_LINE_PROMPT = """
You are a security operations center AI.
Based on today's drone monitoring session:

Total frames: {frames_analyzed}
Total alerts: {total_alerts}
High severity: {high_alerts}
People detected: {people_detected}
Vehicles detected: {vehicles_detected}
Key incidents: {incidents}

Generate ONE concise sentence (max 30 words) summarizing today's security monitoring session. Be specific with numbers and locations. Professional security tone.
"""

SESSION_PROMPT = """
You are a drone security analyst.
Return ONLY valid JSON matching this exact top-level schema:
{
  "session_summary": {
    "date": "YYYY-MM-DD",
    "total_frames_analyzed": 0,
    "total_alerts": 0,
    "high_alerts": 0,
    "medium_alerts": 0,
    "low_alerts": 0,
    "common_objects_detected": [],
    "people_count": 0,
    "vehicles_count": 0,
    "locations_visited": [],
    "session_highlights": "..."
  },
  "system_operations": {
    "average_processing_time_ms": 0,
    "most_frequent_model_used": "unknown",
    "max_confidence": 0.0,
    "common_recommendation": "..."
  },
  "alert_description": {
    "total_alerts_with_no_trigger": 0,
    "frames_triggers": 0
  },
  "battery_status": {
    "initial_battery_level": "unknown",
    "battery_issues_detected_in_frames": []
  },
  "narrative": "..."
}
"""


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _extract_json_payload(text: str) -> Dict[str, Any] | None:
    """Best-effort JSON extraction from model output."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None

    return None


def _build_fallback_summary(
    all_analysis: List[Dict[str, Any]],
    all_alerts: Dict[str, Any],
    session_context: Dict[str, Any],
) -> Dict[str, Any]:
    alerts = all_alerts.get("alerts", []) if isinstance(all_alerts, dict) else []
    alerts = alerts if isinstance(alerts, list) else []

    objects: List[str] = []
    models: List[str] = []
    recommendations: List[str] = []
    processing_times: List[int] = []
    confidences: List[float] = []
    battery_values: List[int] = []
    battery_issue_frames: List[str] = []

    for row in all_analysis:
        if not isinstance(row, dict):
            continue
        objects.extend(_as_string_list(row.get("objects_detected", [])))
        if row.get("model_used"):
            models.append(str(row.get("model_used")))
        if row.get("recommended_action"):
            recommendations.append(str(row.get("recommended_action")))
        processing_times.append(_safe_int(row.get("processing_time_ms", 0)))
        confidences.append(_safe_float(row.get("confidence", 0.0)))

    for row in all_analysis:
        if not isinstance(row, dict):
            continue
        telemetry = row.get("telemetry") if isinstance(row.get("telemetry"), dict) else {}
        battery = telemetry.get("battery") if isinstance(telemetry, dict) else None
        if isinstance(battery, (int, float)):
            battery_values.append(int(battery))
            if battery < 30:
                battery_issue_frames.append(str(row.get("frame_id", "unknown")))

    unique_objects = sorted(set(objects))
    common_objects = unique_objects[:8]
    average_processing_time = int(sum(processing_times) / len(processing_times)) if processing_times else 0
    max_confidence = max(confidences) if confidences else 0.0

    most_frequent_model = "unknown"
    if models:
        most_frequent_model = max(set(models), key=models.count)

    common_recommendation = "Continue monitoring."
    if recommendations:
        common_recommendation = max(set(recommendations), key=recommendations.count)

    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    triggered_alerts = 0
    no_trigger_alerts = 0
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        if bool(alert.get("alert_triggered", False)):
            triggered_alerts += 1
            severity = str(alert.get("severity", "")).upper()
            if severity in severity_counts:
                severity_counts[severity] += 1
        else:
            no_trigger_alerts += 1

    total_frames = _safe_int(session_context.get("frames_analyzed", len(all_analysis)))
    locations = _as_string_list(session_context.get("locations_visited", []))

    return {
        "session_summary": {
            "date": time.strftime("%Y-%m-%d"),
            "total_frames_analyzed": total_frames,
            "total_alerts": _safe_int(session_context.get("total_alerts", triggered_alerts)),
            "high_alerts": _safe_int(session_context.get("high_alerts", severity_counts["HIGH"])),
            "medium_alerts": _safe_int(session_context.get("medium_alerts", severity_counts["MEDIUM"])),
            "low_alerts": _safe_int(session_context.get("low_alerts", severity_counts["LOW"])),
            "common_objects_detected": common_objects,
            "people_count": _safe_int(session_context.get("people_detected", 0)),
            "vehicles_count": _safe_int(session_context.get("vehicles_detected", 0)),
            "locations_visited": locations,
            "session_highlights": (
                f"Processed {total_frames} frames with {_safe_int(session_context.get('total_alerts', triggered_alerts))} alerts. "
                f"Highest severity count: {_safe_int(session_context.get('high_alerts', severity_counts['HIGH']))}."
            ),
        },
        "system_operations": {
            "average_processing_time_ms": average_processing_time,
            "most_frequent_model_used": most_frequent_model,
            "max_confidence": round(max_confidence, 4),
            "common_recommendation": common_recommendation,
        },
        "alert_description": {
            "total_alerts_with_no_trigger": no_trigger_alerts,
            "frames_triggers": triggered_alerts,
        },
        "battery_status": {
            "initial_battery_level": str(battery_values[0]) if battery_values else "unknown",
            "battery_issues_detected_in_frames": battery_issue_frames,
        },
        "narrative": (
            f"Monitoring session covered {len(locations)} locations with {triggered_alerts} triggered alerts. "
            "System maintained continuous frame-level analysis and indexing throughout the run."
        ),
    }


def _enforce_summary_schema(
    candidate: Dict[str, Any],
    all_analysis: List[Dict[str, Any]],
    all_alerts: Dict[str, Any],
    session_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Enforces the summary contract and fills missing fields from deterministic fallback stats."""
    fallback = _build_fallback_summary(all_analysis, all_alerts, session_context)

    session_summary = candidate.get("session_summary", {}) if isinstance(candidate, dict) else {}
    if not isinstance(session_summary, dict):
        session_summary = {}

    system_operations = candidate.get("system_operations", {}) if isinstance(candidate, dict) else {}
    if not isinstance(system_operations, dict):
        system_operations = {}

    alert_description = candidate.get("alert_description", {}) if isinstance(candidate, dict) else {}
    if not isinstance(alert_description, dict):
        alert_description = {}

    battery_status = candidate.get("battery_status", {}) if isinstance(candidate, dict) else {}
    if not isinstance(battery_status, dict):
        battery_status = {}

    normalized = {
        "session_summary": {
            "date": str(session_summary.get("date") or fallback["session_summary"]["date"]),
            "total_frames_analyzed": _safe_int(
                session_summary.get("total_frames_analyzed"),
                fallback["session_summary"]["total_frames_analyzed"],
            ),
            "total_alerts": _safe_int(session_summary.get("total_alerts"), fallback["session_summary"]["total_alerts"]),
            "high_alerts": _safe_int(session_summary.get("high_alerts"), fallback["session_summary"]["high_alerts"]),
            "medium_alerts": _safe_int(
                session_summary.get("medium_alerts"), fallback["session_summary"]["medium_alerts"]
            ),
            "low_alerts": _safe_int(session_summary.get("low_alerts"), fallback["session_summary"]["low_alerts"]),
            "common_objects_detected": _as_string_list(
                session_summary.get("common_objects_detected", fallback["session_summary"]["common_objects_detected"])
            ),
            "people_count": _safe_int(session_summary.get("people_count"), fallback["session_summary"]["people_count"]),
            "vehicles_count": _safe_int(session_summary.get("vehicles_count"), fallback["session_summary"]["vehicles_count"]),
            "locations_visited": _as_string_list(
                session_summary.get("locations_visited", fallback["session_summary"]["locations_visited"])
            ),
            "session_highlights": str(
                session_summary.get("session_highlights") or fallback["session_summary"]["session_highlights"]
            ),
        },
        "system_operations": {
            "average_processing_time_ms": _safe_int(
                system_operations.get("average_processing_time_ms"),
                fallback["system_operations"]["average_processing_time_ms"],
            ),
            "most_frequent_model_used": str(
                system_operations.get("most_frequent_model_used")
                or fallback["system_operations"]["most_frequent_model_used"]
            ),
            "max_confidence": _safe_float(
                system_operations.get("max_confidence"),
                fallback["system_operations"]["max_confidence"],
            ),
            "common_recommendation": str(
                system_operations.get("common_recommendation")
                or fallback["system_operations"]["common_recommendation"]
            ),
        },
        "alert_description": {
            "total_alerts_with_no_trigger": _safe_int(
                alert_description.get("total_alerts_with_no_trigger"),
                fallback["alert_description"]["total_alerts_with_no_trigger"],
            ),
            "frames_triggers": _safe_int(
                alert_description.get("frames_triggers"),
                fallback["alert_description"]["frames_triggers"],
            ),
        },
        "battery_status": {
            "initial_battery_level": str(
                battery_status.get("initial_battery_level")
                or fallback["battery_status"]["initial_battery_level"]
            ),
            "battery_issues_detected_in_frames": _as_string_list(
                battery_status.get(
                    "battery_issues_detected_in_frames",
                    fallback["battery_status"]["battery_issues_detected_in_frames"],
                )
            ),
        },
        "narrative": str(candidate.get("narrative") or fallback["narrative"]),
    }

    # Backward-compatible convenience block for downstream checks.
    normalized["statistics"] = {
        "total_alerts": normalized["session_summary"]["total_alerts"],
        "frames_analyzed": normalized["session_summary"]["total_frames_analyzed"],
        "high_alerts": normalized["session_summary"]["high_alerts"],
        "medium_alerts": normalized["session_summary"]["medium_alerts"],
        "low_alerts": normalized["session_summary"]["low_alerts"],
    }

    return normalized


def generate_one_line_summary(session_context: dict) -> str:
    prompt = ONE_LINE_PROMPT.format(**session_context)
    try:
        return generate_text(prompt, max_output_tokens=60).strip()
    except Exception:
        return (
            f"Session processed {session_context.get('frames_analyzed', 0)} frames with "
            f"{session_context.get('total_alerts', 0)} alerts across "
            f"{len(session_context.get('locations_visited', []))} locations."
        )


def _request_session_summary(prompt: str) -> str:
    return generate_text(
        prompt + "\n\nRespond with valid JSON only.",
        max_output_tokens=1200,
        temperature=0.2,
    ).strip()


def _generate_summary_with_retries(
    base_prompt: str,
    all_analysis: List[Dict[str, Any]],
    all_alerts: Dict[str, Any],
    session_context: Dict[str, Any],
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """Retries summary generation and parse repair before deterministic fallback."""
    fallback = _build_fallback_summary(all_analysis, all_alerts, session_context)
    previous_response = ""

    for attempt in range(1, max_attempts + 1):
        attempt_prompt = base_prompt
        if previous_response:
            attempt_prompt += (
                "\n\nPrevious response failed schema/parse validation. "
                "Repair it to valid JSON with the required schema only:\n"
                f"{previous_response}"
            )

        try:
            raw = _request_session_summary(attempt_prompt)
            parsed = _extract_json_payload(raw)
            if parsed is None:
                previous_response = raw
                continue
            return _enforce_summary_schema(parsed, all_analysis, all_alerts, session_context)
        except Exception as exc:
            previous_response = f"Attempt {attempt} failed: {exc}"

    return _enforce_summary_schema(fallback, all_analysis, all_alerts, session_context)

def generate_session_summary():
    print("\nGenerating session summary...")
    with open(settings.ANALYSIS_DIR / "all_analysis.json", "r", encoding="utf-8") as f:
        all_analysis = json.load(f)
    with open(settings.ALERTS_DIR / "all_alerts.json", "r", encoding="utf-8") as f:
        all_alerts = json.load(f)
    with open(settings.SESSION_DIR / "session_context.json", "r", encoding="utf-8") as f:
        session_context = json.load(f)
    prompt = SESSION_PROMPT + json.dumps({
        "all_analysis": all_analysis,
        "all_alerts": all_alerts,
        "session_context": session_context
    }, indent=2)
    summary = _generate_summary_with_retries(
        base_prompt=prompt,
        all_analysis=all_analysis,
        all_alerts=all_alerts,
        session_context=session_context,
        max_attempts=3,
    )

    summary["one_line_summary"] = generate_one_line_summary(session_context)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Session summary saved to {SUMMARY_PATH}")
    return summary

if __name__ == "__main__":
    generate_session_summary()
