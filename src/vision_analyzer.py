"""
vision_analyzer.py — Analyzes each frame using GPT-4o Vision and generates structured security assessment JSON.

- Loads JPG image, encodes to base64
- Sends to GPT-4o Vision with security prompt and telemetry context
- Parses and saves analysis JSON per frame
"""

import base64
import json
import time
from pathlib import Path
from typing import Dict, Any
from PIL import Image
from openai import OpenAI
from src.config import settings

SESSION_CONTEXT_PATH = settings.SESSION_DIR / "session_context.json"
CONTEXT_SUMMARIES_PATH = settings.SESSION_DIR / "context_summaries.json"

# System and user prompts
SYSTEM_PROMPT = (
    "You are an expert drone security analyst AI. Analyze this CCTV frame and provide a detailed security assessment. "
    "Focus on security-relevant evidence, not generic descriptions. "
    "Always respond in valid JSON format only."
)
USER_PROMPT_TEMPLATE = (
    "Analyze this security camera frame. Telemetry context: {telemetry}. "
    "Return rich, structured security metadata that supports later alert reasoning. "
    "Use the exact JSON format below and include only fields you can support from the image.\n"
    "{{\n"
    "  'vlm_description': 'detailed description of the scene',\n"
    "  'scene_type': 'parking_lot|road|entrance|warehouse|residential|interior|unknown',\n"
    "  'objects_detected': ['list', 'of', 'object labels'],\n"
    "  'object_details': [\n"
    "    {'label': 'person', 'count': 1, 'confidence': 0.9, 'attributes': ['standing', 'near_vehicle']},\n"
    "    {'label': 'vehicle', 'count': 1, 'confidence': 0.85, 'attributes': ['sedan', 'parked']}\n"
    "  ],\n"
    "  'people_count': 0,\n"
    "  'person_features': [\n"
    "    {'id': 'person_1', 'appearance': ['dark_clothes', 'hood'], 'face_visible': false, 'actions': ['loitering']}\n"
    "  ],\n"
    "  'vehicles_detected': [],\n"
    "  'vehicle_details': [\n"
    "    {'type': 'sedan', 'color': 'white', 'position': 'parked', 'direction': 'north', 'occupancy': 'unknown', 'plate_visible': false}\n"
    "  ],\n"
    "  'activity': 'description of activity',\n"
    "  'security_signals': ['after_hours_presence', 'restricted_zone_vehicle'],\n"
    "  'suspicious_elements': ['list of suspicious observations'],\n"
    "  'threat_assessment': 'none/low/medium/high',\n"
    "  'confidence': 0.95,\n"
    "  'recommended_action': 'description',\n"
    "  'alert_reasoning': 'why the threat level was chosen',\n"
    "  'alert_priority_signals': ['person_after_hours', 'vehicle_loitering']\n"
    "}}"
)


def _as_string_list(value: Any) -> list:
    """Normalizes common list-like values into a list of strings."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _normalize_analysis(analysis: Dict[str, Any], telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures the analysis payload has the downstream fields needed for reasoning."""
    normalized = dict(analysis)
    normalized.setdefault("frame_id", telemetry.get("frame_id"))
    normalized.setdefault("scene_type", "unknown")
    normalized.setdefault("object_details", [])
    normalized.setdefault("person_features", [])
    normalized.setdefault("vehicle_details", [])
    normalized.setdefault("security_signals", [])
    normalized.setdefault("alert_reasoning", normalized.get("recommended_action", ""))
    normalized.setdefault("alert_priority_signals", [])

    normalized["objects_detected"] = _as_string_list(normalized.get("objects_detected", []))
    normalized["vehicles_detected"] = _as_string_list(normalized.get("vehicles_detected", []))
    normalized["suspicious_elements"] = _as_string_list(normalized.get("suspicious_elements", []))
    normalized["security_signals"] = _as_string_list(normalized.get("security_signals", []))
    normalized["alert_priority_signals"] = _as_string_list(normalized.get("alert_priority_signals", []))

    try:
        normalized["people_count"] = int(normalized.get("people_count", 0) or 0)
    except Exception:
        normalized["people_count"] = 0

    if not normalized.get("vlm_description"):
        normalized["vlm_description"] = "Scene analysis unavailable."
    if not normalized.get("activity"):
        normalized["activity"] = "Unknown activity"
    if not normalized.get("recommended_action"):
        normalized["recommended_action"] = "Review frame manually"

    return normalized

def image_to_base64(image_path: Path) -> str:
    """Converts an image to base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def extract_json_payload(content: str) -> Dict[str, Any]:
    """Extracts the first JSON object from a model response.

    The model may wrap JSON in markdown fences or add brief prose around it,
    so this helper strips common wrappers and then falls back to balanced-brace
    extraction before parsing.
    """
    cleaned = content.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        candidate = candidate.replace("\r\n", "\n")
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("Model response did not contain valid JSON")


def load_session_context() -> Dict[str, Any]:
    """Loads or initializes the rolling session context state."""
    default_context = {
        "frames_analyzed": 0,
        "total_alerts": 0,
        "high_alerts": 0,
        "medium_alerts": 0,
        "people_detected": 0,
        "vehicles_detected": 0,
        "locations_visited": [],
        "incidents": [],
        "running_narrative": "Session started.",
    }

    if SESSION_CONTEXT_PATH.exists():
        with open(SESSION_CONTEXT_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return {**default_context, **data}

    return default_context


def load_context_summaries() -> Dict[str, Any]:
    """Loads or initializes the per-frame context summaries container."""
    if CONTEXT_SUMMARIES_PATH.exists():
        with open(CONTEXT_SUMMARIES_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data.setdefault("session_id", f"SESSION_{time.strftime('%Y%m%d_%H%M%S')}")
            data.setdefault("summaries", [])
            return data

    return {
        "session_id": f"SESSION_{time.strftime('%Y%m%d_%H%M%S')}",
        "summaries": [],
    }


def derive_alert_from_analysis(analysis: Dict[str, Any], telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Derives a lightweight alert summary from a frame analysis."""
    alert_triggered = False
    severity = "NONE"
    alert_type = None
    reasons = []

    objects_detected = analysis.get("objects_detected", []) or []
    people_count = int(analysis.get("people_count", 0) or 0)
    threat_assessment = str(analysis.get("threat_assessment", "none") or "none").lower()
    activity = str(analysis.get("activity", "") or "").lower()
    security_signals = [str(item).lower() for item in analysis.get("security_signals", []) or []]
    alert_priority_signals = [str(item).lower() for item in analysis.get("alert_priority_signals", []) or []]
    vehicle_details = analysis.get("vehicle_details", []) or []
    person_features = analysis.get("person_features", []) or []

    if telemetry.get("is_after_hours") and "person" in objects_detected:
        alert_triggered = True
        severity = "HIGH"
        alert_type = "after_hours_person"
        reasons.append("person detected after hours")
    elif telemetry.get("is_restricted_zone") and "person" in objects_detected:
        alert_triggered = True
        severity = "HIGH"
        alert_type = "restricted_zone_person"
        reasons.append("person detected in restricted zone")
    elif people_count > 3:
        alert_triggered = True
        severity = "MEDIUM"
        alert_type = "crowd_detected"
        reasons.append("crowd detected")
    elif "loiter" in activity:
        alert_triggered = True
        severity = "MEDIUM"
        alert_type = "loitering_detected"
        reasons.append("loitering activity observed")
    elif threat_assessment == "high":
        alert_triggered = True
        severity = "HIGH"
        alert_type = "threat_assessment_high"
        reasons.append("model threat assessment is high")

    if not alert_triggered:
        if any(signal in {"after_hours_presence", "restricted_zone_presence", "restricted_zone_vehicle"} for signal in security_signals):
            alert_triggered = True
            severity = "MEDIUM"
            alert_type = "security_signal_detected"
            reasons.append("security signal raised by the vision model")
        elif any(signal in {"person_after_hours", "vehicle_loitering", "unknown_vehicle", "masked_person"} for signal in alert_priority_signals):
            alert_triggered = True
            severity = "MEDIUM"
            alert_type = "priority_signal_detected"
            reasons.append("priority security signal detected")

    if not reasons and person_features:
        suspicious_traits = []
        for person in person_features:
            if not isinstance(person, dict):
                continue
            appearance = [str(item).lower() for item in person.get("appearance", []) or []]
            actions = [str(item).lower() for item in person.get("actions", []) or []]
            if any(term in appearance for term in ["hood", "mask", "covered_face", "dark_clothes"]):
                suspicious_traits.append("person appearance is security-relevant")
            if any(term in actions for term in ["loitering", "watching", "following", "running"]):
                suspicious_traits.append("person behavior is suspicious")
        if suspicious_traits and not alert_triggered:
            alert_triggered = True
            severity = "MEDIUM"
            alert_type = "person_behavior_signal"
            reasons.extend(suspicious_traits)

    if not alert_triggered and vehicle_details:
        vehicle_signals = []
        for vehicle in vehicle_details:
            if not isinstance(vehicle, dict):
                continue
            color = str(vehicle.get("color", "")).lower()
            vehicle_type = str(vehicle.get("type", "")).lower()
            position = str(vehicle.get("position", "")).lower()
            plate_visible = bool(vehicle.get("plate_visible", False))
            if color in {"black", "white", "silver", "gray", "grey", "blue", "red"}:
                vehicle_signals.append(f"vehicle color observed: {color}")
            if position in {"parked", "stopped", "idling"} and telemetry.get("is_after_hours"):
                vehicle_signals.append("vehicle stationary after hours")
            if not plate_visible and telemetry.get("is_restricted_zone"):
                vehicle_signals.append("plate not visible in restricted zone")
            if vehicle_type in {"truck", "van", "suv", "sedan"}:
                vehicle_signals.append(f"vehicle type: {vehicle_type}")
        if vehicle_signals:
            reasons.extend(vehicle_signals[:3])
            if telemetry.get("is_after_hours") or telemetry.get("is_restricted_zone"):
                alert_triggered = True
                severity = "MEDIUM"
                alert_type = "vehicle_context_signal"

    return {
        "alert_triggered": alert_triggered,
        "severity": severity,
        "alert_type": alert_type,
        "reasoning": "; ".join(reasons) if reasons else "No alert criteria matched.",
        "reasoning_signals": reasons,
    }


def build_context_summary(
    frame_id: str,
    telemetry: Dict[str, Any],
    analysis: Dict[str, Any],
    session_context: Dict[str, Any],
    alert_summary: Dict[str, Any],
) -> str:
    """Builds a concise cumulative summary for the current frame."""
    frames_analyzed = session_context.get("frames_analyzed", 0)
    total_alerts = session_context.get("total_alerts", 0)
    people_detected = session_context.get("people_detected", 0)
    vehicles_detected = session_context.get("vehicles_detected", 0)
    location = telemetry.get("location", "unknown location")
    timestamp = telemetry.get("timestamp", "unknown time")
    activity = analysis.get("activity", "No activity reported")
    threat = str(analysis.get("threat_assessment", "none") or "none").lower()
    reasoning = str(analysis.get("alert_reasoning", "") or "").strip()
    priority_signals = ", ".join(analysis.get("alert_priority_signals", []) or [])

    summary_bits = [
        f"Frame {frame_id} analyzed at {location} ({timestamp}).",
        f"Cumulative status: {frames_analyzed} frames, {total_alerts} alerts, {people_detected} people, {vehicles_detected} vehicles.",
        f"Current frame activity: {activity}.",
    ]

    if alert_summary.get("alert_triggered"):
        summary_bits.append(
            f"Alert: {alert_summary.get('severity', 'UNKNOWN')} {alert_summary.get('alert_type', 'unknown_alert')} triggered."
        )
        if alert_summary.get("reasoning"):
            summary_bits.append(f"Alert reasoning: {alert_summary.get('reasoning')}.")
    else:
        summary_bits.append("No alert triggered for this frame.")

    if threat != "none":
        summary_bits.append(f"Threat assessment for this frame: {threat}.")

    if reasoning:
        summary_bits.append(f"Vision reasoning: {reasoning}.")
    if priority_signals:
        summary_bits.append(f"Priority signals: {priority_signals}.")

    return " ".join(summary_bits)


def update_session_context(
    session_context: Dict[str, Any],
    telemetry: Dict[str, Any],
    analysis: Dict[str, Any],
    alert_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Updates rolling session metrics from the latest frame."""
    session_context["frames_analyzed"] = int(session_context.get("frames_analyzed", 0)) + 1

    session_context["people_detected"] = int(session_context.get("people_detected", 0)) + int(
        analysis.get("people_count", 0) or 0
    )

    session_context["vehicles_detected"] = int(session_context.get("vehicles_detected", 0)) + len(
        analysis.get("vehicles_detected", []) or []
    )

    location = telemetry.get("location")
    locations = session_context.setdefault("locations_visited", [])
    if location and location not in locations:
        locations.append(location)

    if alert_summary.get("alert_triggered"):
        session_context["total_alerts"] = int(session_context.get("total_alerts", 0)) + 1
        if alert_summary.get("severity") == "HIGH":
            session_context["high_alerts"] = int(session_context.get("high_alerts", 0)) + 1
        elif alert_summary.get("severity") == "MEDIUM":
            session_context["medium_alerts"] = int(session_context.get("medium_alerts", 0)) + 1

        incidents = session_context.setdefault("incidents", [])
        incidents.append(
            {
                "frame_id": telemetry.get("frame_id"),
                "timestamp": telemetry.get("timestamp"),
                "location": location,
                "severity": alert_summary.get("severity"),
                "alert_type": alert_summary.get("alert_type"),
                "reasoning": alert_summary.get("reasoning"),
            }
        )

    frames = int(session_context.get("frames_analyzed", 0))
    people = int(session_context.get("people_detected", 0))
    vehicles = int(session_context.get("vehicles_detected", 0))
    alerts = int(session_context.get("total_alerts", 0))
    locations_count = len(session_context.get("locations_visited", []))

    session_context["running_narrative"] = (
        f"Analyzed {frames} frames so far. {people} people detected, {vehicles} vehicles detected, "
        f"{alerts} alerts generated across {locations_count} locations."
    )

    return session_context


def persist_context_summary(
    frame_id: str,
    telemetry: Dict[str, Any],
    analysis: Dict[str, Any],
    session_context: Dict[str, Any],
    alert_summary: Dict[str, Any],
    context_store: Dict[str, Any],
) -> None:
    """Appends a new per-frame context summary entry and saves both session files."""
    frames_analyzed = int(session_context.get("frames_analyzed", 0))
    summary_entry = {
        "after_frame": frame_id,
        "timestamp": telemetry.get("timestamp"),
        "frames_analyzed": frames_analyzed,
        "frames_remaining": max(0, settings.MAX_FRAMES - frames_analyzed),
        "context_summary": build_context_summary(frame_id, telemetry, analysis, session_context, alert_summary),
        "running_stats": {
            "total_alerts": session_context.get("total_alerts", 0),
            "people_detected": session_context.get("people_detected", 0),
            "vehicles_detected": session_context.get("vehicles_detected", 0),
            "high_severity_events": session_context.get("high_alerts", 0),
        },
        "agent_memory_snapshot": session_context.get("running_narrative", ""),
    }

    context_store.setdefault("summaries", []).append(summary_entry)

    with open(CONTEXT_SUMMARIES_PATH, "w", encoding="utf-8") as handle:
        json.dump(context_store, handle, indent=2)

    with open(SESSION_CONTEXT_PATH, "w", encoding="utf-8") as handle:
        json.dump(session_context, handle, indent=2)

def analyze_frame(
    frame_id: str,
    image_path: Path,
    telemetry: Dict[str, Any],
    output_dir: Path = settings.ANALYSIS_DIR
) -> Dict[str, Any]:
    """
    Analyzes a frame using GPT-4o Vision and saves the result as JSON.
    Returns the analysis dict.
    """
    print(f"\n🧠 Analyzing {frame_id} with GPT-4o Vision...")
    img_b64 = image_to_base64(image_path)
    user_prompt = USER_PROMPT_TEMPLATE.format(telemetry=json.dumps(telemetry, indent=2))
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    start = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=512
        )
        elapsed = int((time.time() - start) * 1000)
        content = response.choices[0].message.content
        # Parse JSON from response.
        try:
            analysis = extract_json_payload(content)
        except Exception:
            print(f"⚠️ Could not parse JSON for {frame_id}, saving raw content.")
            analysis = {"raw_response": content}
        analysis = _normalize_analysis(analysis, telemetry)
        analysis["alert_reasoning"] = analysis.get("alert_reasoning") or analysis.get("recommended_action", "")
        analysis["reasoning_signals"] = analysis.get("reasoning_signals", [])
        # Build final analysis dict
        result = {
            "frame_id": frame_id,
            "timestamp": telemetry["timestamp"],
            "location": telemetry["location"],
            **analysis,
            "model_used": "gpt-4o",
            "processing_time_ms": elapsed
        }
        out_path = output_dir / f"{frame_id}_analysis.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Analysis saved for {frame_id} ({elapsed} ms)")
        return result
    except Exception as e:
        print(f"❌ Vision analysis failed for {frame_id}: {e}")
        return {}

def analyze_all_frames():
    """
    Runs analysis for all frames using telemetry and extracted images.
    """
    meta_path = settings.OUTPUTS_DIR / "extraction_log.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        frame_meta = json.load(f)["frames"]
    telemetry_path = settings.TELEMETRY_DIR / "all_telemetry.json"
    with open(telemetry_path, "r", encoding="utf-8") as f:
        all_telemetry = json.load(f)
    session_context = load_session_context()
    context_store = load_context_summaries()
    all_results = []
    for i, frame in enumerate(frame_meta):
        frame_id = f"frame_{i+1:03}"
        image_path = settings.EXTRACTED_DIR / frame["filename"]
        telemetry = all_telemetry[i]
        result = analyze_frame(frame_id, image_path, telemetry)
        all_results.append(result)

        if result:
            alert_summary = derive_alert_from_analysis(result, telemetry)
            session_context = update_session_context(session_context, telemetry, result, alert_summary)
            persist_context_summary(frame_id, telemetry, result, session_context, alert_summary, context_store)
    # Save combined
    combined_path = settings.ANALYSIS_DIR / "all_analysis.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n📄 Combined analysis saved to {combined_path}")
    return all_results

if __name__ == "__main__":
    analyze_all_frames()
