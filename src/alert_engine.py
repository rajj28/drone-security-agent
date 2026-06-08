"""
alert_engine.py — Two-layer alert system for drone security frames.

- Layer 1: Rule-based alerting (fast, deterministic)
- Layer 2: Gemini validation for borderline cases
- Saves per-frame and combined alert JSONs
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from langchain.memory import ConversationSummaryBufferMemory
from src.config import settings
from src.gemini_client import generate_text
from src.gemini_langchain import GeminiLangChain
from src.behavioral_analyzer import analyze_behavioral_threats

RULES = [
    (lambda a, t: t["is_after_hours"] and "person" in a.get("objects_detected", []), "HIGH", "after_hours_person"),
    (lambda a, t: t["is_restricted_zone"] and "person" in a.get("objects_detected", []), "HIGH", "restricted_zone_person"),
    (lambda a, t: a.get("people_count", 0) > 3, "MEDIUM", "crowd_detected"),
    (lambda a, t: t["is_after_hours"] and any(v not in ["sedan","SUV"] for v in a.get("vehicles_detected", [])), "HIGH", "unknown_vehicle_after_hours"),
    (lambda a, t: "loitering" in a.get("activity", "").lower(), "MEDIUM", "loitering_detected"),
    (lambda a, t: a.get("threat_assessment") == "high", "HIGH", "threat_assessment_high"),
    # Enhanced theft detection rules
    (lambda a, t: a.get("people_count", 0) >= 2 and "shop" in a.get("vlm_description", "").lower(), "MEDIUM", "potential_theft_scenario"),
    (lambda a, t: any(action in str(a.get("person_features", [])).lower() for action in ["pocket", "conceal", "hide", "reach"]), "HIGH", "suspicious_hand_actions"),
    (lambda a, t: any(word in a.get("vlm_description", "").lower() for word in ["phone", "mobile", "electronics"]) and a.get("people_count", 0) >= 2, "MEDIUM", "electronics_theft_risk"),
    (lambda a, t: any(word in a.get("activity", "").lower() for word in ["distract", "avoid", "nervous", "suspicious"]), "HIGH", "suspicious_behavior"),
    (lambda a, t: a.get("people_count", 0) >= 4 and "interior" in a.get("scene_type", "").lower(), "HIGH", "multiple_persons_interior"),
    # Behavioral analysis rules
    (lambda a, t: _behavioral_threat_check(a, t), "HIGH", "behavioral_threat_detected"),
    (lambda a, t: _shoplifting_pattern_check(a, t), "HIGH", "shoplifting_pattern_detected"),
]

BORDERLINE = ["MEDIUM", "threat_assessment:medium"]

LLM_PROMPT = (
    "You are a drone security alert validator. Given the frame analysis and telemetry, "
    "confirm or override the alert severity and type. Respond in valid JSON: {\n"
    "  'llm_validated': true,\n"
    "  'severity': 'HIGH/MEDIUM/LOW/NONE',\n"
    "  'llm_reasoning': 'explanation',\n"
    "  'recommended_action': 'action'\n"
    "}"
)

def rule_based_alert(analysis: Dict[str, Any], telemetry: Dict[str, Any]) -> Dict[str, Any]:
    for rule, severity, rule_name in RULES:
        if rule(analysis, telemetry):
            return {"alert_triggered": True, "severity": severity, "rule_triggered": rule_name}
    return {"alert_triggered": False, "severity": "NONE", "rule_triggered": None}


def _behavioral_threat_check(analysis: Dict[str, Any], telemetry: Dict[str, Any]) -> bool:
    """Check for behavioral threats using the behavioral analyzer."""
    try:
        behavioral_result = analyze_behavioral_threats(analysis)
        overall_threat = behavioral_result.get('overall_threat_level', 'low')
        return overall_threat in ['high', 'medium']
    except Exception:
        return False

def _shoplifting_pattern_check(analysis: Dict[str, Any], telemetry: Dict[str, Any]) -> bool:
    """Check specifically for shoplifting patterns."""
    try:
        behavioral_result = analyze_behavioral_threats(analysis)
        shoplifting = behavioral_result.get('shoplifting_detection', {})
        return shoplifting.get('shoplifting_risk', False) or shoplifting.get('risk_level') == 'high'
    except Exception:
        return False

def _load_json_file(path: Path, default: Any) -> Any:
    """Loads JSON from disk with a fallback default."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return default


def _write_json_file(path: Path, payload: Any) -> None:
    """Writes human-readable JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _extract_frame_number(frame_id: str) -> int:
    """Extracts the numeric portion from frame_### identifiers."""
    try:
        return int(frame_id.split("_")[-1])
    except Exception:
        return 1


def _default_session_context() -> Dict[str, Any]:
    """Returns a blank rolling session context for alerts."""
    return {
        "frames_analyzed": 0,
        "total_alerts": 0,
        "high_alerts": 0,
        "medium_alerts": 0,
        "low_alerts": 0,
        "people_detected": 0,
        "vehicles_detected": 0,
        "locations_visited": [],
        "incidents": [],
        "alert_processed_frames": [],
        "running_narrative": "Alert session started.",
    }


def _load_context_summaries() -> Dict[str, Any]:
    """Loads or initializes the per-frame context summary store."""
    path = settings.SESSION_DIR / "context_summaries.json"
    default_store = {
        "session_id": f"SESSION_{time.strftime('%Y%m%d_%H%M%S')}",
        "summaries": [],
    }
    if path.exists():
        existing = _load_json_file(path, default_store)
        if isinstance(existing, dict):
            existing.setdefault("session_id", default_store["session_id"])
            existing.setdefault("summaries", [])
            return existing
    return default_store


def _load_memory_log() -> Dict[str, Any]:
    """Loads the explicit persisted memory log for the alert agent."""
    path = settings.SESSION_DIR / "alert_memory_log.json"
    return _load_json_file(path, {"session_id": f"SESSION_{time.strftime('%Y%m%d_%H%M%S')}", "events": []})


def _append_memory_log(role: str, input_text: str, output_text: str, metadata: Dict[str, Any] | None = None) -> None:
    """Appends a compact persisted memory event for later reasoning."""
    path = settings.SESSION_DIR / "alert_memory_log.json"
    store = _load_memory_log()
    store.setdefault("events", []).append(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "role": role,
            "input": input_text,
            "output": output_text,
            "metadata": metadata or {},
        }
    )
    _write_json_file(path, store)


def _load_recent_memory_events(limit: int = 5) -> List[Dict[str, Any]]:
    """Returns the most recent persisted memory events."""
    store = _load_memory_log()
    events = store.get("events", []) if isinstance(store, dict) else []
    if not isinstance(events, list):
        return []
    return events[-limit:]


def _load_recent_context_summaries(limit: int = 5) -> List[Dict[str, Any]]:
    """Loads the most recent context summaries for alert reasoning."""
    store = _load_json_file(settings.SESSION_DIR / "context_summaries.json", {"summaries": []})
    summaries = store.get("summaries", []) if isinstance(store, dict) else []
    if not isinstance(summaries, list):
        return []
    return summaries[-limit:]


def _build_context_summary(
    frame_id: str,
    analysis: Dict[str, Any],
    telemetry: Dict[str, Any],
    session_context: Dict[str, Any],
    alert_json: Dict[str, Any],
) -> str:
    """Builds a concise cumulative summary for a processed alert frame."""
    return (
        f"Frame {frame_id} reviewed at {telemetry.get('location', 'unknown location')} ({telemetry.get('timestamp', 'unknown time')}). "
        f"Cumulative status: {session_context.get('frames_analyzed', 0)} frames, "
        f"{session_context.get('total_alerts', 0)} alerts, {session_context.get('people_detected', 0)} people, "
        f"{session_context.get('vehicles_detected', 0)} vehicles. "
        f"Current alert: {alert_json.get('severity', 'NONE')} {alert_json.get('alert_type', 'no_alert')}.")


def _persist_context_summary(
    frame_id: str,
    analysis: Dict[str, Any],
    telemetry: Dict[str, Any],
    session_context: Dict[str, Any],
    alert_json: Dict[str, Any],
) -> None:
    """Appends a new context summary entry and saves the store to disk."""
    path = settings.SESSION_DIR / "context_summaries.json"
    context_store = _load_context_summaries()
    summary_entry = {
        "after_frame": frame_id,
        "timestamp": telemetry.get("timestamp"),
        "frames_analyzed": int(session_context.get("frames_analyzed", 0)),
        "frames_remaining": max(0, settings.MAX_FRAMES - int(session_context.get("frames_analyzed", 0))),
        "context_summary": _build_context_summary(frame_id, analysis, telemetry, session_context, alert_json),
        "running_stats": {
            "total_alerts": session_context.get("total_alerts", 0),
            "people_detected": session_context.get("people_detected", 0),
            "vehicles_detected": session_context.get("vehicles_detected", 0),
            "high_severity_events": session_context.get("high_alerts", 0),
        },
        "agent_memory_snapshot": session_context.get("running_narrative", ""),
    }
    context_store.setdefault("summaries", []).append(summary_entry)
    _write_json_file(path, context_store)

def llm_validate_alert(alert: Dict[str, Any], analysis: Dict[str, Any], telemetry: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = f"Frame analysis: {json.dumps(analysis)}\nTelemetry: {json.dumps(telemetry)}\nRule-based alert: {json.dumps(alert)}"
    start = time.time()
    try:
        content = generate_text(
            user_prompt,
            system_instruction=LLM_PROMPT,
            max_output_tokens=256,
        )
        elapsed = int((time.time() - start) * 1000)
        try:
            llm_result = json.loads(content.replace("'", '"'))
        except Exception:
            llm_result = {"llm_validated": True, "severity": alert["severity"], "llm_reasoning": content, "recommended_action": "Review manually"}
        llm_result["processing_time_ms"] = elapsed
        return llm_result
    except Exception as e:
        return {"llm_validated": False, "llm_reasoning": str(e), "severity": alert["severity"], "recommended_action": "Manual review required"}

class AlertEngineAgent:
    """Stateful alert engine agent with memory and session context."""

    def __init__(self):
        self.llm = GeminiLangChain(model=settings.GEMINI_MODEL)
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=2000,
            return_messages=True,
            memory_key="chat_history",
        )
        self.session_context = self._load_session_context()
        self.alert_runs: List[Dict[str, Any]] = []

    def _load_session_context(self) -> Dict[str, Any]:
        """Loads the rolling alert session context from disk."""
        path = settings.SESSION_DIR / "session_context.json"
        session_context = _load_json_file(path, _default_session_context())
        if isinstance(session_context, dict):
            session_context.setdefault("alert_processed_frames", [])
        return session_context

    def _save_session_context(self) -> None:
        """Persists the rolling session context to disk."""
        path = settings.SESSION_DIR / "session_context.json"
        _write_json_file(path, self.session_context)

    def _record_memory(self, user_text: str, assistant_text: str) -> None:
        """Stores a compact interaction in ConversationSummaryBufferMemory."""
        try:
            self.memory.save_context({"input": user_text}, {"output": assistant_text})
        except Exception:
            pass
        _append_memory_log("alert_agent", user_text, assistant_text, {"source": "conversation_memory"})

    def _serialize_recent_memory(self, limit: int = 4) -> List[str]:
        """Serializes the latest memory messages into compact text snippets."""
        return [
            f"{event.get('role', 'unknown')}: {event.get('input', '')} -> {event.get('output', '')}"
            for event in _load_recent_memory_events(limit)
        ]

    def _reason_about_alert(
        self,
        frame_id: str,
        analysis: Dict[str, Any],
        telemetry: Dict[str, Any],
        alert: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Uses the LLM plus memory and recent context to reason about the alert."""
        recent_context = _load_recent_context_summaries(limit=4)
        recent_memory = self._serialize_recent_memory(limit=4)
        prompt = f"""
You are the security alert reasoning layer for a docked drone monitoring system.
Decide whether the rule-based alert should be confirmed, escalated, reduced, or left unchanged.
Use only the supplied data and keep the answer in JSON.

Current frame:
{json.dumps({
    'frame_id': frame_id,
    'analysis': analysis,
    'telemetry': telemetry,
    'rule_alert': alert,
}, indent=2)}

Recent context summaries:
{json.dumps(recent_context, indent=2)}

Recent memory:
{json.dumps(recent_memory, indent=2)}

Return JSON with keys:
{{
  "severity": "HIGH|MEDIUM|LOW|NONE",
  "alert_triggered": true,
  "llm_validated": true,
  "llm_reasoning": "brief explanation",
  "recommended_action": "operator action",
  "confidence": 0.0,
  "context_signal": "what prior context influenced the decision"
}}
"""

        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", str(response))
            try:
                parsed = json.loads(content.replace("```json", "").replace("```", "").strip())
                parsed.setdefault("llm_validated", True)
                return parsed
            except Exception:
                return {
                    "severity": alert.get("severity", "NONE"),
                    "alert_triggered": alert.get("alert_triggered", False),
                    "llm_validated": True,
                    "llm_reasoning": content,
                    "recommended_action": "Review manually",
                    "confidence": 0.5,
                    "context_signal": "LLM response was unstructured; using fallback reasoning.",
                }
        except Exception as exc:
            return {
                "severity": alert.get("severity", "NONE"),
                "alert_triggered": alert.get("alert_triggered", False),
                "llm_validated": False,
                "llm_reasoning": f"Reasoning unavailable: {exc}",
                "recommended_action": "Manual review required",
                "confidence": 0.4,
                "context_signal": "LLM reasoning failed; using fallback.",
            }

    def _update_session_context(self, analysis: Dict[str, Any], telemetry: Dict[str, Any], alert_json: Dict[str, Any]) -> None:
        """Updates cumulative stats and narrative after processing a frame."""
        self.session_context["frames_analyzed"] = int(self.session_context.get("frames_analyzed", 0)) + 1
        self.session_context["people_detected"] = int(self.session_context.get("people_detected", 0)) + int(
            analysis.get("people_count", 0) or 0
        )
        self.session_context["vehicles_detected"] = int(self.session_context.get("vehicles_detected", 0)) + len(
            analysis.get("vehicles_detected", []) or []
        )

        if alert_json.get("alert_triggered"):
            self.session_context["total_alerts"] = int(self.session_context.get("total_alerts", 0)) + 1
            severity = alert_json.get("severity")
            if severity == "HIGH":
                self.session_context["high_alerts"] = int(self.session_context.get("high_alerts", 0)) + 1
            elif severity == "MEDIUM":
                self.session_context["medium_alerts"] = int(self.session_context.get("medium_alerts", 0)) + 1
            elif severity == "LOW":
                self.session_context["low_alerts"] = int(self.session_context.get("low_alerts", 0)) + 1

            incidents = self.session_context.setdefault("incidents", [])
            incidents.append(
                {
                    "frame_id": analysis.get("frame_id"),
                    "timestamp": telemetry.get("timestamp"),
                    "location": telemetry.get("location"),
                    "severity": severity,
                    "alert_type": alert_json.get("alert_type"),
                }
            )

        locations = self.session_context.setdefault("locations_visited", [])
        location = telemetry.get("location")
        if location and location not in locations:
            locations.append(location)

        self.session_context["running_narrative"] = (
            f"Analyzed {self.session_context.get('frames_analyzed', 0)} frames. "
            f"Alerts: {self.session_context.get('total_alerts', 0)} total, "
            f"{self.session_context.get('high_alerts', 0)} high, "
            f"{self.session_context.get('medium_alerts', 0)} medium, "
            f"{self.session_context.get('low_alerts', 0)} low."
        )

        self._save_session_context()

    def _mark_frame_processed(self, frame_id: str) -> bool:
        """Marks a frame as processed and returns False if it was already seen."""
        processed_frames = self.session_context.setdefault("alert_processed_frames", [])
        if frame_id in processed_frames:
            return False
        processed_frames.append(frame_id)
        return True
        _persist_context_summary(analysis.get("frame_id", "unknown_frame"), analysis, telemetry, self.session_context, alert_json)

    def _build_alert_json(
        self,
        frame_id: str,
        analysis: Dict[str, Any],
        telemetry: Dict[str, Any],
        alert: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Builds a standard alert JSON record."""
        return {
            "alert_id": f"ALT_{datetime.utcnow().strftime('%Y%m%d')}_{_extract_frame_number(frame_id):03}",
            "frame_id": frame_id,
            "timestamp": analysis.get("timestamp", telemetry.get("timestamp", "00:00:00")),
            "location": analysis.get("location", telemetry.get("location", "unknown")),
            "alert_triggered": alert.get("alert_triggered", False),
            "severity": alert.get("severity", "NONE"),
            "alert_type": alert.get("rule_triggered"),
            "message": f"Alert for {frame_id} at {analysis.get('location', telemetry.get('location', 'unknown'))} {analysis.get('timestamp', telemetry.get('timestamp', '00:00:00'))}",
            "objects_involved": analysis.get("objects_detected", []),
            "rule_triggered": alert.get("rule_triggered"),
            "llm_validated": alert.get("llm_validated", False),
            "llm_reasoning": alert.get("llm_reasoning", ""),
            "recommended_action": alert.get("recommended_action", "No action required"),
            "acknowledged": False,
            "auto_escalate": alert.get("severity") == "HIGH",
        }

    def process_frame(self, frame_id: str) -> Dict[str, Any]:
        """Processes a single frame and writes the alert output plus memory state."""
        if not self._mark_frame_processed(frame_id):
            existing = _load_json_file(settings.ALERTS_DIR / f"{frame_id}_alert.json", {})
            existing["duplicate"] = True
            return existing

        analysis = _load_json_file(settings.ANALYSIS_DIR / f"{frame_id}_analysis.json", {})
        telemetry = _load_json_file(settings.TELEMETRY_DIR / f"{frame_id}_telemetry.json", {})
        if not analysis or not telemetry:
            raise FileNotFoundError(f"Missing analysis or telemetry for {frame_id}")

        start = time.time()
        alert = rule_based_alert(analysis, telemetry)
        if alert.get("severity") in ["MEDIUM", "HIGH"]:
            llm_result = self._reason_about_alert(frame_id, analysis, telemetry, alert)
            alert.update(llm_result)

        alert_json = self._build_alert_json(frame_id, analysis, telemetry, alert)
        alert_json["structured_reasoning"] = {
            "severity": alert.get("severity", "NONE"),
            "alert_triggered": alert.get("alert_triggered", False),
            "llm_validated": alert.get("llm_validated", False),
            "llm_reasoning": alert.get("llm_reasoning", ""),
            "recommended_action": alert.get("recommended_action", "No action required"),
            "confidence": alert.get("confidence", 0.0),
            "context_signal": alert.get("context_signal", ""),
        }
        _write_json_file(settings.ALERTS_DIR / f"{frame_id}_alert.json", alert_json)
        self._update_session_context(analysis, telemetry, alert_json)

        summary = (
            f"Processed {frame_id} at {telemetry.get('location', 'unknown location')} with severity {alert_json['severity']}. "
            f"Trigger: {alert_json.get('alert_type') or 'none'}. "
            f"Reasoning: {alert_json['structured_reasoning'].get('llm_reasoning') or 'rule-based only'}."
        )
        self._record_memory(f"Process alert for {frame_id}", summary)
        _append_memory_log(
            "alert",
            f"Process alert for {frame_id}",
            summary,
            {
                "frame_id": frame_id,
                "severity": alert_json.get("severity"),
                "alert_triggered": alert_json.get("alert_triggered"),
                "alert_type": alert_json.get("alert_type"),
            },
        )

        self.alert_runs.append(
            {
                "frame_id": frame_id,
                "severity": alert_json["severity"],
                "alert_triggered": alert_json["alert_triggered"],
                "processing_time_ms": int((time.time() - start) * 1000),
            }
        )
        return alert_json

    def process_all(self) -> List[Dict[str, Any]]:
        """Processes all available analysis frames and writes the combined log."""
        print("\nAlertEngineAgent processing all frames...")
        all_analysis = _load_json_file(settings.ANALYSIS_DIR / "all_analysis.json", [])
        results: List[Dict[str, Any]] = []
        for frame in all_analysis:
            # Skip None entries (from vision analyzer skipping missing frames)
            if frame is None:
                continue
            frame_id = frame.get("frame_id")
            if not frame_id:
                continue
            try:
                result = self.process_frame(frame_id)
                results.append(result)
                print(f"{'Alert' if result['alert_triggered'] else 'Warning'} Alert processed for {frame_id} ({result['severity']})")
            except Exception as exc:
                print(f"Failed to process {frame_id}: {exc}")

        combined = {
            "session_date": time.strftime("%Y-%m-%d"),
            "total_alerts": sum(1 for item in results if item.get("alert_triggered")),
            "high_severity": sum(1 for item in results if item.get("severity") == "HIGH"),
            "medium_severity": sum(1 for item in results if item.get("severity") == "MEDIUM"),
            "low_severity": sum(1 for item in results if item.get("severity") == "LOW"),
            "alerts": results,
        }
        _write_json_file(settings.ALERTS_DIR / "all_alerts.json", combined)

        self._record_memory(
            "Process all alerts",
            f"Generated {combined['total_alerts']} alerts across {len(results)} frames.",
        )

        self._write_agent_runs()
        print(f"\nCombined alerts saved to {settings.ALERTS_DIR / 'all_alerts.json'}")
        return results

    def _write_agent_runs(self) -> None:
        """Writes a compact alert run log to disk."""
        payload = {
            "session_id": f"SESSION_{time.strftime('%Y%m%d_%H%M%S')}",
            "total_runs": len(self.alert_runs),
            "runs": self.alert_runs,
        }
        _write_json_file(settings.SESSION_DIR / "alert_agent_runs.json", payload)


def process_alerts():
    """Backwards-compatible procedural entrypoint that delegates to the agent."""
    return AlertEngineAgent().process_all()


def run_demo() -> None:
    """Runs a small demo so the module produces visible output when executed."""
    agent = AlertEngineAgent()
    print("\nRunning AlertEngineAgent demo...")
    print(f"Session context: {json.dumps(agent.session_context, indent=2)}")
    results = agent.process_all()
    print(f"Alert demo complete. Processed {len(results)} frames.")

if __name__ == "__main__":
    run_demo()
