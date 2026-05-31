"""
agent.py — LangChain agent with tools, memory, and session context for drone security analysis.

- Provides tools: analyze_frame, search_frames, get_telemetry, trigger_alert, get_session_context, query_event_history
- Uses ConversationSummaryBufferMemory
- Maintains session context and logs agent runs
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from src.config import settings
from src.vision_analyzer import analyze_frame
from src.pinecone_indexer import search_frames
from src.alert_engine import rule_based_alert

SESSION_CONTEXT_PATH = settings.SESSION_DIR / "session_context.json"
CONTEXT_SUMMARIES_PATH = settings.SESSION_DIR / "context_summaries.json"
AGENT_RUNS_PATH = settings.SESSION_DIR / "agent_runs.json"
AGENT_MEMORY_LOG_PATH = settings.SESSION_DIR / "agent_memory_log.json"
ANALYSIS_DIR = settings.ANALYSIS_DIR
TELEMETRY_DIR = settings.TELEMETRY_DIR
ALERTS_DIR = settings.ALERTS_DIR


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
    """Extracts the numeric portion from a frame id like frame_001."""
    try:
        return int(frame_id.split("_")[-1])
    except Exception:
        return 1


def _frame_timestamp_from_telemetry(telemetry: Dict[str, Any]) -> str:
    """Returns telemetry timestamp or a safe fallback string."""
    return str(telemetry.get("timestamp", "00:00:00"))


def _match_time_range(timestamp: str, time_range: str) -> bool:
    """Checks whether a HH:MM:SS timestamp falls inside a HH:MM-HH:MM range."""
    if not time_range or time_range.lower() == "all":
        return True

    if "-" not in time_range:
        return True

    try:
        start_text, end_text = [part.strip() for part in time_range.split("-", 1)]
        timestamp_dt = datetime.strptime(timestamp, "%H:%M:%S")
        start_dt = datetime.strptime(start_text, "%H:%M")
        end_dt = datetime.strptime(end_text, "%H:%M")
        return start_dt.time() <= timestamp_dt.time() <= end_dt.time()
    except Exception:
        return True


def _load_context_summaries() -> Dict[str, Any]:
    """Loads or initializes the rolling context summary store."""
    default_store = {
        "session_id": f"SESSION_{time.strftime('%Y%m%d_%H%M%S')}",
        "summaries": [],
    }
    if CONTEXT_SUMMARIES_PATH.exists():
        existing = _load_json_file(CONTEXT_SUMMARIES_PATH, default_store)
        if isinstance(existing, dict):
            existing.setdefault("session_id", default_store["session_id"])
            existing.setdefault("summaries", [])
            return existing
    return default_store


def _default_session_context() -> Dict[str, Any]:
    """Returns a blank rolling session context for the main agent."""
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
        "agent_processed_frames": [],
        "running_narrative": "Session started.",
    }


def _build_context_summary(
    frame_id: str,
    telemetry: Dict[str, Any],
    analysis: Dict[str, Any],
    session_context: Dict[str, Any],
    alert_summary: Dict[str, Any],
) -> str:
    """Builds a concise cumulative summary for a processed frame."""
    frames = int(session_context.get("frames_analyzed", 0))
    alerts = int(session_context.get("total_alerts", 0))
    people = int(session_context.get("people_detected", 0))
    vehicles = int(session_context.get("vehicles_detected", 0))
    location = telemetry.get("location", "unknown location")
    timestamp = telemetry.get("timestamp", "unknown time")
    activity = analysis.get("activity", "No activity reported")

    summary_parts = [
        f"Frame {frame_id} analyzed at {location} ({timestamp}).",
        f"Cumulative status: {frames} frames, {alerts} alerts, {people} people, {vehicles} vehicles.",
        f"Current frame activity: {activity}.",
    ]

    if alert_summary.get("alert_triggered"):
        summary_parts.append(
            f"Alert: {alert_summary.get('severity', 'UNKNOWN')} {alert_summary.get('alert_type', 'unknown_alert')} triggered."
        )
    else:
        summary_parts.append("No alert triggered for this frame.")

    return " ".join(summary_parts)


def _persist_context_summary(
    frame_id: str,
    telemetry: Dict[str, Any],
    analysis: Dict[str, Any],
    session_context: Dict[str, Any],
    alert_summary: Dict[str, Any],
) -> None:
    """Appends a context summary entry and persists session files."""
    context_store = _load_context_summaries()
    summary_entry = {
        "after_frame": frame_id,
        "timestamp": telemetry.get("timestamp"),
        "frames_analyzed": int(session_context.get("frames_analyzed", 0)),
        "frames_remaining": max(0, settings.MAX_FRAMES - int(session_context.get("frames_analyzed", 0))),
        "context_summary": _build_context_summary(frame_id, telemetry, analysis, session_context, alert_summary),
        "running_stats": {
            "total_alerts": session_context.get("total_alerts", 0),
            "people_detected": session_context.get("people_detected", 0),
            "vehicles_detected": session_context.get("vehicles_detected", 0),
            "high_severity_events": session_context.get("high_alerts", 0),
        },
        "agent_memory_snapshot": session_context.get("running_narrative", ""),
    }
    context_store.setdefault("summaries", []).append(summary_entry)
    _write_json_file(CONTEXT_SUMMARIES_PATH, context_store)


def _load_recent_context_summaries(limit: int = 5) -> List[Dict[str, Any]]:
    """Loads the most recent context summaries for reasoning prompts."""
    store = _load_json_file(CONTEXT_SUMMARIES_PATH, {"summaries": []})
    summaries = store.get("summaries", []) if isinstance(store, dict) else []
    if not isinstance(summaries, list):
        return []
    return summaries[-limit:]


def _serialize_recent_memory(memory: ConversationSummaryBufferMemory, limit: int = 5) -> List[str]:
    """Serializes the latest memory messages into a compact string list."""
    try:
        messages = memory.load_memory_variables({}).get("chat_history", [])
    except Exception:
        messages = []

    if not isinstance(messages, list):
        return []

    serialized: List[str] = []
    for message in messages[-limit:]:
        try:
            serialized.append(f"{message.type}: {message.content}")
        except Exception:
            serialized.append(str(message))
    return serialized


def _load_memory_log() -> Dict[str, Any]:
    """Loads the explicit persisted memory log for the agent."""
    return _load_json_file(AGENT_MEMORY_LOG_PATH, {"session_id": f"SESSION_{time.strftime('%Y%m%d_%H%M%S')}", "events": []})


def _append_memory_log(role: str, input_text: str, output_text: str, metadata: Dict[str, Any] | None = None) -> None:
    """Appends a compact persisted memory event for later reasoning."""
    store = _load_memory_log()
    event = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "role": role,
        "input": input_text,
        "output": output_text,
        "metadata": metadata or {},
    }
    store.setdefault("events", []).append(event)
    _write_json_file(AGENT_MEMORY_LOG_PATH, store)


def _load_recent_memory_events(limit: int = 5) -> List[Dict[str, Any]]:
    """Returns the most recent persisted memory events."""
    store = _load_memory_log()
    events = store.get("events", []) if isinstance(store, dict) else []
    if not isinstance(events, list):
        return []
    return events[-limit:]

class DroneSecurityAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", openai_api_key=settings.OPENAI_API_KEY)
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=2000,
            return_messages=True,
            memory_key="chat_history"
        )
        self.session_context = self._load_session_context()
        self.agent_runs = []

    def _record_memory(self, user_text: str, assistant_text: str) -> None:
        """Stores a compact interaction in ConversationSummaryBufferMemory."""
        try:
            self.memory.save_context({"input": user_text}, {"output": assistant_text})
        except Exception:
            # Memory is supportive, not critical for the agent to keep running.
            pass
        _append_memory_log("agent", user_text, assistant_text, {"source": "conversation_memory"})

    def _generate_reasoning(
        self,
        frame_id: str,
        telemetry: Dict[str, Any],
        analysis: Dict[str, Any],
        alert_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Uses the LLM plus memory/context to produce a structured operational recommendation."""
        recent_context = _load_recent_context_summaries(limit=4)
        recent_memory = _load_recent_memory_events(limit=4)
        prompt = f"""
You are a drone security operations analyst.
Use the current frame, telemetry, recent context summaries, and recent memory to produce a concise structured JSON response only.

Current frame:
{json.dumps({
    'frame_id': frame_id,
    'telemetry': telemetry,
    'analysis': analysis,
    'rule_alert': alert_summary,
}, indent=2)}

Recent context summaries:
{json.dumps(recent_context, indent=2)}

Recent memory events:
{json.dumps(recent_memory, indent=2)}

Return JSON with keys:
{{
  "security_assessment": "normal|watch|suspicious|critical",
  "confidence": 0.0,
  "reasoning": "brief explanation",
  "recommended_action": "what the operator should do",
  "follow_up_question": "one follow-up question to ask next",
  "objects_focus": ["key", "objects"],
  "context_signal": "what changed compared with prior frames"
}}
"""

        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", str(response))
            try:
                return json.loads(content.replace("```json", "").replace("```", "").strip())
            except Exception:
                return {
                    "security_assessment": alert_summary.get("severity", "NONE").lower() if alert_summary.get("severity") else "normal",
                    "confidence": 0.5,
                    "reasoning": content,
                    "recommended_action": "Review frame with operator",
                    "follow_up_question": "Should this frame be cross-checked against prior footage?",
                    "objects_focus": analysis.get("objects_detected", []),
                    "context_signal": "LLM returned unstructured text; using fallback reasoning.",
                }
        except Exception as exc:
            return {
                "security_assessment": alert_summary.get("severity", "NONE").lower() if alert_summary.get("severity") else "normal",
                "confidence": 0.4,
                "reasoning": f"Reasoning unavailable: {exc}",
                "recommended_action": "Manual review required",
                "follow_up_question": "What additional context is available for this frame?",
                "objects_focus": analysis.get("objects_detected", []),
                "context_signal": "LLM reasoning failed; using fallback.",
            }

    def _sync_session_context_from_frame(
        self,
        frame_id: str,
        telemetry: Dict[str, Any],
        analysis: Dict[str, Any],
        alert_summary: Dict[str, Any],
    ) -> None:
        """Updates rolling session stats from a newly processed frame."""
        self.session_context["frames_analyzed"] = int(self.session_context.get("frames_analyzed", 0)) + 1
        self.session_context["people_detected"] = int(self.session_context.get("people_detected", 0)) + int(
            analysis.get("people_count", 0) or 0
        )
        self.session_context["vehicles_detected"] = int(self.session_context.get("vehicles_detected", 0)) + len(
            analysis.get("vehicles_detected", []) or []
        )

        if alert_summary.get("alert_triggered"):
            self.session_context["total_alerts"] = int(self.session_context.get("total_alerts", 0)) + 1
            if alert_summary.get("severity") == "HIGH":
                self.session_context["high_alerts"] = int(self.session_context.get("high_alerts", 0)) + 1
            elif alert_summary.get("severity") == "MEDIUM":
                self.session_context["medium_alerts"] = int(self.session_context.get("medium_alerts", 0)) + 1

            incidents = self.session_context.setdefault("incidents", [])
            incidents.append(
                {
                    "frame_id": frame_id,
                    "timestamp": telemetry.get("timestamp"),
                    "location": telemetry.get("location"),
                    "severity": alert_summary.get("severity"),
                    "alert_type": alert_summary.get("alert_type"),
                }
            )

        locations = self.session_context.setdefault("locations_visited", [])
        location = telemetry.get("location")
        if location and location not in locations:
            locations.append(location)

        frames = int(self.session_context.get("frames_analyzed", 0))
        people = int(self.session_context.get("people_detected", 0))
        vehicles = int(self.session_context.get("vehicles_detected", 0))
        alerts = int(self.session_context.get("total_alerts", 0))
        self.session_context["running_narrative"] = (
            f"Analyzed {frames} frames so far. {people} people detected, {vehicles} vehicles detected, "
            f"{alerts} alerts generated across {len(locations)} locations."
        )

        self._save_session_context()

    def _mark_frame_processed(self, frame_id: str) -> bool:
        """Marks a frame as processed and returns False if it was already seen."""
        processed_frames = self.session_context.setdefault("agent_processed_frames", [])
        if frame_id in processed_frames:
            return False
        processed_frames.append(frame_id)
        return True

    def _build_alert_payload(
        self,
        frame_id: str,
        analysis: Dict[str, Any],
        telemetry: Dict[str, Any],
        severity: str,
        message: str,
    ) -> Dict[str, Any]:
        """Creates a standard alert JSON payload for a frame."""
        alert_type = "manual_trigger"
        if severity == "HIGH":
            alert_type = "high_severity_event"
        elif severity == "MEDIUM":
            alert_type = "medium_severity_event"
        elif severity == "LOW":
            alert_type = "low_severity_event"

        return {
            "alert_id": f"ALT_{datetime.utcnow().strftime('%Y%m%d')}_{_extract_frame_number(frame_id):03}",
            "frame_id": frame_id,
            "timestamp": telemetry.get("timestamp", analysis.get("timestamp", "00:00:00")),
            "location": telemetry.get("location", analysis.get("location", "unknown")),
            "alert_triggered": severity != "NONE",
            "severity": severity,
            "alert_type": alert_type,
            "message": message,
            "objects_involved": analysis.get("objects_detected", []),
            "rule_triggered": "manual_override",
            "llm_validated": False,
            "llm_reasoning": "Created by agent trigger_alert tool.",
            "recommended_action": message,
            "acknowledged": False,
            "auto_escalate": severity == "HIGH",
        }

    def _load_session_context(self) -> Dict[str, Any]:
        if SESSION_CONTEXT_PATH.exists():
            with open(SESSION_CONTEXT_PATH, "r", encoding="utf-8") as f:
                session_context = json.load(f)
                if isinstance(session_context, dict):
                    session_context.setdefault("agent_processed_frames", [])
                    return session_context
        return _default_session_context()

    def _save_session_context(self):
        with open(SESSION_CONTEXT_PATH, "w", encoding="utf-8") as f:
            json.dump(self.session_context, f, indent=2)

    def analyze_frame(self, frame_id: str) -> Dict[str, Any]:
        """Analyzes a single frame, updates session state, and records the run."""
        if not self._mark_frame_processed(frame_id):
            existing = _load_json_file(ANALYSIS_DIR / f"{frame_id}_analysis.json", {})
            existing["duplicate"] = True
            return existing

        telemetry = self.get_telemetry(frame_id)
        image_path = settings.EXTRACTED_DIR / f"{frame_id}.jpg"
        start = time.time()
        analysis = analyze_frame(frame_id, image_path, telemetry)
        alert_summary = rule_based_alert(analysis, telemetry)

        if alert_summary.get("alert_triggered"):
            self.trigger_alert(
                frame_id=frame_id,
                severity=alert_summary.get("severity", "NONE"),
                message=f"{alert_summary.get('rule_triggered', 'alert')} detected for {frame_id}",
            )

        # Agent reasoning feature removed - direct analysis only
        self._sync_session_context_from_frame(frame_id, telemetry, analysis, alert_summary)
        _persist_context_summary(frame_id, telemetry, analysis, self.session_context, alert_summary)

        summary = (
            f"Frame {frame_id} analyzed at {telemetry.get('location', 'unknown location')}. "
            f"People: {analysis.get('people_count', 0)}. "
            f"Vehicles: {len(analysis.get('vehicles_detected', []) or [])}. "
            f"Alert: {alert_summary.get('severity', 'NONE')}."
        )
        self._record_memory(f"Analyze {frame_id}", summary)
        _append_memory_log(
            "analysis",
            f"Analyze {frame_id}",
            summary,
            {
                "frame_id": frame_id,
                "severity": alert_summary.get("severity", "NONE"),
                "people_count": analysis.get("people_count", 0),
                "vehicles_count": len(analysis.get("vehicles_detected", []) or []),
            },
        )

        run_payload = {
                "run_id": len(self.agent_runs) + 1,
                "frame_id": frame_id,
                "input": f"Analyze {frame_id} for security threats",
                "agent_reasoning": summary,
                "structured_reasoning": reasoning,
                "tools_used": ["analyze_frame", "trigger_alert" if alert_summary.get("alert_triggered") else "analyze_frame"],
                "output": analysis,
                "alerts_generated": 1 if alert_summary.get("alert_triggered") else 0,
                "processing_time_ms": int((time.time() - start) * 1000),
            }
        self.log_agent_run(run_payload)

        enriched_output = dict(analysis)
        enriched_output["agent_reasoning"] = reasoning
        enriched_output["session_context"] = self.session_context
        return enriched_output

    def search_frames(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches Pinecone for semantically similar frame descriptions."""
        results = search_frames(query, top_k)
        hits = results.get("results", []) if isinstance(results, dict) else []
        self._record_memory(f"Search frames: {query}", f"Found {len(hits)} matching frames.")
        return hits

    def get_telemetry(self, frame_id: str) -> Dict[str, Any]:
        """Loads telemetry for a frame from disk."""
        telemetry_path = TELEMETRY_DIR / f"{frame_id}_telemetry.json"
        return _load_json_file(telemetry_path, {})

    def trigger_alert(self, frame_id: str, severity: str, message: str) -> Dict[str, Any]:
        """Creates and saves a manual alert for a frame."""
        telemetry = self.get_telemetry(frame_id)
        analysis = _load_json_file(ANALYSIS_DIR / f"{frame_id}_analysis.json", {})
        alert_payload = self._build_alert_payload(frame_id, analysis, telemetry, severity, message)
        alert_path = ALERTS_DIR / f"{frame_id}_alert.json"
        _write_json_file(alert_path, alert_payload)
        self._record_memory(
            f"Trigger alert for {frame_id}",
            f"Saved {severity} alert at {telemetry.get('location', 'unknown location')}: {message}",
        )
        return alert_payload

    def get_session_context(self) -> Dict[str, Any]:
        return self.session_context

    def query_event_history(self, object_type: str, time_range: str) -> List[Dict[str, Any]]:
        """Finds historical events that match an object type and optional time range."""
        object_type_normalized = object_type.lower().strip()
        all_analysis = _load_json_file(settings.ANALYSIS_DIR / "all_analysis.json", [])
        all_alerts = _load_json_file(settings.ALERTS_DIR / "all_alerts.json", {"alerts": []}).get("alerts", [])

        matches: List[Dict[str, Any]] = []
        for item in all_analysis:
            timestamp = str(item.get("timestamp", "00:00:00"))
            if not _match_time_range(timestamp, time_range):
                continue

            objects_detected = [str(obj).lower() for obj in item.get("objects_detected", []) or []]
            vehicles_detected = [str(vehicle).lower() for vehicle in item.get("vehicles_detected", []) or []]
            activity_text = str(item.get("activity", "") or "").lower()
            haystack = " ".join(objects_detected + vehicles_detected + [activity_text])

            if object_type_normalized in haystack:
                matches.append(
                    {
                        "frame_id": item.get("frame_id"),
                        "timestamp": timestamp,
                        "location": item.get("location"),
                        "activity": item.get("activity"),
                        "threat_assessment": item.get("threat_assessment"),
                    }
                )

        for alert in all_alerts:
            timestamp = str(alert.get("timestamp", "00:00:00"))
            if not _match_time_range(timestamp, time_range):
                continue
            alert_text = f"{alert.get('alert_type', '')} {alert.get('message', '')} {alert.get('objects_involved', [])}".lower()
            if object_type_normalized in alert_text:
                matches.append(
                    {
                        "frame_id": alert.get("frame_id"),
                        "timestamp": timestamp,
                        "location": alert.get("location"),
                        "severity": alert.get("severity"),
                        "alert_type": alert.get("alert_type"),
                    }
                )

        self._record_memory(
            f"Query event history for {object_type}",
            f"Found {len(matches)} events in range '{time_range}'.",
        )
        return matches

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Uses the agent memory, session context, and search/history to answer a follow-up question."""
        search_hits = self.search_frames(question, top_k=5)
        history_hits = self.query_event_history(question, "all")
        prompt = f"""
You are the Drone Security Analyst Agent.
Answer the operator's question using the session context, retrieved frames, and event history.
Be precise, concise, and operationally useful.

Question:
{question}

Session context:
{json.dumps(self.session_context, indent=2)}

Retrieved frames:
{json.dumps(search_hits, indent=2)}

Event history:
{json.dumps(history_hits, indent=2)}

Respond in JSON with keys:
{{
  "answer": "direct answer",
  "confidence": 0.0,
  "sources": ["frame_001"],
  "next_action": "what to do next",
  "reasoning": "short reasoning"
}}
"""
        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", str(response))
            parsed = json.loads(content.replace("```json", "").replace("```", "").strip())
        except Exception:
            parsed = {
                "answer": f"I found {len(search_hits)} relevant frame hits and {len(history_hits)} matching historical events.",
                "confidence": 0.45,
                "sources": [hit.get("frame_id") for hit in search_hits if isinstance(hit, dict) and hit.get("frame_id")][:5],
                "next_action": "Review the top matching frames and alerts.",
                "reasoning": "Fallback answer used because the LLM response was not parseable.",
            }

        self._record_memory(f"Question: {question}", parsed.get("answer", ""))
        return parsed

    def log_agent_run(self, run: Dict[str, Any]):
        self.agent_runs.append(run)
        with open(AGENT_RUNS_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": f"SESSION_{time.strftime('%Y%m%d_%H%M%S')}",
                "total_runs": len(self.agent_runs),
                "runs": self.agent_runs
            }, f, indent=2)

def run_demo() -> None:
    """Runs a small demo so the module produces visible output when executed."""
    print("\n🤖 Running DroneSecurityAgent demo...")
    agent = DroneSecurityAgent()
    print(f"📍 Session context: {json.dumps(agent.get_session_context(), indent=2)}")

    try:
        search_results = agent.search_frames("person detected", top_k=3)
        print(f"🔍 Search hits: {len(search_results)}")
    except Exception as exc:
        print(f"❌ Search demo failed: {exc}")

    try:
        history = agent.query_event_history("person", "all")
        print(f"🕓 Event history matches: {len(history)}")
    except Exception as exc:
        print(f"❌ History demo failed: {exc}")

    try:
        answer = agent.answer_question("What is the most important incident today?")
        print(f"🧠 Agent answer: {answer.get('answer', '')}")
    except Exception as exc:
        print(f"❌ Question demo failed: {exc}")

    print("✅ Agent demo complete.")


if __name__ == "__main__":
    run_demo()
