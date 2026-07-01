"""
unified_context.py — Single context layer for VLM, cloud analyzer, and agent.

Writes per session:
  - session_context.json
  - context_summaries.json
  - rich_context.json (via context_manager)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings

_unified_instance: Optional["UnifiedContext"] = None


def _frame_number_from_id(frame_id: str) -> int:
    try:
        return int(str(frame_id).split("_")[-1])
    except (ValueError, IndexError):
        return 1


def _load_json(path: Path, default: Any) -> Any:
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


class UnifiedContext:
    """Session-scoped context for VLMs and agents."""

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.session_id = os.environ.get("SESSION_ID", self.session_dir.name)
        self.session_context_path = self.session_dir / "session_context.json"
        self.summaries_path = self.session_dir / "context_summaries.json"

    def ensure_telemetry_session_id(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        telemetry = dict(telemetry)
        telemetry["session_id"] = self.session_id
        telemetry.setdefault("frame_id", telemetry.get("frame_id", "unknown"))
        return telemetry

    def _default_session_context(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
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

    def load_session_context(self) -> Dict[str, Any]:
        data = _load_json(self.session_context_path, {})
        if isinstance(data, dict):
            return {**self._default_session_context(), **data}
        return self._default_session_context()

    def load_summaries_store(self) -> Dict[str, Any]:
        data = _load_json(self.summaries_path, {})
        if isinstance(data, dict):
            data.setdefault("session_id", self.session_id)
            data.setdefault("summaries", [])
            return data
        return {"session_id": self.session_id, "summaries": []}

    def _build_summary_text(
        self,
        frame_id: str,
        telemetry: Dict[str, Any],
        analysis: Dict[str, Any],
        session_context: Dict[str, Any],
        alert_summary: Dict[str, Any],
    ) -> str:
        frames_analyzed = session_context.get("frames_analyzed", 0)
        total_alerts = session_context.get("total_alerts", 0)
        people_detected = session_context.get("people_detected", 0)
        vehicles_detected = session_context.get("vehicles_detected", 0)
        location = telemetry.get("location", "unknown location")
        timestamp = telemetry.get("timestamp", "unknown time")
        activity = analysis.get("activity", "No activity reported")
        threat = str(
            analysis.get("threat_level")
            or analysis.get("threat_assessment", "none")
            or "none"
        ).upper()
        reasoning = str(analysis.get("alert_reasoning", "") or "").strip()

        parts = [
            f"Frame {frame_id} analyzed at {location} ({timestamp}).",
            f"Cumulative: {frames_analyzed} frames, {total_alerts} alerts, "
            f"{people_detected} people, {vehicles_detected} vehicles.",
            f"Activity: {activity}. Threat: {threat}.",
        ]
        if alert_summary.get("alert_triggered"):
            parts.append(
                f"ALERT {alert_summary.get('severity', '?')}: "
                f"{alert_summary.get('alert_type', 'unknown')}."
            )
        if reasoning:
            parts.append(f"Reasoning: {reasoning[:300]}")
        return " ".join(parts)

    def _update_session_context(
        self,
        session_context: Dict[str, Any],
        telemetry: Dict[str, Any],
        analysis: Dict[str, Any],
        alert_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
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
            sev = alert_summary.get("severity")
            if sev == "HIGH":
                session_context["high_alerts"] = int(session_context.get("high_alerts", 0)) + 1
            elif sev == "MEDIUM":
                session_context["medium_alerts"] = int(session_context.get("medium_alerts", 0)) + 1
            session_context.setdefault("incidents", []).append(
                {
                    "frame_id": telemetry.get("frame_id"),
                    "timestamp": telemetry.get("timestamp"),
                    "location": location,
                    "severity": sev,
                    "alert_type": alert_summary.get("alert_type"),
                    "threat_type": analysis.get("threat_type"),
                    "description": analysis.get("vlm_description"),
                    "activity": analysis.get("activity"),
                }
            )

        n = int(session_context.get("frames_analyzed", 0))
        session_context["running_narrative"] = (
            f"Analyzed {n} frames; {session_context.get('total_alerts', 0)} alerts; "
            f"max people {session_context.get('people_detected', 0)}."
        )
        return session_context

    def _sync_rich_context(
        self,
        frame_id: str,
        telemetry: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> None:
        try:
            from src.context_manager import update_frame_context

            threat = str(
                analysis.get("threat_level")
                or analysis.get("threat_assessment", "CLEAR")
                or "CLEAR"
            ).upper()
            frame_data = {
                "frame_id": frame_id,
                "timestamp": telemetry.get("unix_time", 0),
                "threat_level": threat,
                "threat_type": analysis.get("threat_type", threat.lower()),
                "people_count": int(analysis.get("people_count", 0) or 0),
                "activity": analysis.get("activity", ""),
                "security_signals": analysis.get("security_signals", []) or [],
                "vlm_description": (
                    analysis.get("vlm_description")
                    or analysis.get("description")
                    or ""
                )[:500],
            }
            update_frame_context(self.session_id, frame_data)
        except Exception as exc:
            print(f"[CONTEXT] Rich context sync skipped: {exc}")

    def record_frame(
        self,
        frame_id: str,
        telemetry: Dict[str, Any],
        analysis: Dict[str, Any],
        alert_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Persist all context artifacts for one analyzed frame."""
        telemetry = self.ensure_telemetry_session_id(telemetry)
        telemetry["frame_id"] = frame_id

        session_context = self._update_session_context(
            self.load_session_context(), telemetry, analysis, alert_summary
        )
        _write_json(self.session_context_path, session_context)

        summary_text = self._build_summary_text(
            frame_id, telemetry, analysis, session_context, alert_summary
        )
        store = self.load_summaries_store()
        store["summaries"].append(
            {
                "after_frame": frame_id,
                "timestamp": telemetry.get("timestamp"),
                "frames_analyzed": session_context.get("frames_analyzed"),
                "context_summary": summary_text,
                "running_stats": {
                    "total_alerts": session_context.get("total_alerts", 0),
                    "people_detected": session_context.get("people_detected", 0),
                },
                "agent_memory_snapshot": session_context.get("running_narrative", ""),
            }
        )
        _write_json(self.summaries_path, store)
        self._sync_rich_context(frame_id, telemetry, analysis)
        return session_context


def get_unified_context(session_dir: Optional[Path] = None) -> UnifiedContext:
    global _unified_instance
    root = Path(session_dir) if session_dir else settings.SESSION_DIR
    if _unified_instance is None or _unified_instance.session_dir != root:
        _unified_instance = UnifiedContext(root)
    return _unified_instance


def get_vlm_prompt_context(frame_id: str, window: int = 5) -> str:
    """Text block injected into VLM prompts (prior frames)."""
    session_id = os.environ.get("SESSION_ID", settings.SESSION_DIR.name)
    frame_num = _frame_number_from_id(frame_id)

    try:
        from src.context_manager import get_frame_context_summary

        text = get_frame_context_summary(session_id, frame_num, window_size=window)
        if text and text.strip():
            return text
    except Exception:
        pass

    summaries = get_unified_context().load_summaries_store().get("summaries", [])
    recent = summaries[-window:] if summaries else []
    if not recent:
        return "No prior frames in this session."

    lines = []
    for entry in recent:
        lines.append(entry.get("context_summary", ""))
    return "\n".join(lines)


def get_recent_context_for_agent(limit: int = 5) -> List[Dict[str, Any]]:
    store = get_unified_context().load_summaries_store()
    summaries = store.get("summaries", [])
    if not isinstance(summaries, list):
        return []
    return summaries[-limit:]
