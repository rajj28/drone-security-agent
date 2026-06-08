"""
context_manager.py - Session context across frames (situation memory).

Persistence order for local testing:
  1. In-process memory (current run)
  2. Local JSON under settings.SESSION_DIR / rich_context.json
  3. MongoDB (optional, when MONGODB_URI connects)
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

# Try to import MongoDB
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("[WARNING] PyMongo not available, using in-memory context only")


@dataclass
class FrameContext:
    """Context for a single frame"""
    frame_id: str
    timestamp: float
    threat_level: str
    threat_type: str
    people_count: int
    key_activities: List[str] = field(default_factory=list)
    security_signals: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass  
class SessionContext:
    """Complete context for a video session"""
    session_id: str
    frames: List[FrameContext] = field(default_factory=list)
    running_summary: str = ""
    total_people_seen: int = 0
    threat_timeline: List[Dict] = field(default_factory=list)
    suspicious_activities: List[str] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_frame_context(self, frame_data: Dict[str, Any]):
        """Add context from a new frame"""
        frame_ctx = FrameContext(
            frame_id=frame_data.get('frame_id', 'unknown'),
            timestamp=frame_data.get('timestamp', 0),
            threat_level=frame_data.get('threat_level', 'CLEAR'),
            threat_type=frame_data.get('threat_type', 'clear'),
            people_count=frame_data.get('people_count', 0),
            key_activities=frame_data.get('activity', '').split(', ') if frame_data.get('activity') else [],
            security_signals=frame_data.get('security_signals', []),
            summary=frame_data.get('vlm_description', '')[:200]  # Truncate for summary
        )
        self.frames.append(frame_ctx)
        
        # Update running stats
        self.total_people_seen = max(self.total_people_seen, frame_ctx.people_count)
        
        # Track threats
        if frame_ctx.threat_level in ['CRITICAL', 'HIGH']:
            self.threat_timeline.append({
                'frame': frame_ctx.frame_id,
                'level': frame_ctx.threat_level,
                'type': frame_ctx.threat_type,
                'time': frame_ctx.timestamp
            })
        
        # Track suspicious patterns
        for signal in frame_ctx.security_signals:
            if signal not in self.suspicious_activities:
                self.suspicious_activities.append(signal)
        
        # Update running summary (keep last 5 frames for context)
        recent_frames = self.frames[-5:]
        self.running_summary = self._generate_summary(recent_frames)
        self.last_updated = datetime.now().isoformat()
    
    def _generate_summary(self, recent_frames: List[FrameContext]) -> str:
        """Generate a summary of recent activity"""
        if not recent_frames:
            return "No activity recorded yet."
        
        summaries = []
        people_counts = [f.people_count for f in recent_frames]
        avg_people = sum(people_counts) / len(people_counts) if people_counts else 0
        
        # Check for patterns
        high_threat_count = sum(1 for f in recent_frames if f.threat_level in ['CRITICAL', 'HIGH'])
        
        if high_threat_count > 0:
            summaries.append(f"⚠️ {high_threat_count} high-threat events in recent frames")
        
        if avg_people > 0:
            summaries.append(f"👥 ~{int(avg_people)} people visible on average")
        
        # Check for movement patterns
        if len(recent_frames) >= 2:
            if recent_frames[-1].people_count > recent_frames[0].people_count:
                summaries.append("📈 More people appearing (possible entry/approach)")
            elif recent_frames[-1].people_count < recent_frames[0].people_count:
                summaries.append("📉 People dispersing (possible exit)")
        
        # Activity patterns
        all_signals = []
        for f in recent_frames:
            all_signals.extend(f.security_signals)
        
        if all_signals:
            unique_signals = list(set(all_signals))[:3]  # Top 3 unique signals
            summaries.append(f"🎯 Behaviors: {', '.join(unique_signals)}")
        
        return " | ".join(summaries) if summaries else "Normal monitoring in progress."
    
    def get_context_for_frame(self, frame_number: int, window_size: int = 5) -> str:
        """Get context summary for a specific frame considering previous frames"""
        # Find frame index
        frame_idx = None
        for i, f in enumerate(self.frames):
            if f.frame_id == f"frame_{frame_number:03d}":
                frame_idx = i
                break
        
        if frame_idx is None:
            return self.running_summary  # Return general summary if frame not found
        
        # Get window of previous frames
        start_idx = max(0, frame_idx - window_size)
        previous_frames = self.frames[start_idx:frame_idx]
        
        if not previous_frames:
            return "First frame in sequence - establishing baseline."
        
        # Build contextual summary
        context_parts = []
        
        # Recent threat history
        recent_threats = [f for f in previous_frames if f.threat_level in ['CRITICAL', 'HIGH']]
        if recent_threats:
            threat_frames = [f.frame_id for f in recent_threats[-3:]]  # Last 3 threats
            context_parts.append(f"⚠️ Previous threats detected in: {', '.join(threat_frames)}")
        
        # People movement
        people_trend = [f.people_count for f in previous_frames]
        if people_trend:
            if people_trend[-1] > people_trend[0]:
                context_parts.append(f"📈 People count increased from {people_trend[0]} to {people_trend[-1]}")
            elif people_trend[-1] < people_trend[0]:
                context_parts.append(f"📉 People count decreased from {people_trend[0]} to {people_trend[-1]}")
        
        # Accumulated suspicious behaviors
        all_signals = []
        for f in previous_frames:
            all_signals.extend(f.security_signals)
        
        if all_signals:
            signal_counts = {}
            for s in all_signals:
                signal_counts[s] = signal_counts.get(s, 0) + 1
            
            # Show repeated behaviors
            repeated = [s for s, c in signal_counts.items() if c > 1]
            if repeated:
                context_parts.append(f"🔄 Repeated behaviors: {', '.join(repeated[:3])}")
        
        # Overall situation
        latest = previous_frames[-1]
        context_parts.append(f"📍 Previous frame ({latest.frame_id}): {latest.threat_level} - {latest.summary[:100]}")
        
        return "\n".join(context_parts)


class ContextStore:
    """Session context: memory + local JSON; MongoDB optional for cloud deploy."""

    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self._memory_store: Dict[str, SessionContext] = {}
        self._use_mongo = os.environ.get("USE_MONGO_CONTEXT", "").lower() in ("1", "true", "yes")

        if MONGODB_AVAILABLE and self._use_mongo:
            self._connect_mongodb()

    def _local_context_path(self) -> Path:
        from src.config import settings

        settings.SESSION_DIR.mkdir(parents=True, exist_ok=True)
        return settings.SESSION_DIR / "rich_context.json"

    def _context_to_dict(self, session_id: str, context: SessionContext) -> Dict[str, Any]:
        return {
            "session_id": session_id,
            "frames": [asdict(f) for f in context.frames],
            "running_summary": context.running_summary,
            "total_people_seen": context.total_people_seen,
            "threat_timeline": context.threat_timeline,
            "suspicious_activities": context.suspicious_activities,
            "last_updated": context.last_updated,
        }

    def _context_from_dict(self, doc: Dict[str, Any]) -> SessionContext:
        context = SessionContext(
            session_id=doc["session_id"],
            running_summary=doc.get("running_summary", ""),
            total_people_seen=doc.get("total_people_seen", 0),
            threat_timeline=doc.get("threat_timeline", []),
            suspicious_activities=doc.get("suspicious_activities", []),
            last_updated=doc.get("last_updated", ""),
        )
        for frame in doc.get("frames", []):
            context.frames.append(FrameContext(**frame))
        return context

    def _save_local(self, session_id: str, context: SessionContext) -> None:
        path = self._local_context_path()
        payload = self._context_to_dict(session_id, context)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[CONTEXT] Saved local context -> {path}")

    def _load_local(self, session_id: str) -> Optional[SessionContext]:
        path = self._local_context_path()
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                doc = json.load(handle)
            if doc.get("session_id") != session_id:
                return None
            return self._context_from_dict(doc)
        except Exception as exc:
            print(f"[CONTEXT] Local context load failed: {exc}")
            return None
    
    def _connect_mongodb(self):
        """Connect to MongoDB"""
        try:
            mongo_uri = os.environ.get('MONGODB_URI', '')
            if mongo_uri:
                self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
                self.client.admin.command("ping")
                self.db = self.client["drone_security"]
                self.collection = self.db["session_contexts"]
                print("[CONTEXT] MongoDB connected (USE_MONGO_CONTEXT=true)")
            else:
                print("[CONTEXT] No MONGODB_URI — local JSON context only")
        except Exception as e:
            print(f"[CONTEXT] MongoDB connection failed: {e} — local JSON context only")
            self.client = None
    
    def save_context(self, session_id: str, context: SessionContext):
        """Save context to memory, local JSON, and optionally MongoDB."""
        try:
            self._memory_store[session_id] = context
            self._save_local(session_id, context)

            if self.collection:
                self.collection.update_one(
                    {"session_id": session_id},
                    {"$set": self._context_to_dict(session_id, context)},
                    upsert=True,
                )
                print(f"[CONTEXT] Saved MongoDB context for session {session_id}")
        except Exception as e:
            print(f"[CONTEXT] Save failed (memory/local may still be OK): {e}")

    def load_context(self, session_id: str) -> Optional[SessionContext]:
        """Load context: memory -> local file -> MongoDB."""
        if session_id in self._memory_store:
            return self._memory_store[session_id]

        context = self._load_local(session_id)
        if context is not None:
            self._memory_store[session_id] = context
            print(f"[CONTEXT] Loaded local context for session {session_id}")
            return context

        if self.collection:
            try:
                doc = self.collection.find_one({"session_id": session_id})
                if doc:
                    context = self._context_from_dict(doc)
                    self._memory_store[session_id] = context
                    self._save_local(session_id, context)
                    print(f"[CONTEXT] Loaded MongoDB context for session {session_id}")
                    return context
            except Exception as e:
                print(f"[CONTEXT] MongoDB load failed: {e}")

        return None
    
    def get_or_create_context(self, session_id: str) -> SessionContext:
        """Get existing context or create new one"""
        context = self.load_context(session_id)
        if context is None:
            context = SessionContext(session_id=session_id)
            self._memory_store[session_id] = context
        return context


# Global context store
_context_store = ContextStore()


def get_session_context(session_id: str) -> SessionContext:
    """Get or create session context"""
    return _context_store.get_or_create_context(session_id)


def save_session_context(session_id: str, context: SessionContext):
    """Save session context"""
    _context_store.save_context(session_id, context)


def update_frame_context(session_id: str, frame_data: Dict[str, Any]):
    """Update context with new frame data"""
    context = get_session_context(session_id)
    context.add_frame_context(frame_data)
    save_session_context(session_id, context)
    return context


def get_frame_context_summary(session_id: str, frame_number: int, window_size: int = 5) -> str:
    """Get context summary for a specific frame"""
    context = get_session_context(session_id)
    return context.get_context_for_frame(frame_number, window_size)
