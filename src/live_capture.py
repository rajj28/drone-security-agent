"""
live_capture.py — Live frame ingestion for drone / mobile camera feeds.

Two ingestion modes, both producing the exact same session layout as an
uploaded video (data/sessions/{id}/extracted/frame_NNN.jpg + extraction_log.json),
so the existing telemetry → vision → alerts → tracking → summary pipeline
runs on live captures unchanged:

- "browser": the phone/laptop browser captures camera frames (getUserMedia)
  and POSTs each JPEG to /live/{session_id}/frame.
- "stream": the server pulls frames itself from a drone / IP-camera stream URL
  (RTSP, RTMP, HTTP/MJPEG — anything OpenCV's ffmpeg backend can open) on a
  background thread, saving one frame every capture_interval seconds.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LiveSession:
    """Tracks one live capture session and its saved frames."""

    def __init__(self, session_id: str, extracted_dir: Path, source: str, max_frames: int = 100):
        self.session_id = session_id
        self.extracted_dir = Path(extracted_dir)
        self.source = source  # "browser" | "stream"
        self.max_frames = max_frames
        self.start_time = time.time()
        self.frames: List[Dict] = []
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    def add_frame_bytes(self, data: bytes) -> Dict:
        """Save one JPEG frame to disk with a real elapsed-time timestamp."""
        with self._lock:
            idx = len(self.frames) + 1
            frame_id = f"frame_{idx:03d}"
            filename = f"{frame_id}.jpg"
            (self.extracted_dir / filename).write_bytes(data)
            entry = {
                "filename": filename,
                "frame_id": frame_id,
                "timestamp_seconds": round(time.time() - self.start_time, 2),
            }
            self.frames.append(entry)
            return entry

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def is_full(self) -> bool:
        return len(self.frames) >= self.max_frames

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)

    def write_extraction_log(self, outputs_dir: Path) -> Path:
        """Write extraction_log.json in the format the telemetry stage expects."""
        outputs_dir = Path(outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        duration = max(1, int(round(time.time() - self.start_time)))
        log_data = {
            "total_frames": len(self.frames),
            "total_frames_extracted": len(self.frames),
            "video_duration_seconds": duration,
            "fps": round(len(self.frames) / duration, 3) if duration else 1,
            "resolution": "live",
            "codec": "live-jpeg",
            "source": f"live-{self.source}",
            "frames": self.frames,
        }
        log_path = outputs_dir / "extraction_log.json"
        log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
        return log_path


# Registry of active live sessions (in-memory; a live capture is bound to one process)
_live_sessions: Dict[str, LiveSession] = {}
_registry_lock = threading.Lock()


def start_live_session(
    session_id: str,
    extracted_dir: Path,
    source: str,
    max_frames: int = 100,
    stream_url: Optional[str] = None,
    capture_interval: float = 2.0,
) -> LiveSession:
    session = LiveSession(session_id, extracted_dir, source, max_frames)
    if source == "stream":
        if not stream_url:
            raise ValueError("stream source requires a stream_url")
        session.thread = threading.Thread(
            target=_stream_worker,
            args=(session, stream_url, capture_interval),
            daemon=True,
            name=f"live-stream-{session_id[:8]}",
        )
        session.thread.start()
    with _registry_lock:
        _live_sessions[session_id] = session
    return session


def get_live_session(session_id: str) -> Optional[LiveSession]:
    return _live_sessions.get(session_id)


def remove_live_session(session_id: str) -> Optional[LiveSession]:
    with _registry_lock:
        return _live_sessions.pop(session_id, None)


def _stream_worker(session: LiveSession, stream_url: str, interval: float):
    """Continuously read a drone/IP-camera stream, saving one frame per interval.

    Reads every frame to keep the stream buffer drained (RTSP sources stall
    otherwise) but only persists a frame each `interval` seconds.
    """
    import cv2

    logger.info(f"[LIVE {session.session_id}] Opening stream: {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        session.error = f"Could not open stream: {stream_url}"
        logger.error(f"[LIVE {session.session_id}] {session.error}")
        return

    last_saved = 0.0
    consecutive_failures = 0
    try:
        while not session.stop_event.is_set() and not session.is_full():
            ok, frame = cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= 50:
                    session.error = "Stream ended or became unreadable"
                    logger.warning(f"[LIVE {session.session_id}] {session.error}")
                    break
                time.sleep(0.2)
                continue
            consecutive_failures = 0
            now = time.time()
            if now - last_saved >= interval:
                ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok_enc:
                    entry = session.add_frame_bytes(buf.tobytes())
                    last_saved = now
                    logger.info(f"[LIVE {session.session_id}] Captured {entry['filename']} @ {entry['timestamp_seconds']}s")
    finally:
        cap.release()
        logger.info(f"[LIVE {session.session_id}] Stream capture finished with {session.frame_count} frames")
