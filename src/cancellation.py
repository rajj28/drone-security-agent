import threading

_lock = threading.Lock()
_cancelled = set()

def cancel_session(session_id: str):
    """Mark a session as cancelled in memory."""
    with _lock:
        _cancelled.add(session_id)

def is_cancelled(session_id: str) -> bool:
    """Check if a session has been cancelled."""
    if not session_id:
        return False
    with _lock:
        return session_id in _cancelled

def clear_cancellation(session_id: str):
    """Remove a session from the cancellation list."""
    with _lock:
        _cancelled.discard(session_id)
