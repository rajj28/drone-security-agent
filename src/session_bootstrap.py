"""
session_bootstrap.py — Apply one-video-one-session directory layout from SESSION_ID.

All pipeline steps should call apply_session_layout() so outputs land under:
  data/sessions/{SESSION_ID}/
"""

import os
from pathlib import Path
from typing import Optional


def apply_session_layout(session_id: Optional[str] = None) -> Path:
    from src.config import settings
    """
    Point settings at data/sessions/{SESSION_ID}/ and create directories.
    Returns the session root path.
    """
    sid = session_id or os.environ.get("SESSION_ID", "").strip()
    if not sid:
        return settings.SESSION_DIR

    root = Path("data") / "sessions" / sid
    root.mkdir(parents=True, exist_ok=True)

    settings.SESSION_DIR = root
    settings.EXTRACTED_DIR = root / "extracted"
    settings.OUTPUTS_DIR = root / "outputs"
    settings.TELEMETRY_DIR = root / "telemetry"
    settings.ANALYSIS_DIR = root / "analysis"
    settings.ALERTS_DIR = root / "alerts"
    settings.INDEX_DIR = root / "index"
    (settings.INDEX_DIR / "query_results").mkdir(parents=True, exist_ok=True)

    for path in (
        settings.EXTRACTED_DIR,
        settings.OUTPUTS_DIR,
        settings.TELEMETRY_DIR,
        settings.ANALYSIS_DIR,
        settings.ALERTS_DIR,
        settings.INDEX_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["SESSION_ID"] = sid
    print(f"[SESSION] Layout: {root}")
    return root
