"""
telemetry_generator.py — Generates realistic simulated telemetry for each extracted frame.

- Cycles through locations
- Updates battery, after_hours, restricted_zone flags
- Saves per-frame and combined telemetry JSONs
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta
from src.config import settings

from src.session_bootstrap import apply_session_layout

apply_session_layout()


def generate_telemetry(
    frame_metadata: List[Dict],
    start_unix: int = settings.VIDEO_START_UNIX,
    output_dir: Path = settings.TELEMETRY_DIR
) -> List[Dict]:
    """
    Generates telemetry for each frame and saves per-frame JSON.
    Returns list of all telemetry dicts.
    """
    print("\nGenerating telemetry for frames...")
    locations = settings.LOCATIONS
    battery = 100.0
    battery_drop = 100.0 / max(1, len(frame_metadata))
    all_telemetry = []
    for i, frame in enumerate(frame_metadata):
        frame_id = f"frame_{i+1:03}"
        # Handle both 'timestamp_seconds' and 'timestamp' field names
        ts_seconds = frame.get("timestamp_seconds") or frame.get("timestamp", 0)
        unix_time = start_unix + ts_seconds
        dt = datetime.utcfromtimestamp(unix_time)
        timestamp = dt.strftime("%H:%M:%S")
        location = locations[i % len(locations)]
        zone = "restricted" if location in ["Back Entrance", "Restricted Zone"] else "public"
        after_hours = dt.hour < 6 or dt.hour >= 22
        session_id = os.environ.get("SESSION_ID", "local_dev")
        telemetry = {
            "frame_id": frame_id,
            "session_id": session_id,
            "timestamp": timestamp,
            "unix_time": unix_time,
            "drone": {
                "latitude": 18.5204,
                "longitude": 73.8567,
                "altitude_m": 15.0,
                "battery_pct": round(battery, 1),
                "heading_degrees": 90,
                "speed_ms": 0.0,
                "status": "patrolling",
                "signal_strength": "strong"
            },
            "location": location,
            "zone": zone,
            "weather": "clear",
            "temperature_c": 24.5,
            "visibility": "good",
            "is_restricted_zone": location in ["Back Entrance", "Restricted Zone"],
            "is_after_hours": after_hours
        }
        # Save per-frame telemetry
        out_path = output_dir / f"{frame_id}_telemetry.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2)
        all_telemetry.append(telemetry)
        battery = max(0.0, battery - battery_drop)
        print(f"Telemetry for {frame_id} at {timestamp} ({location})")
    # Save combined
    combined_path = output_dir / "all_telemetry.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_telemetry, f, indent=2)
    print(f"\nCombined telemetry saved to {combined_path}")
    return all_telemetry

if __name__ == "__main__":
    import os
    
    # Check for SESSION_ID to use session-specific directories
    session_id = os.environ.get("SESSION_ID")
    
    if session_id:
        # Use session-specific directories
        session_dir = Path("data") / "sessions" / session_id
        extracted_dir = session_dir / "extracted"
        outputs_dir = session_dir / "outputs"
        telemetry_dir = session_dir / "telemetry"
        
        # Create directories
        outputs_dir.mkdir(parents=True, exist_ok=True)
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        
        # Override settings for this session
        settings.EXTRACTED_DIR = extracted_dir
        settings.OUTPUTS_DIR = outputs_dir
        settings.TELEMETRY_DIR = telemetry_dir
        
        print(f"[TELEMETRY] Using session directories for {session_id}")
        print(f"[TELEMETRY] EXTRACTED_DIR: {extracted_dir}")
        print(f"[TELEMETRY] OUTPUTS_DIR: {outputs_dir}")
        print(f"[TELEMETRY] TELEMETRY_DIR: {telemetry_dir}")
    
    # Load extraction log
    meta_path = settings.OUTPUTS_DIR / "extraction_log.json"
    
    print(f"[TELEMETRY] Looking for extraction log at: {meta_path}")
    
    # Check if extraction log exists
    if not meta_path.exists():
        print(f"ERROR: Extraction log not found at: {meta_path}")
        print("Available files in EXTRACTED_DIR:")
        if settings.EXTRACTED_DIR.exists():
            for f in settings.EXTRACTED_DIR.iterdir():
                print(f"  - {f.name}")
        else:
            print(f"  EXTRACTED_DIR does not exist: {settings.EXTRACTED_DIR}")
        exit(1)
    
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Check if the data has the expected structure
        if "frames" not in data:
            print("Extraction log does not contain 'frames' key")
            print("Available keys:", list(data.keys()))
            exit(1)
        
        frame_meta = data["frames"]
        
        # Verify frame metadata structure
        if frame_meta:
            # Check for either 'timestamp_seconds' or 'timestamp' key
            if "timestamp_seconds" not in frame_meta[0] and "timestamp" not in frame_meta[0]:
                print("Frame metadata does not contain 'timestamp_seconds' or 'timestamp' key")
                print("Available keys:", list(frame_meta[0].keys()))
                exit(1)
        
        print(f"Loaded {len(frame_meta)} frames from extraction log")
        generate_telemetry(frame_meta)
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse extraction log: {e}")
        exit(1)
    except Exception as e:
        print(f"Error loading extraction log: {e}")
        exit(1)
