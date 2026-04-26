"""
telemetry_generator.py — Generates realistic simulated telemetry for each extracted frame.

- Cycles through locations
- Updates battery, after_hours, restricted_zone flags
- Saves per-frame and combined telemetry JSONs
"""

import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta
from src.config import settings


def generate_telemetry(
    frame_metadata: List[Dict],
    start_unix: int = settings.VIDEO_START_UNIX,
    output_dir: Path = settings.TELEMETRY_DIR
) -> List[Dict]:
    """
    Generates telemetry for each frame and saves per-frame JSON.
    Returns list of all telemetry dicts.
    """
    print("\n📡 Generating telemetry for frames...")
    locations = settings.LOCATIONS
    battery = 100.0
    battery_drop = 100.0 / max(1, len(frame_metadata))
    all_telemetry = []
    for i, frame in enumerate(frame_metadata):
        frame_id = f"frame_{i+1:03}"
        ts_seconds = frame["timestamp_seconds"]
        unix_time = start_unix + ts_seconds
        dt = datetime.utcfromtimestamp(unix_time)
        timestamp = dt.strftime("%H:%M:%S")
        location = locations[i % len(locations)]
        zone = "restricted" if location in ["Back Entrance", "Restricted Zone"] else "public"
        after_hours = dt.hour < 6 or dt.hour >= 22
        telemetry = {
            "frame_id": frame_id,
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
        print(f"✅ Telemetry for {frame_id} at {timestamp} ({location})")
    # Save combined
    combined_path = output_dir / "all_telemetry.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_telemetry, f, indent=2)
    print(f"\n📄 Combined telemetry saved to {combined_path}")
    return all_telemetry

if __name__ == "__main__":
    # Load extraction log
    meta_path = settings.OUTPUTS_DIR / "extraction_log.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        frame_meta = json.load(f)["frames"]
    generate_telemetry(frame_meta)
