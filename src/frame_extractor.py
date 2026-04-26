"""
frame_extractor.py — Extracts frames from a .dav CCTV video file using ffmpeg.

- Extracts exactly 25 frames evenly across the video.
- Saves frames as JPGs in data/extracted/
- Logs extraction metadata to outputs/extraction_log.json
"""

import subprocess
import json
import time
from pathlib import Path
from typing import List, Dict
from PIL import Image
from src.config import settings


def extract_frames(
    video_path: Path = settings.VIDEO_FILE,
    output_dir: Path = settings.EXTRACTED_DIR,
    num_frames: int = settings.MAX_FRAMES,
    duration: int = settings.VIDEO_DURATION_SECONDS,
    fps: int = settings.VIDEO_FPS,
    resolution: str = "1920x1080",
    codec: str = "hevc"
) -> List[Dict]:
    """
    Extracts frames from the video using ffmpeg and saves them as JPGs.
    Returns a list of frame metadata dicts.
    """
    print("\n🎬 Starting frame extraction...")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_interval = duration // num_frames
    frame_metadata = []
    for i in range(num_frames):
        timestamp = i * frame_interval
        hh = timestamp // 3600
        mm = (timestamp % 3600) // 60
        ss = timestamp % 60
        ts_formatted = f"{hh:02}:{mm:02}:{ss:02}"
        frame_name = f"frame_{i+1:03}.jpg"
        output_path = output_dir / frame_name
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-ss", ts_formatted,
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path)
        ]
        try:
            start = time.time()
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elapsed = int((time.time() - start) * 1000)
            # Validate image
            with Image.open(output_path) as img:
                img.verify()
            file_size_kb = round(output_path.stat().st_size / 1024, 1)
            frame_metadata.append({
                "frame_number": i+1,
                "filename": frame_name,
                "timestamp_seconds": timestamp,
                "timestamp_formatted": ts_formatted,
                "file_size_kb": file_size_kb,
                "extraction_time_ms": elapsed
            })
            print(f"✅ Extracted {frame_name} at {ts_formatted} ({file_size_kb} KB)")
        except Exception as e:
            print(f"❌ Failed to extract {frame_name} at {ts_formatted}: {e}")
    return frame_metadata

def save_extraction_log(
    frame_metadata: List[Dict],
    output_path: Path = settings.OUTPUTS_DIR / "extraction_log.json",
    duration: int = settings.VIDEO_DURATION_SECONDS,
    fps: int = settings.VIDEO_FPS,
    resolution: str = "1920x1080",
    codec: str = "hevc"
):
    """
    Saves extraction metadata to a JSON file.
    """
    log = {
        "total_frames": len(frame_metadata),
        "video_duration_seconds": duration,
        "fps": fps,
        "resolution": resolution,
        "codec": codec,
        "frames": frame_metadata
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"\n📄 Extraction log saved to {output_path}")

if __name__ == "__main__":
    meta = extract_frames()
    save_extraction_log(meta)
