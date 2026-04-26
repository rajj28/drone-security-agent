import subprocess
import os
import sys
from pathlib import Path

MP4_FILE = r"data/frames/XVR_ch1_main_20230201160000_20230201170000(1).mp4"
OUTPUT_DIR = "data/frames/extracted"

# Extract 200 frames, each 10 minutes apart (0, 10, 20, ..., 1990 minutes)
TIMESTAMPS = {}
for i in range(200):
    seconds = i * 10 * 60
    frame_name = f"frame_{i+1:03d}"
    TIMESTAMPS[frame_name] = seconds

def extract_frames_at_timestamps():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("🚁 Drone Security Agent — Frame Extractor")
    print("=" * 45)
    print(f"📹 Extracting {len(TIMESTAMPS)} frames spread across 1 hour\n")

    extracted = []


    for frame_name, timestamp in TIMESTAMPS.items():
        output_path = f"{OUTPUT_DIR}/{frame_name}.jpg"
        mins = timestamp // 60
        secs = timestamp % 60

        cmd = [
            "ffmpeg",
            "-ss", str(timestamp),      # seek to timestamp
            "-i", MP4_FILE,
            "-vframes", "1",            # exactly 1 frame
            "-q:v", "2",               # high quality
            "-y",                       # overwrite if exists
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and Path(output_path).exists():
            size_kb = Path(output_path).stat().st_size / 1024
            print(f"   ✅ {frame_name}.jpg @ {mins:02d}:{secs:02d} ({size_kb:.1f} KB)")
            extracted.append(output_path)
        else:
            print(f"   ❌ Failed at {mins:02d}:{secs:02d} — {result.stderr[:50]}")

    print(f"\n🎯 Done! {len(extracted)}/{len(TIMESTAMPS)} frames extracted")
    print(f"📁 Saved to: {OUTPUT_DIR}")
    return extracted

if __name__ == "__main__":
    if not Path(MP4_FILE).exists():
        print(f"❌ File not found: {MP4_FILE}")
        sys.exit(1)

    extract_frames_at_timestamps()