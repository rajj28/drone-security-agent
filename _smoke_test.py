"""Temporary smoke test: upload, poll to completion, then print alerts + summary stats."""
import time
import json
import requests
from pathlib import Path

API = "http://localhost:8000"
def main():
    video = next(Path(".").glob("*.mp4"))
    print(f"Uploading: {video.name}")

    with open(video, "rb") as f:
        r = requests.post(
            f"{API}/upload-video",
            files={"file": (video.name, f, "video/mp4")},
            params={"max_frames": 10, "extraction_strategy": "uniform"},
            timeout=120,
        )
    r.raise_for_status()
    session_id = r.json()["session_id"]
    print(f"Session: {session_id}")

    start = time.time()
    last = None
    while True:
        s = requests.get(f"{API}/processing-status/{session_id}", timeout=30).json()
        key = (s.get("current_step"), s.get("progress"), s.get("status"))
        if key != last:
            print(f"  [{int(time.time()-start)}s] step={key[0]} progress={key[1]} status={key[2]}")
            last = key
        if s.get("status") in ("completed", "failed"):
            print(f"FINAL: status={s.get('status')} total={int(time.time()-start)}s")
            if s.get("status") == "failed":
                print("ERROR:", str(s.get("error"))[:400])
            break
        if time.time() - start > 600:
            print(f"TIMEOUT last_step={key[0]}")
            break
        time.sleep(4)

    # Inspect outputs
    d = Path("data/sessions") / session_id
    try:
        al = json.load(open(d / "alerts/all_alerts.json", encoding="utf-8"))
        print(f"\nALERTS: total={al.get('total_alerts')} high={al.get('high_severity')} med={al.get('medium_severity')} low={al.get('low_severity')}")
        for a in al.get("alerts", []):
            if a.get("alert_triggered"):
                print(f"  - {a.get('frame_id')}: {a.get('severity')} {a.get('alert_type')} :: {str(a.get('llm_reasoning',''))[:120]}")
    except Exception as e:
        print("alerts read error:", e)
    try:
        s = json.load(open(d / "session_summary.json", encoding="utf-8"))
        ss = s.get("session_summary", {})
        print(f"\nSUMMARY: frames={ss.get('total_frames_analyzed')} people={ss.get('people_count')} alerts={ss.get('total_alerts')}")
        print("one_line:", s.get("one_line_summary"))
    except Exception as e:
        print("summary read error:", e)

if __name__ == "__main__":
    main()
