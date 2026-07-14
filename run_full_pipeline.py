#!/usr/bin/env python3
"""
Full drone security pipeline — one video, one session, sequential context build.

Steps: extract → telemetry → vision → agent Q&A → [optional Pinecone] → health report

Usage:
  .\\venv\\Scripts\\Activate.ps1
  python run_full_pipeline.py --video "Sneaky Thieves....mp4" --session-id theft_newsflare_001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))


def _set_env(session_id: str, use_mongo: bool, offline: bool, standard: bool) -> None:
    os.environ["SESSION_ID"] = session_id
    os.environ.setdefault("GEMINI_PREFER_FLASH", "true")
    os.environ.setdefault("GEMINI_FLASH_ONLY", "true")
    if not use_mongo:
        os.environ["USE_MONGO_CONTEXT"] = "false"
    if offline:
        os.environ["OFFLINE_VISION"] = "true"
        os.environ["USE_CLOUD_ANALYZER"] = "false"
        os.environ["SKIP_HF_APIS"] = "true"
    elif standard:
        os.environ["OFFLINE_VISION"] = "false"
        os.environ["USE_CLOUD_ANALYZER"] = "false"
        os.environ["SKIP_HF_APIS"] = "true"
    else:
        os.environ.setdefault("USE_CLOUD_ANALYZER", "false")
        os.environ.setdefault("GPT_SINGLE_STAGE", "true")
        os.environ.setdefault("SKIP_HF_APIS", "true")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full security pipeline with unified context")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--session-id", default="theft_newsflare_001", help="SESSION_ID for this run")
    parser.add_argument("--max-frames", type=int, default=12, help="Max frames to extract/analyze")
    parser.add_argument(
        "--vision-delay",
        type=float,
        default=0.0,
        help="Extra seconds between frames (API_MIN_INTERVAL_SEC is the main throttle)",
    )
    parser.add_argument(
        "--api-interval",
        type=float,
        default=10.0,
        help="Minimum seconds between OpenAI/HF calls (default 10)",
    )
    parser.add_argument(
        "--offline-vision",
        action="store_true",
        help="No OpenAI/HF — OpenCV heuristics only (use when quota/429)",
    )
    parser.add_argument(
        "--standard-vision",
        action="store_true",
        help="One GPT-4o call per frame only (no HF CLIP/BLIP)",
    )
    parser.add_argument(
        "--extraction-strategy",
        choices=["hybrid", "uniform", "motion"],
        default="uniform",
        help="uniform spreads frames across video (best for incident clips)",
    )
    parser.add_argument("--target-fps", type=float, default=0.25, help="Uniform extraction rate")
    parser.add_argument("--use-mongo", action="store_true", help="Enable USE_MONGO_CONTEXT (production)")
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Reuse existing frames in session extracted/ dir",
    )
    parser.add_argument("--skip-telemetry", action="store_true")
    parser.add_argument(
        "--with-pinecone",
        action="store_true",
        help="Index frame descriptions in Pinecone and run smoke search queries",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Video not found: {video_path}")
        return 1

    _set_env(args.session_id, args.use_mongo, args.offline_vision, args.standard_vision)
    os.environ["API_MIN_INTERVAL_SEC"] = str(args.api_interval)

    from src.api_retry import reset_api_run_state

    reset_api_run_state()

    # Import after env so bootstrap picks up SESSION_ID
    from src.session_bootstrap import apply_session_layout
    from src.config import settings
    from src.intelligent_frame_extractor import (
        IntelligentFrameExtractor,
        ExtractionStrategy,
    )
    from src.telemetry_generator import generate_telemetry
    from src.vision_analyzer import analyze_all_frames, derive_alert_from_analysis
    from src.unified_context import get_unified_context

    session_root = apply_session_layout(args.session_id)
    print("\n" + "=" * 60)
    print("DRONE SECURITY — FULL PIPELINE")
    print("=" * 60)
    print(f"Video:     {video_path.name}")
    print(f"Session:   {args.session_id}")
    print(f"Root:      {session_root}")
    print(f"Mongo:     {args.use_mongo}")
    print(f"Frames:    max {args.max_frames} ({args.extraction_strategy})")
    mode = "offline" if args.offline_vision else ("standard-gpt" if args.standard_vision else "cloud")
    print(f"Vision:    {mode} | API interval {args.api_interval}s")
    print("=" * 60)

    report: dict = {"session_id": args.session_id, "steps": {}}

    log_path = settings.OUTPUTS_DIR / "extraction_log.json"

    # --- 1. Extraction ---
    if args.skip_extraction and log_path.exists():
        with open(log_path, encoding="utf-8") as handle:
            frames_n = len(json.load(handle).get("frames", []))
        report["steps"]["extraction"] = {"skipped": True, "frames": frames_n}
        print(f"\n[1/5] Extraction: skipped ({frames_n} frames in log)")
        if frames_n == 0:
            print("No frames in extraction log — aborting.")
            return 1
    else:
        t0 = time.time()
        for old in settings.EXTRACTED_DIR.glob("*.jpg"):
            old.unlink(missing_ok=True)
        extractor = IntelligentFrameExtractor()
        strategy = ExtractionStrategy(args.extraction_strategy)
        frames = extractor.extract_frames_intelligently(
            video_path,
            settings.EXTRACTED_DIR,
            strategy=strategy,
            max_total_frames=args.max_frames,
            target_fps=args.target_fps,
        )
        extractor.save_extraction_log(frames, video_path, strategy, output_path=log_path)
        report["steps"]["extraction"] = {
            "frames": len(frames),
            "elapsed_sec": round(time.time() - t0, 1),
            "files": [f.filename for f in frames],
        }
        print(f"\n[1/5] Extraction: {len(frames)} frames in {report['steps']['extraction']['elapsed_sec']}s")
        if not frames:
            print("No frames extracted — aborting.")
            return 1

    # --- 2. Telemetry ---
    if args.skip_telemetry and (settings.TELEMETRY_DIR / "all_telemetry.json").exists():
        report["steps"]["telemetry"] = {"skipped": True}
        print(f"[2/5] Telemetry: skipped ({settings.TELEMETRY_DIR / 'all_telemetry.json'})")
    else:
        t0 = time.time()
        with open(log_path, encoding="utf-8") as handle:
            frame_meta = json.load(handle)["frames"]
        generate_telemetry(frame_meta, output_dir=settings.TELEMETRY_DIR)
        report["steps"]["telemetry"] = {"elapsed_sec": round(time.time() - t0, 1)}
        print(f"[2/5] Telemetry: {settings.TELEMETRY_DIR / 'all_telemetry.json'}")

    # --- 3. Vision ---
    if not args.skip_vision:
        t0 = time.time()
        # Patch so vision uses session extracted dir, not data/extractedN
        import src.api as api_module

        api_module.get_latest_extracted_folder = lambda: settings.EXTRACTED_DIR

        if args.vision_delay > 0:
            import src.vision_analyzer as va

            _orig = va.analyze_frame

            def _delayed_analyze(*a, **kw):
                time.sleep(args.vision_delay)
                return _orig(*a, **kw)

            va.analyze_frame = _delayed_analyze

        results = analyze_all_frames()
        ok = [r for r in results if r]
        threats = {}
        for r in ok:
            tl = str(r.get("threat_level", r.get("threat_assessment", "UNKNOWN"))).upper()
            threats[tl] = threats.get(tl, 0) + 1
        report["steps"]["vision"] = {
            "elapsed_sec": round(time.time() - t0, 1),
            "analyzed": len(ok),
            "threat_distribution": threats,
        }
        print(f"[3/5] Vision: {len(ok)} frames, threats={threats}")
    else:
        print("[3/5] Vision: skipped")

    # --- 4. Agent ---
    if not args.skip_agent:
        from src.unified_context import get_recent_context_for_agent
        from src.api_retry import quota_exhausted

        questions = [
            "What suspicious activity happened in this session?",
            "How many people were visible?",
            "How many people were visible and were there any high-threat moments?",
            "Was there activity near a display case or counter?",
        ]
        answers = []
        # Allow non-Gemini agent providers (Groq/Ollama) to work even when Gemini vision quota exhausted
        agent_provider = os.environ.get("AGENT_LLM_PROVIDER", "gemini")
        agent_needs_gemini = agent_provider == "gemini"
        if args.offline_vision or (quota_exhausted() and agent_needs_gemini):
            recent = get_recent_context_for_agent(8)
            summary_text = "\n".join(
                s.get("context_summary", "") for s in recent if isinstance(s, dict)
            )
            for q in questions:
                answers.append(
                    {
                        "question": q,
                        "answer": (
                            "Answered from saved session context (no LLM — API quota/429):\n"
                            + (summary_text[:800] or "No summaries yet.")
                        ),
                    }
                )
        else:
            from src.agent import DroneSecurityAgent

            agent = DroneSecurityAgent()
            for q in questions:
                try:
                    ans = agent.answer_question(q)
                    answers.append({"question": q, "answer": ans.get("answer", str(ans))[:500]})
                except Exception as exc:
                    answers.append({"question": q, "error": str(exc)})
        report["steps"]["agent"] = {"answers": answers}
        print(f"[4/5] Agent: answered {len(answers)} questions")
    else:
        print("[4/5] Agent: skipped")

    # --- 5. Pinecone (integrated inference on flytbase) ---
    if args.with_pinecone:
        t0 = time.time()
        analysis_path = settings.ANALYSIS_DIR / "all_analysis.json"
        pinecone_step: dict = {"skipped": False}
        if not analysis_path.exists():
            pinecone_step = {"skipped": True, "reason": "no all_analysis.json"}
            print("[5/5] Pinecone: skipped (no analysis)")
        else:
            from src.pinecone_indexer import index_frames, search_frames

            try:
                index_log = index_frames()
                pinecone_step["indexed"] = index_log.get("total_indexed", 0)
                pinecone_step["namespace"] = index_log.get("namespace")
                queries = [
                    "person stealing phone or valuables",
                    "suspicious activity near cash register",
                ]
                pinecone_step["search"] = {}
                for q in queries:
                    sr = search_frames(q, top_k=3)
                    pinecone_step["search"][q] = {
                        "hits": len(sr.get("results", [])),
                        "top_score": sr["results"][0]["similarity_score"]
                        if sr.get("results")
                        else None,
                    }
                pinecone_step["elapsed_sec"] = round(time.time() - t0, 1)
                print(
                    f"[5/5] Pinecone: indexed {pinecone_step.get('indexed', 0)} "
                    f"in namespace {pinecone_step.get('namespace')}"
                )
            except Exception as exc:
                pinecone_step = {"error": str(exc), "elapsed_sec": round(time.time() - t0, 1)}
                print(f"[5/5] Pinecone: FAILED — {exc}")
        report["steps"]["pinecone"] = pinecone_step
    else:
        print("[5/5] Pinecone: skipped (use --with-pinecone)")

    # --- Health check ---
    unified = get_unified_context(session_dir=settings.SESSION_DIR)
    health = {
        "session_context.json": unified.session_context_path.exists(),
        "context_summaries.json": unified.summaries_path.exists(),
        "rich_context.json": (settings.SESSION_DIR / "rich_context.json").exists(),
        "summary_count": len(unified.load_summaries_store().get("summaries", [])),
        "frames_analyzed": unified.load_session_context().get("frames_analyzed", 0),
    }
    report["health"] = health

    out = settings.OUTPUTS_DIR / "pipeline_run_report.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n" + "=" * 60)
    print("CONTEXT HEALTH CHECK")
    print("=" * 60)
    for key, val in health.items():
        print(f"  {key}: {val}")
    print(f"\nReport: {out}")
    print("=" * 60)

    passed = (
        health["session_context.json"]
        and health["context_summaries.json"]
        and health["summary_count"] > 0
    )
    if args.with_pinecone:
        pc = report.get("steps", {}).get("pinecone", {})
        passed = passed and not pc.get("error") and pc.get("indexed", 0) > 0
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
