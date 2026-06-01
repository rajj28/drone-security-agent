#!/usr/bin/env python3
"""
Local Pipeline Test Script
Tests the complete drone security pipeline locally to verify:
1. Frame extraction (with quality filtering)
2. Telemetry generation
3. Vision analysis (CLIP + BLIP + GPT-4o)
4. Alert generation
5. Context management

Usage:
    python test_pipeline.py --video path/to/video.mp4
"""

import argparse
import sys
import os
from pathlib import Path
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import settings
from src.intelligent_frame_extractor import IntelligentFrameExtractor
from src.telemetry_generator import generate_telemetry
from src.vision_analyzer import analyze_all_frames
from src.context_manager import load_context_summaries, load_session_context

def test_frame_extraction(video_path: str, max_frames: int = 10):
    """Test frame extraction with quality filtering"""
    print("\n" + "="*60)
    print("🎬 STEP 1: FRAME EXTRACTION")
    print("="*60)
    
    extractor = IntelligentFrameExtractor()
    video_file = Path(video_path)
    
    # Create test session directory
    session_id = f"test_{int(time.time())}"
    session_dir = Path("data/test_sessions") / session_id
    extracted_dir = session_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    
    # Override settings for test
    settings.VIDEO_FILE = video_file
    settings.EXTRACTED_DIR = extracted_dir
    settings.OUTPUTS_DIR = session_dir / "outputs"
    settings.OUTPUTS_DIR.mkdir(exist_ok=True)
    
    print(f"Video: {video_path}")
    print(f"Session: {session_id}")
    print(f"Output: {extracted_dir}")
    
    # Extract frames
    start = time.time()
    frames = extractor.extract_hybrid(video_file, str(extracted_dir), max_frames=max_frames)
    elapsed = time.time() - start
    
    print(f"\n✅ Extracted {len(frames)} frames in {elapsed:.1f}s")
    print(f"   Frame files: {[f.filename for f in frames]}")
    
    # Check extraction log
    log_path = settings.OUTPUTS_DIR / "extraction_log.json"
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
        print(f"   Total frames in log: {len(log.get('frames', []))}")
    
    return session_id, frames

def test_telemetry_generation(session_id: str, frames: list):
    """Test telemetry generation"""
    print("\n" + "="*60)
    print("📊 STEP 2: TELEMETRY GENERATION")
    print("="*60)
    
    session_dir = Path("data/test_sessions") / session_id
    settings.EXTRACTED_DIR = session_dir / "extracted"
    settings.OUTPUTS_DIR = session_dir / "outputs"
    settings.TELEMETRY_DIR = session_dir / "telemetry"
    settings.TELEMETRY_DIR.mkdir(exist_ok=True)
    
    # Generate telemetry
    start = time.time()
    generate_telemetry()
    elapsed = time.time() - start
    
    # Check output
    telemetry_file = settings.TELEMETRY_DIR / "all_telemetry.json"
    if telemetry_file.exists():
        with open(telemetry_file) as f:
            telemetry = json.load(f)
        print(f"✅ Generated telemetry for {len(telemetry)} frames in {elapsed:.1f}s")
        
        # Show sample
        if telemetry:
            sample = telemetry[0]
            print(f"\n   Sample (frame {sample.get('frame_id', 'N/A')}):")
            print(f"   - People: {sample.get('people_count', 'N/A')}")
            print(f"   - Motion: {sample.get('motion_detected', 'N/A')}")
            print(f"   - Location: {sample.get('location', 'N/A')}")
    else:
        print("❌ Telemetry generation failed!")
        return None
    
    return telemetry

def test_vision_analysis(session_id: str, frames: list):
    """Test vision analysis"""
    print("\n" + "="*60)
    print("👁️  STEP 3: VISION ANALYSIS (CLIP + BLIP + GPT-4o)")
    print("="*60)
    
    session_dir = Path("data/test_sessions") / session_id
    
    # Set up all directories
    settings.EXTRACTED_DIR = session_dir / "extracted"
    settings.OUTPUTS_DIR = session_dir / "outputs"
    settings.TELEMETRY_DIR = session_dir / "telemetry"
    settings.ANALYSIS_DIR = session_dir / "analysis"
    settings.ANALYSIS_DIR.mkdir(exist_ok=True)
    
    # Run analysis
    start = time.time()
    try:
        results = analyze_all_frames()
        elapsed = time.time() - start
        
        print(f"✅ Analyzed {len(results)} frames in {elapsed:.1f}s")
        
        # Check for alerts
        alerts = [r for r in results if r and r.get('threat_level') in ['CRITICAL', 'HIGH']]
        print(f"   🚨 Critical/High alerts: {len(alerts)}")
        
        # Show sample results
        for i, result in enumerate(results[:3]):
            if result:
                print(f"\n   Frame {result.get('frame_id', i+1)}:")
                print(f"   - Threat: {result.get('threat_level', 'N/A')}")
                print(f"   - People: {result.get('people_count', 'N/A')}")
                print(f"   - Scene: {result.get('scene_type', 'N/A')}")
                print(f"   - Model: {result.get('model_used', 'N/A')}")
                
                # Show alert if any
                if result.get('alert'):
                    alert = result['alert']
                    print(f"   - ALERT: {alert.get('severity', 'N/A')} - {alert.get('reason', 'N/A')[:50]}...")
        
        return results
        
    except Exception as e:
        print(f"❌ Vision analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_context_summaries(session_id: str):
    """Check if context summaries were saved"""
    print("\n" + "="*60)
    print("🧠 STEP 4: CONTEXT MANAGEMENT")
    print("="*60)
    
    # Check if MongoDB context exists
    try:
        context_store = load_context_summaries()
        if context_store:
            print(f"✅ Context summaries loaded: {len(context_store)} entries")
            # Show latest
            if context_store:
                latest = list(context_store.values())[-1] if isinstance(context_store, dict) else context_store[-1]
                print(f"   Latest summary: {latest.get('situation_summary', 'N/A')[:100]}...")
        else:
            print("⚠️  No context summaries found (MongoDB may not be connected)")
    except Exception as e:
        print(f"⚠️  Context loading issue: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test the drone security pipeline locally")
    parser.add_argument("--video", required=True, help="Path to test video file")
    parser.add_argument("--max-frames", type=int, default=10, help="Max frames to extract (default: 10)")
    parser.add_argument("--skip-telemetry", action="store_true", help="Skip telemetry step")
    parser.add_argument("--skip-vision", action="store_true", help="Skip vision analysis step")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚁 DRONE SECURITY PIPELINE - LOCAL TEST")
    print("="*60)
    print(f"Testing with video: {args.video}")
    print(f"Max frames: {args.max_frames}")
    
    # Step 1: Frame Extraction
    session_id, frames = test_frame_extraction(args.video, args.max_frames)
    if not frames:
        print("\n❌ Frame extraction failed - aborting")
        return 1
    
    # Step 2: Telemetry Generation
    if not args.skip_telemetry:
        telemetry = test_telemetry_generation(session_id, frames)
        if not telemetry:
            print("\n❌ Telemetry generation failed")
    
    # Step 3: Vision Analysis
    if not args.skip_vision:
        results = test_vision_analysis(session_id, frames)
    
    # Step 4: Context Check
    check_context_summaries(session_id)
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    print(f"Session ID: {session_id}")
    print(f"Session Dir: data/test_sessions/{session_id}")
    print(f"\nOutputs:")
    print(f"  - Frames: data/test_sessions/{session_id}/extracted/")
    print(f"  - Telemetry: data/test_sessions/{session_id}/telemetry/all_telemetry.json")
    print(f"  - Analysis: data/test_sessions/{session_id}/analysis/all_analysis.json")
    print("\n✅ Pipeline test complete!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
