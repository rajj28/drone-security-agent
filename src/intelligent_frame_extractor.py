"""
intelligent_frame_extractor.py — Production-grade intelligent frame extraction system.

- Motion-based frame extraction to capture critical events
- Scene change detection for important moments
- Adaptive frame rate based on activity levels
- Multi-strategy extraction (uniform + event-driven)
- Production-ready error handling and logging
"""

import subprocess
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import cv2
from dataclasses import dataclass
from enum import Enum

from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractionStrategy(Enum):
    """Frame extraction strategies."""
    UNIFORM = "uniform"
    MOTION_BASED = "motion_based"
    SCENE_CHANGE = "scene_change"
    HYBRID = "hybrid"

@dataclass
class FrameInfo:
    """Frame information structure."""
    frame_number: int
    timestamp: float
    filename: str
    file_size_kb: float
    extraction_time_ms: float
    motion_score: float = 0.0
    scene_change_score: float = 0.0
    importance_score: float = 0.0
    extraction_reason: str = "uniform"

class IntelligentFrameExtractor:
    """Production-grade intelligent frame extractor."""
    
    def __init__(self):
        self.motion_threshold = 0.15  # Threshold for motion detection
        self.scene_change_threshold = 0.3  # Threshold for scene change detection
        self.min_frame_interval = 0.5  # Minimum seconds between frames
        self.max_frames_per_minute = 30  # Maximum frames to extract per minute
        
    def extract_frames_intelligently(
        self,
        video_path: Path,
        output_dir: Path,
        strategy: ExtractionStrategy = ExtractionStrategy.HYBRID,
        max_total_frames: int = 100,
        target_fps: float = 2.0
    ) -> List[FrameInfo]:
        """
        Extract frames using intelligent strategies.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save frames
            strategy: Extraction strategy to use
            max_total_frames: Maximum total frames to extract
            target_fps: Target frames per second for uniform extraction
            
        Returns:
            List of frame information
        """
        logger.info(f"Starting intelligent frame extraction with strategy: {strategy.value}")
        
        # Get video information
        video_info = self._get_video_info(video_path)
        if not video_info:
            raise ValueError(f"Could not read video information from {video_path}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Choose extraction strategy
        if strategy == ExtractionStrategy.UNIFORM:
            frames = self._extract_uniform_frames(video_path, output_dir, video_info, target_fps)
        elif strategy == ExtractionStrategy.MOTION_BASED:
            frames = self._extract_motion_frames(video_path, output_dir, video_info, max_total_frames)
        elif strategy == ExtractionStrategy.SCENE_CHANGE:
            frames = self._extract_scene_change_frames(video_path, output_dir, video_info, max_total_frames)
        elif strategy == ExtractionStrategy.HYBRID:
            frames = self._extract_hybrid_frames(video_path, output_dir, video_info, max_total_frames, target_fps)
        else:
            raise ValueError(f"Unknown extraction strategy: {strategy}")
        
        # Sort frames by timestamp
        frames.sort(key=lambda x: x.timestamp)
        
        # Renumber frames sequentially and rename from temp files
        for i, frame in enumerate(frames):
            # Get the temp filename that was actually saved to disk
            temp_filename = getattr(frame, 'temp_filename', frame.filename)
            old_path = output_dir / temp_filename
            new_filename = f"frame_{i+1:03d}.jpg"
            new_path = output_dir / new_filename
            
            if old_path.exists():
                old_path.rename(new_path)
                frame.filename = new_filename
                frame.frame_number = i + 1
                frame.temp_filename = new_filename  # Update temp reference
        
        logger.info(f"Extracted {len(frames)} frames using {strategy.value} strategy")
        return frames
    
    def _get_video_info(self, video_path: Path) -> Optional[Dict]:
        """Get video information using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Extract relevant information
            video_stream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
            if not video_stream:
                return None
            
            return {
                "duration": float(info["format"]["duration"]),
                "fps": eval(video_stream["r_frame_rate"]),
                "width": video_stream["width"],
                "height": video_stream["height"],
                "codec": video_stream["codec_name"]
            }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return None
    
    def _extract_uniform_frames(
        self, 
        video_path: Path, 
        output_dir: Path, 
        video_info: Dict, 
        target_fps: float
    ) -> List[FrameInfo]:
        """Extract frames at uniform intervals."""
        duration = video_info["duration"]
        interval = 1.0 / target_fps
        num_frames = int(duration / interval)
        
        frames = []
        for i in range(min(num_frames, 100)):  # Cap at 100 frames
            timestamp = i * interval
            frame = self._extract_frame_at_timestamp(
                video_path, output_dir, timestamp, i + 1, "uniform"
            )
            if frame:
                frames.append(frame)
        
        return frames
    
    def _extract_motion_frames(
        self, 
        video_path: Path, 
        output_dir: Path, 
        video_info: Dict, 
        max_frames: int
    ) -> List[FrameInfo]:
        """Extract frames based on motion detection."""
        logger.info("Analyzing video for motion-based extraction...")
        
        # Open video for motion analysis
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = video_info["fps"]
        duration = video_info["duration"]
        frame_count = int(duration * fps)
        
        # Initialize motion detection
        prev_gray = None
        motion_events = []
        frame_number = 0
        
        logger.info(f"Analyzing {frame_count} frames for motion...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every 10th frame for efficiency
            if frame_number % 10 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    # Calculate motion using optical flow
                    flow = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, 
                        np.array([[100, 100]], dtype=np.float32).reshape(-1, 1, 2),
                        None
                    )[0]
                    
                    if flow is not None and len(flow) > 0:
                        motion_magnitude = np.linalg.norm(flow)
                        timestamp = frame_number / fps
                        
                        if motion_magnitude > self.motion_threshold:
                            motion_events.append({
                                "timestamp": timestamp,
                                "motion_score": float(motion_magnitude),
                                "frame_number": frame_number
                            })
                
                prev_gray = gray
            
            frame_number += 1
            
            # Progress update
            if frame_number % 1000 == 0:
                progress = (frame_number / frame_count) * 100
                logger.info(f"Motion analysis progress: {progress:.1f}%")
        
        cap.release()
        
        # Sort motion events by score and select top events
        motion_events.sort(key=lambda x: x["motion_score"], reverse=True)
        selected_events = motion_events[:max_frames]
        
        # Extract frames at motion events
        frames = []
        for i, event in enumerate(selected_events):
            frame = self._extract_frame_at_timestamp(
                video_path, output_dir, event["timestamp"], i + 1, 
                f"motion_score_{event['motion_score']:.2f}"
            )
            if frame:
                frame.motion_score = event["motion_score"]
                frame.importance_score = event["motion_score"]
                frames.append(frame)
        
        logger.info(f"Found {len(motion_events)} motion events, extracted top {len(frames)}")
        return frames
    
    def _extract_scene_change_frames(
        self, 
        video_path: Path, 
        output_dir: Path, 
        video_info: Dict, 
        max_frames: int
    ) -> List[FrameInfo]:
        """Extract frames at scene changes."""
        logger.info("Analyzing video for scene changes...")
        
        # Use ffmpeg to detect scene changes
        scene_change_cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", "select='gt(scene,0.3)',showinfo",
            "-f", "null", "-"
        ]
        
        try:
            result = subprocess.run(scene_change_cmd, capture_output=True, text=True)
            scene_changes = self._parse_scene_changes(result.stderr)
            
            # Limit number of scene changes
            selected_changes = scene_changes[:max_frames]
            
            # Extract frames at scene changes
            frames = []
            for i, change in enumerate(selected_changes):
                frame = self._extract_frame_at_timestamp(
                    video_path, output_dir, change["timestamp"], i + 1,
                    f"scene_change_{change['score']:.2f}"
                )
                if frame:
                    frame.scene_change_score = change["score"]
                    frame.importance_score = change["score"]
                    frames.append(frame)
            
            logger.info(f"Found {len(scene_changes)} scene changes, extracted {len(frames)}")
            return frames
            
        except Exception as e:
            logger.error(f"Scene change detection failed: {e}")
            # Fallback to uniform extraction
            return self._extract_uniform_frames(video_path, output_dir, video_info, 1.0)
    
    def _extract_hybrid_frames(
        self, 
        video_path: Path, 
        output_dir: Path, 
        video_info: Dict, 
        max_frames: int, 
        target_fps: float
    ) -> List[FrameInfo]:
        """Hybrid extraction combining uniform, motion, and scene change detection."""
        logger.info("Using hybrid extraction strategy...")
        
        # Allocate frame budget
        uniform_budget = max_frames // 3
        motion_budget = max_frames // 3
        scene_budget = max_frames - uniform_budget - motion_budget
        
        frames = []
        
        # 1. Extract uniform frames (baseline)
        uniform_frames = self._extract_uniform_frames(
            video_path, output_dir, video_info, target_fps * 0.5
        )
        frames.extend(uniform_frames[:uniform_budget])
        
        # 2. Extract motion-based frames
        motion_frames = self._extract_motion_frames(
            video_path, output_dir, video_info, motion_budget
        )
        frames.extend(motion_frames)
        
        # 3. Extract scene change frames
        scene_frames = self._extract_scene_change_frames(
            video_path, output_dir, video_info, scene_budget
        )
        frames.extend(scene_frames)
        
        # Remove duplicates (frames too close in time)
        frames = self._deduplicate_frames(frames, min_interval=self.min_frame_interval)
        
        # Limit total frames
        if len(frames) > max_frames:
            # Sort by importance score and keep top frames
            frames.sort(key=lambda x: x.importance_score, reverse=True)
            frames = frames[:max_frames]
            frames.sort(key=lambda x: x.timestamp)  # Re-sort by time
        
        logger.info(f"Hybrid extraction complete: {len(frames)} frames")
        return frames
    
    def _extract_frame_at_timestamp(
        self, 
        video_path: Path, 
        output_dir: Path, 
        timestamp: float, 
        frame_number: int,
        reason: str
    ) -> Optional[FrameInfo]:
        """Extract a single frame at a specific timestamp."""
        # Use unique temp filename to avoid overwriting during hybrid extraction
        import uuid
        temp_name = f"temp_{uuid.uuid4().hex[:8]}.jpg"
        output_path = output_dir / temp_name
        # Final name will be assigned after deduplication (stored in frame_info temporarily)
        final_frame_name = f"frame_{frame_number:03d}.jpg"
        
        # Format timestamp for ffmpeg
        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = timestamp % 60
        ts_formatted = f"{hours:02}:{minutes:02}:{seconds:06.3f}"
        
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-ss", ts_formatted,
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",  # High quality
            "-vf", "scale=1920:1080",  # Standardize resolution
            str(output_path)
        ]
        
        try:
            start_time = time.time()
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Validate the extracted frame
            if output_path.exists():
                with Image.open(output_path) as img:
                    img.verify()
                
                file_size_kb = output_path.stat().st_size / 1024
                
                frame_info = FrameInfo(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    filename=final_frame_name,  # Will be updated after deduplication
                    file_size_kb=file_size_kb,
                    extraction_time_ms=elapsed_ms,
                    extraction_reason=reason,
                    importance_score=self._calculate_importance_score(reason)
                )
                # Store temp filename for renaming after deduplication
                frame_info.temp_filename = temp_name
                
                logger.debug(f"Extracted frame at {ts_formatted} ({reason})")
                return frame_info
            else:
                logger.warning(f"Frame not extracted at {ts_formatted}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract frame at {ts_formatted}: {e}")
            return None
    
    def _deduplicate_frames(self, frames: List[FrameInfo], min_interval: float) -> List[FrameInfo]:
        """Remove frames that are too close in time."""
        if not frames:
            return frames
        
        # Sort by timestamp
        frames.sort(key=lambda x: x.timestamp)
        
        deduplicated = [frames[0]]
        for frame in frames[1:]:
            if frame.timestamp - deduplicated[-1].timestamp >= min_interval:
                deduplicated.append(frame)
        
        logger.info(f"Deduplicated {len(frames)} frames to {len(deduplicated)} frames")
        return deduplicated
    
    def _calculate_importance_score(self, reason: str) -> float:
        """Calculate importance score based on extraction reason."""
        if "motion" in reason.lower():
            return 0.8
        elif "scene_change" in reason.lower():
            return 0.9
        elif "uniform" in reason.lower():
            return 0.3
        else:
            return 0.5
    
    def _parse_scene_changes(self, ffmpeg_output: str) -> List[Dict]:
        """Parse scene change information from ffmpeg output."""
        scene_changes = []
        for line in ffmpeg_output.split('\n'):
            if 'scene:' in line and 'score:' in line:
                try:
                    # Extract timestamp and score
                    parts = line.split()
                    time_part = [p for p in parts if p.startswith('pts_time:')]
                    score_part = [p for p in parts if p.startswith('score:')]
                    
                    if time_part and score_part:
                        timestamp = float(time_part[0].split(':')[1])
                        score = float(score_part[0].split(':')[1])
                        
                        scene_changes.append({
                            "timestamp": timestamp,
                            "score": score
                        })
                except Exception as e:
                    logger.debug(f"Could not parse scene change line: {line}")
        
        return scene_changes
    
    def save_extraction_log(
        self, 
        frames: List[FrameInfo], 
        video_path: Path, 
        strategy: ExtractionStrategy,
        output_path: Path = settings.OUTPUTS_DIR / "extraction_log.json"
    ):
        """Save detailed extraction log."""
        log = {
            "extraction_strategy": strategy.value,
            "video_path": str(video_path),
            "total_frames_extracted": len(frames),
            "extraction_timestamp": time.time(),
            "frames": [
                {
                    "frame_number": f.frame_number,
                    "filename": f.filename,
                    "timestamp": f.timestamp,
                    "file_size_kb": f.file_size_kb,
                    "extraction_time_ms": f.extraction_time_ms,
                    "motion_score": f.motion_score,
                    "scene_change_score": f.scene_change_score,
                    "importance_score": f.importance_score,
                    "extraction_reason": f.extraction_reason
                }
                for f in frames
            ],
            "statistics": {
                "avg_file_size_kb": sum(f.file_size_kb for f in frames) / len(frames) if frames else 0,
                "avg_extraction_time_ms": sum(f.extraction_time_ms for f in frames) / len(frames) if frames else 0,
                "avg_importance_score": sum(f.importance_score for f in frames) / len(frames) if frames else 0,
                "motion_frames": len([f for f in frames if f.motion_score > 0]),
                "scene_change_frames": len([f for f in frames if f.scene_change_score > 0])
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        
        logger.info(f"Extraction log saved to {output_path}")

# Global extractor instance
extractor = IntelligentFrameExtractor()

def extract_frames_intelligently(
    video_path: Path = settings.VIDEO_FILE,
    output_dir: Path = settings.EXTRACTED_DIR,
    strategy: str = "hybrid",
    max_frames: int = 100
) -> List[Dict]:
    """Main function for intelligent frame extraction."""
    try:
        strategy_enum = ExtractionStrategy(strategy)
        frames = extractor.extract_frames_intelligently(
            video_path, output_dir, strategy_enum, max_frames
        )
        extractor.save_extraction_log(frames, video_path, strategy_enum)
        
        # Convert to dict format for compatibility
        return [
            {
                "frame_number": f.frame_number,
                "filename": f.filename,
                "timestamp_seconds": f.timestamp,
                "timestamp_formatted": f"{int(f.timestamp//3600):02}:{int((f.timestamp%3600)//60):02}:{f.timestamp%60:06.3f}",
                "file_size_kb": f.file_size_kb,
                "extraction_time_ms": f.extraction_time_ms,
                "motion_score": f.motion_score,
                "scene_change_score": f.scene_change_score,
                "importance_score": f.importance_score,
                "extraction_reason": f.extraction_reason
            }
            for f in frames
        ]
    except Exception as e:
        logger.error(f"Intelligent frame extraction failed: {e}")
        raise

if __name__ == "__main__":
    # Test the intelligent extractor
    frames = extract_frames_intelligently()
    print(f"Extracted {len(frames)} frames intelligently")
