"""
video_folder_indexer.py - Video-specific folder indexing system

Each video gets its own folder with:
- Original video file
- Extracted frames
- Frame analysis results
- Video metadata
- Index files for fast retrieval

Structure:
data/videos/
├── video_001/
│   ├── video.mp4
│   ├── frames/
│   │   ├── frame_001.jpg
│   │   ├── frame_002.jpg
│   │   └── ...
│   ├── analysis/
│   │   ├── frame_001.json
│   │   ├── frame_002.json
│   │   └── ...
│   ├── metadata.json
│   ├── index.json
│   └── alerts.json
├── video_002/
│   └── ...
"""

import asyncio
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import uuid
import hashlib

from src.vision_analyzer import analyze_frame
from src.ai_learning_agent import ai_learning_agent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """Video metadata structure"""
    video_id: str
    filename: str
    file_path: str
    file_size: int
    duration: float
    fps: float
    resolution: Tuple[int, int]
    codec: str
    created_at: datetime
    location: str
    camera_id: Optional[str] = None
    tags: List[str] = None
    total_frames: int = 0
    processed_frames: int = 0
    threat_frames: int = 0
    indexing_status: str = "pending"  # pending, processing, completed, failed
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)

@dataclass
class FrameAnalysis:
    """Frame analysis result structure"""
    frame_id: str
    video_id: str
    timestamp: float
    frame_number: int
    analysis_result: Dict[str, Any]
    threat_level: str
    confidence: float
    people_detected: int
    objects_detected: List[str]
    created_at: datetime
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)

class VideoFolderIndexer:
    """Video-specific folder indexing system"""
    
    def __init__(self, base_path: str = "data/videos"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.frames_dir_name = "frames"
        self.analysis_dir_name = "analysis"
        
        # File names
        self.metadata_file = "metadata.json"
        self.index_file = "index.json"
        self.alerts_file = "alerts.json"
        self.summary_file = "summary.json"
        
        logger.info(f"Video folder indexer initialized at: {self.base_path}")
    
    def create_video_folder(self, video_metadata: VideoMetadata) -> Path:
        """Create folder structure for a new video"""
        video_folder = self.base_path / video_metadata.video_id
        
        try:
            # Create main video folder
            video_folder.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (video_folder / self.frames_dir_name).mkdir(exist_ok=True)
            (video_folder / self.analysis_dir_name).mkdir(exist_ok=True)
            
            # Save metadata
            self.save_video_metadata(video_metadata)
            
            # Create empty index file
            self.create_index_file(video_metadata.video_id)
            
            # Create empty alerts file
            self.create_alerts_file(video_metadata.video_id)
            
            logger.info(f"Created video folder: {video_folder}")
            return video_folder
            
        except Exception as e:
            logger.error(f"Failed to create video folder: {e}")
            raise
    
    def save_video_metadata(self, video_metadata: VideoMetadata) -> None:
        """Save video metadata to metadata.json"""
        metadata_file = self.base_path / video_metadata.video_id / self.metadata_file
        
        try:
            # Convert to dict and handle datetime serialization
            metadata_dict = asdict(video_metadata)
            metadata_dict['created_at'] = video_metadata.created_at.isoformat()
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
                
            logger.info(f"Saved metadata for video: {video_metadata.video_id}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def load_video_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """Load video metadata from metadata.json"""
        metadata_file = self.base_path / video_id / self.metadata_file
        
        try:
            if not metadata_file.exists():
                return None
                
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
                
            return VideoMetadata(**metadata_dict)
            
        except Exception as e:
            logger.error(f"Failed to load metadata for video {video_id}: {e}")
            return None
    
    def create_index_file(self, video_id: str) -> None:
        """Create empty index file for video"""
        index_file = self.base_path / video_id / self.index_file
        
        index_data = {
            "video_id": video_id,
            "frames": [],
            "total_frames": 0,
            "processed_frames": 0,
            "threat_frames": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def create_alerts_file(self, video_id: str) -> None:
        """Create empty alerts file for video"""
        alerts_file = self.base_path / video_id / self.alerts_file
        
        alerts_data = {
            "video_id": video_id,
            "alerts": [],
            "total_alerts": 0,
            "high_alerts": 0,
            "medium_alerts": 0,
            "low_alerts": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        with open(alerts_file, 'w') as f:
            json.dump(alerts_data, f, indent=2)
    
    def extract_frames(self, video_path: Path, video_id: str, 
                      interval_seconds: int = 1) -> List[Tuple[str, float, int]]:
        """Extract frames from video and save to video folder"""
        frames_folder = self.base_path / video_id / self.frames_dir_name
        frames_list = []
        
        try:
            # Use ffmpeg for frame extraction
            import subprocess
            
            # Calculate frame extraction interval
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', f'select=not(mod(n\\,{interval_seconds}))',
                '-vsync', 'vfr',
                '-q:v', '2',
                str(frames_folder / 'frame_%06d.jpg')
            ]
            
            logger.info(f"Extracting frames from {video_path}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Get extracted frames
                frame_files = sorted(frames_folder.glob('frame_*.jpg'))
                
                for frame_file in frame_files:
                    frame_number = int(frame_file.stem.split('_')[1])
                    timestamp = frame_number  # Will be updated with actual video timing
                    frame_id = f"{video_id}_frame_{frame_number:06d}"
                    frames_list.append((frame_id, timestamp, frame_number))
                
                logger.info(f"Extracted {len(frames_list)} frames")
                return frames_list
            else:
                logger.error(f"Frame extraction failed: {result.stderr}")
                return []
                
        except Exception as e:
            logger.error(f"Frame extraction error: {e}")
            return []
    
    async def process_video_frames(self, video_id: str, frames_list: List[Tuple[str, float, int]],
                                 telemetry_base: Dict[str, Any]) -> None:
        """Process frames and save analysis results"""
        video_folder = self.base_path / video_id
        analysis_folder = video_folder / self.analysis_dir_name
        
        # Load video metadata
        metadata = self.load_video_metadata(video_id)
        if not metadata:
            raise ValueError(f"Video metadata not found for {video_id}")
        
        # Update status
        metadata.indexing_status = "processing"
        self.save_video_metadata(metadata)
        
        processed_frames = 0
        threat_frames = 0
        frame_analyses = []
        
        logger.info(f"Processing {len(frames_list)} frames for video {video_id}")
        
        for frame_id, timestamp, frame_number in frames_list:
            try:
                # Get frame path
                frame_path = video_folder / self.frames_dir_name / f"frame_{frame_number:06d}.jpg"
                
                if not frame_path.exists():
                    logger.warning(f"Frame file not found: {frame_path}")
                    continue
                
                # Prepare telemetry
                telemetry = {
                    **telemetry_base,
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'frame_number': frame_number,
                    'video_id': video_id
                }
                
                # Analyze frame using configured analyzer (respects environment variables)
                result = analyze_frame(frame_id, frame_path, telemetry)
                
                # Apply learning agent for adaptive analysis
                # Handle both standard and enhanced analyzer outputs
                threat_level = result.get('overall_threat_level', result.get('threat_assessment', 'low'))
                confidence = result.get('confidence', 0.8)
                people_detected = result.get('people_count', 0)
                objects_detected = result.get('objects_detected', [])
                
                frame_data = {
                    'frame_id': frame_id,
                    'video_id': video_id,
                    'timestamp': timestamp,
                    'frame_number': frame_number,
                    'analysis_result': result,
                    'threat_level': threat_level,
                    'confidence': confidence,
                    'people_detected': people_detected,
                    'objects_detected': objects_detected
                }
                
                adaptive_result = ai_learning_agent.get_adaptive_analysis(video_id, frame_data)
                
                # Create frame analysis object with learning insights
                frame_analysis = FrameAnalysis(
                    frame_id=frame_id,
                    video_id=video_id,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    analysis_result=adaptive_result,
                    threat_level=threat_level,
                    confidence=confidence,
                    people_detected=people_detected,
                    objects_detected=objects_detected,
                    created_at=datetime.utcnow()
                )
                
                # Save frame analysis
                analysis_file = analysis_folder / f"{frame_id}.json"
                with open(analysis_file, 'w') as f:
                    analysis_dict = asdict(frame_analysis)
                    analysis_dict['created_at'] = frame_analysis.created_at.isoformat()
                    analysis_dict['analysis_result'] = result  # Keep full analysis
                    json.dump(analysis_dict, f, indent=2, default=str)
                
                frame_analyses.append({
                    'frame_id': frame_id,
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'threat_level': frame_analysis.threat_level,
                    'confidence': frame_analysis.confidence,
                    'people_detected': frame_analysis.people_detected,
                    'analysis_file': str(analysis_file.relative_to(self.base_path))
                })
                
                processed_frames += 1
                if frame_analysis.threat_level.lower() in ['high', 'medium']:
                    threat_frames += 1
                
                # Log progress
                if processed_frames % 10 == 0:
                    logger.info(f"Processed {processed_frames}/{len(frames_list)} frames")
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_id}: {e}")
                continue
        
        # Update video metadata
        metadata.processed_frames = processed_frames
        metadata.threat_frames = threat_frames
        metadata.total_frames = len(frames_list)
        metadata.indexing_status = "completed"
        self.save_video_metadata(metadata)
        
        # Update index file
        self.update_index_file(video_id, frame_analyses)
        
        # Generate alerts for threat frames
        await self.generate_video_alerts(video_id, frame_analyses)
        
        # Create summary
        self.create_video_summary(video_id)
        
        # Apply learning agent to the entire video
        video_metadata_dict = {
            'location': telemetry_base.get('location', 'unknown'),
            'camera_id': telemetry_base.get('camera_id', 'unknown'),
            'duration': metadata.duration,
            'fps': metadata.fps,
            'resolution': metadata.resolution
        }
        
        # Convert frame analyses to proper format for learning
        frame_analyses_for_learning = []
        for frame in frame_analyses:
            frame_analyses_for_learning.append({
                'frame_id': frame['frame_id'],
                'frame_number': frame['frame_number'],
                'timestamp': frame['timestamp'],
                'threat_level': frame['threat_level'],
                'confidence': frame['confidence'],
                'people_detected': frame['people_detected'],
                'analysis_result': frame.get('analysis_result', {})
            })
        
        # Learn from this video using AI agent
        ai_analysis = await ai_learning_agent.analyze_video_with_ai(
            video_id, frame_analyses_for_learning, video_metadata_dict
        )
        
        # Save AI learning knowledge
        logger.info(f"AI Analysis completed for {video_id}: adaptation_level={ai_analysis.get('adaptation_level', 0):.1f}%")
        
        logger.info(f"Completed processing video {video_id}: {processed_frames} frames, {threat_frames} threats")
        logger.info(f"AI Analysis completed: score={ai_analysis.get('learning_score', 0):.2f}, adaptation={ai_analysis.get('adaptation_level', 0)}%")
    
    def update_index_file(self, video_id: str, frame_analyses: List[Dict[str, Any]]) -> None:
        """Update index file with frame analyses"""
        index_file = self.base_path / video_id / self.index_file
        
        try:
            # Load existing index
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            # Update with new analyses
            index_data['frames'] = frame_analyses
            index_data['total_frames'] = len(frame_analyses)
            index_data['processed_frames'] = len(frame_analyses)
            index_data['threat_frames'] = len([f for f in frame_analyses if f['threat_level'].lower() in ['high', 'medium']])
            index_data['updated_at'] = datetime.utcnow().isoformat()
            
            # Save updated index
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update index file: {e}")
    
    async def generate_video_alerts(self, video_id: str, frame_analyses: List[Dict[str, Any]]) -> None:
        """Generate alerts for threat frames"""
        alerts_file = self.base_path / video_id / self.alerts_file
        
        try:
            # Load existing alerts
            with open(alerts_file, 'r') as f:
                alerts_data = json.load(f)
            
            # Generate new alerts for threat frames
            new_alerts = []
            for frame in frame_analyses:
                if frame['threat_level'].lower() in ['high', 'medium']:
                    alert = {
                        'alert_id': f"alert_{frame['frame_id']}_{int(datetime.utcnow().timestamp())}",
                        'frame_id': frame['frame_id'],
                        'frame_number': frame['frame_number'],
                        'timestamp': frame['timestamp'],
                        'threat_level': frame['threat_level'],
                        'confidence': frame['confidence'],
                        'people_detected': frame['people_detected'],
                        'title': f"Threat detected in {video_id}",
                        'description': f"Threat level {frame['threat_level']} detected at frame {frame['frame_number']}",
                        'status': 'open',
                        'created_at': datetime.utcnow().isoformat()
                    }
                    new_alerts.append(alert)
            
            # Update alerts data
            alerts_data['alerts'].extend(new_alerts)
            alerts_data['total_alerts'] = len(alerts_data['alerts'])
            alerts_data['high_alerts'] = len([a for a in alerts_data['alerts'] if a['threat_level'].lower() == 'high'])
            alerts_data['medium_alerts'] = len([a for a in alerts_data['alerts'] if a['threat_level'].lower() == 'medium'])
            alerts_data['low_alerts'] = len([a for a in alerts_data['alerts'] if a['threat_level'].lower() == 'low'])
            alerts_data['updated_at'] = datetime.utcnow().isoformat()
            
            # Save updated alerts
            with open(alerts_file, 'w') as f:
                json.dump(alerts_data, f, indent=2)
                
            logger.info(f"Generated {len(new_alerts)} alerts for video {video_id}")
            
        except Exception as e:
            logger.error(f"Failed to generate alerts: {e}")
    
    def create_video_summary(self, video_id: str) -> None:
        """Create summary file for video"""
        summary_file = self.base_path / video_id / self.summary_file
        
        try:
            # Load metadata, index, and alerts
            metadata = self.load_video_metadata(video_id)
            
            index_file = self.base_path / video_id / self.index_file
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            alerts_file = self.base_path / video_id / self.alerts_file
            with open(alerts_file, 'r') as f:
                alerts_data = json.load(f)
            
            # Create summary
            summary = {
                'video_id': video_id,
                'filename': metadata.filename,
                'location': metadata.location,
                'created_at': metadata.created_at.isoformat(),
                'processing_completed_at': datetime.utcnow().isoformat(),
                'statistics': {
                    'total_frames': index_data['total_frames'],
                    'processed_frames': index_data['processed_frames'],
                    'threat_frames': index_data['threat_frames'],
                    'total_alerts': alerts_data['total_alerts'],
                    'high_alerts': alerts_data['high_alerts'],
                    'medium_alerts': alerts_data['medium_alerts'],
                    'low_alerts': alerts_data['low_alerts']
                },
                'threat_distribution': self.calculate_threat_distribution(index_data['frames']),
                'timeline': self.create_threat_timeline(index_data['frames']),
                'folder_structure': {
                    'video_folder': str(self.base_path / video_id),
                    'frames_folder': str(self.base_path / video_id / self.frames_dir_name),
                    'analysis_folder': str(self.base_path / video_id / self.analysis_dir_name),
                    'metadata_file': str(self.base_path / video_id / self.metadata_file),
                    'index_file': str(self.base_path / video_id / self.index_file),
                    'alerts_file': str(self.base_path / video_id / self.alerts_file),
                    'summary_file': str(summary_file)
                }
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
                
            logger.info(f"Created summary for video {video_id}")
            
        except Exception as e:
            logger.error(f"Failed to create summary: {e}")
    
    def calculate_threat_distribution(self, frames: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate threat level distribution"""
        distribution = {'low': 0, 'medium': 0, 'high': 0, 'none': 0}
        
        for frame in frames:
            threat_level = frame['threat_level'].lower()
            if threat_level in distribution:
                distribution[threat_level] += 1
        
        return distribution
    
    def create_threat_timeline(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create timeline of threat events"""
        threat_frames = [f for f in frames if f['threat_level'].lower() in ['high', 'medium']]
        
        timeline = []
        for frame in threat_frames:
            timeline.append({
                'frame_number': frame['frame_number'],
                'timestamp': frame['timestamp'],
                'threat_level': frame['threat_level'],
                'confidence': frame['confidence'],
                'people_detected': frame['people_detected']
            })
        
        return sorted(timeline, key=lambda x: x['frame_number'])
    
    async def index_video(self, video_path: Path, video_metadata: VideoMetadata,
                         telemetry_base: Dict[str, Any], 
                         frame_interval: int = 1) -> str:
        """Complete video indexing pipeline"""
        try:
            logger.info(f"Starting video indexing for {video_metadata.video_id}")
            
            # 1. Create video folder structure
            video_folder = self.create_video_folder(video_metadata)
            
            # 2. Copy video file to folder
            video_dest = video_folder / video_metadata.filename
            shutil.copy2(video_path, video_dest)
            
            # 3. Extract frames
            frames_list = self.extract_frames(video_dest, video_metadata.video_id, frame_interval)
            
            if not frames_list:
                raise ValueError("No frames extracted from video")
            
            # 4. Process frames and analyze
            await self.process_video_frames(video_metadata.video_id, frames_list, telemetry_base)
            
            logger.info(f"Successfully indexed video {video_metadata.video_id}")
            return video_metadata.video_id
            
        except Exception as e:
            logger.error(f"Video indexing failed: {e}")
            # Update status to failed
            metadata = self.load_video_metadata(video_metadata.video_id)
            if metadata:
                metadata.indexing_status = "failed"
                self.save_video_metadata(metadata)
            raise
    
    def get_video_analysis(self, video_id: str, frame_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analysis results for video or specific frame"""
        video_folder = self.base_path / video_id
        
        if not video_folder.exists():
            return {'error': f'Video {video_id} not found'}
        
        if frame_id:
            # Get specific frame analysis
            analysis_file = video_folder / self.analysis_dir_name / f"{frame_id}.json"
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    return json.load(f)
            else:
                return {'error': f'Frame {frame_id} not found'}
        else:
            # Get video summary
            summary_file = video_folder / self.summary_file
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    return json.load(f)
            else:
                return {'error': f'Video summary not found'}
    
    def search_video_frames(self, video_id: str, threat_level: Optional[str] = None,
                           min_confidence: Optional[float] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Search frames within a video"""
        index_file = self.base_path / video_id / self.index_file
        
        try:
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            frames = index_data['frames']
            
            # Apply filters
            if threat_level:
                frames = [f for f in frames if f['threat_level'].lower() == threat_level.lower()]
            
            if min_confidence:
                frames = [f for f in frames if f['confidence'] >= min_confidence]
            
            # Sort by frame number and limit
            frames = sorted(frames, key=lambda x: x['frame_number'])[:limit]
            
            return frames
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def list_videos(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all indexed videos"""
        videos = []
        
        try:
            for video_folder in self.base_path.iterdir():
                if video_folder.is_dir():
                    video_id = video_folder.name
                    
                    # Load metadata
                    metadata = self.load_video_metadata(video_id)
                    if metadata:
                        video_info = {
                            'video_id': video_id,
                            'filename': metadata.filename,
                            'location': metadata.location,
                            'created_at': metadata.created_at.isoformat(),
                            'total_frames': metadata.total_frames,
                            'processed_frames': metadata.processed_frames,
                            'threat_frames': metadata.threat_frames,
                            'indexing_status': metadata.indexing_status,
                            'folder_path': str(video_folder)
                        }
                        
                        if status is None or metadata.indexing_status == status:
                            videos.append(video_info)
            
            return sorted(videos, key=lambda x: x['created_at'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list videos: {e}")
            return []
    
    def get_video_alerts(self, video_id: str, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alerts for a video"""
        alerts_file = self.base_path / video_id / self.alerts_file
        
        try:
            with open(alerts_file, 'r') as f:
                alerts_data = json.load(f)
            
            alerts = alerts_data['alerts']
            
            if severity:
                alerts = [a for a in alerts if a['threat_level'].lower() == severity.lower()]
            
            return sorted(alerts, key=lambda x: x['created_at'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []

# Global instance
video_folder_indexer = VideoFolderIndexer()
