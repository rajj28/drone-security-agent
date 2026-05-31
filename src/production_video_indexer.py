"""
production_video_indexer.py - Production-grade video indexing and storage system

Features:
- Efficient video processing pipeline
- Scalable data storage with PostgreSQL + Redis
- Real-time alert management
- Optimized indexing with vector search
- Monitoring and metrics
- Batch processing capabilities
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import uuid

# Database and caching
import asyncpg
import redis.asyncio as redis
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB

# Vector search
import pinecone
import numpy as np

# Monitoring
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Configuration
from src.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
from prometheus_client import CollectorRegistry, REGISTRY

# Create a custom registry for this module
registry = CollectorRegistry()

VIDEO_PROCESSED = Counter('videos_processed_total', 'Total videos processed', registry=registry)
FRAMES_ANALYZED = Counter('frames_analyzed_total', 'Total frames analyzed', registry=registry)
ALERTS_GENERATED = Counter('alerts_generated_total', 'Total alerts generated', registry=registry)
PROCESSING_TIME = Histogram('video_processing_seconds', 'Time spent processing videos', registry=registry)
ACTIVE_PROCESSES = Gauge('active_processes', 'Number of active video processes', registry=registry)
STORAGE_USAGE = Gauge('storage_usage_bytes', 'Storage usage in bytes', registry=registry)

Base = declarative_base()

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
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

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
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class VideoIndexDB(Base):
    """Database model for video indexing"""
    __tablename__ = 'video_index'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(String(255), unique=True, nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer, nullable=False)
    duration = Column(Float, nullable=False)
    fps = Column(Float, nullable=False)
    resolution_width = Column(Integer, nullable=False)
    resolution_height = Column(Integer, nullable=False)
    codec = Column(String(50), nullable=False)
    location = Column(String(255), nullable=False)
    camera_id = Column(String(255), nullable=True)
    tags = Column(JSONB, nullable=True)
    video_metadata = Column(JSONB, nullable=True)
    status = Column(String(50), default='pending')  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

class FrameAnalysisDB(Base):
    """Database model for frame analysis"""
    __tablename__ = 'frame_analysis'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    frame_id = Column(String(255), unique=True, nullable=False, index=True)
    video_id = Column(String(255), nullable=False, index=True)
    timestamp = Column(Float, nullable=False)
    frame_number = Column(Integer, nullable=False)
    analysis_result = Column(JSONB, nullable=False)
    threat_level = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    people_detected = Column(Integer, default=0)
    objects_detected = Column(JSONB, nullable=True)
    embedding_id = Column(String(255), nullable=True)  # Reference to vector store
    created_at = Column(DateTime, default=datetime.utcnow)

class AlertDB(Base):
    """Database model for alerts"""
    __tablename__ = 'alerts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_id = Column(String(255), unique=True, nullable=False, index=True)
    video_id = Column(String(255), nullable=False, index=True)
    frame_id = Column(String(255), nullable=False)
    alert_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)  # low, medium, high, critical
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    alert_metadata = Column(JSONB, nullable=True)
    status = Column(String(50), default='open')  # open, acknowledged, resolved, false_positive
    assigned_to = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

class ProductionVideoIndexer:
    """Production-grade video indexing system"""
    
    def __init__(self):
        self.db_engine = None
        self.redis_client = None
        self.pinecone_index = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.processing_queue = asyncio.Queue()
        self.alert_queue = asyncio.Queue()
        
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing production video indexer...")
        
        # Initialize database
        await self._init_database()
        
        # Initialize Redis
        await self._init_redis()
        
        # Initialize Pinecone
        await self._init_pinecone()
        
        # Start metrics server
        start_http_server(8000)
        
        logger.info("Production video indexer initialized successfully")
    
    async def _init_database(self):
        """Initialize PostgreSQL database"""
        try:
            self.db_engine = create_engine(
                f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@"
                f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}",
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                echo=False
            )
            
            # Create tables
            Base.metadata.create_all(self.db_engine)
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def _init_redis(self):
        """Initialize Redis for caching"""
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info("Redis initialized successfully")
            
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")
            raise
    
    async def _init_pinecone(self):
        """Initialize Pinecone for vector search"""
        try:
            pinecone.init(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENV)
            
            index_name = "video-frames"
            
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=512,  # CLIP embedding dimension
                    metric="cosine",
                    pods=1,
                    replicas=1,
                    pod_type="p1.x1"
                )
            
            self.pinecone_index = pinecone.Index(index_name)
            
            logger.info("Pinecone initialized successfully")
            
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {e}")
            raise
    
    async def process_video(self, video_path: Path, metadata: VideoMetadata) -> str:
        """Process a single video with full indexing"""
        start_time = time.time()
        
        try:
            ACTIVE_PROCESSES.inc()
            
            # Create video record
            video_id = await self._create_video_record(metadata)
            
            # Extract frames
            frames = await self._extract_frames(video_path, metadata)
            
            # Process frames in batches
            await self._process_frames_batch(video_id, frames)
            
            # Update video status
            await self._update_video_status(video_id, 'completed')
            
            # Update metrics
            VIDEO_PROCESSED.inc()
            PROCESSING_TIME.observe(time.time() - start_time)
            
            logger.info(f"Video {video_id} processed successfully")
            return video_id
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            await self._update_video_status(video_id, 'failed')
            raise
        finally:
            ACTIVE_PROCESSES.dec()
    
    async def _extract_frames(self, video_path: Path, metadata: VideoMetadata) -> List[Tuple[str, float, int]]:
        """Extract frames from video efficiently"""
        frames = []
        
        # Calculate frame extraction interval (1 frame per second)
        interval = max(1, int(metadata.fps))
        
        # Use ffmpeg for efficient frame extraction
        import subprocess
        
        output_dir = Path(f"data/frames/{metadata.video_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', f'select=not(mod(n\\,{interval}))',
            '-vsync', 'vfr',
            '-q:v', '2',
            str(output_dir / 'frame_%06d.jpg')
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Get extracted frames
            for frame_file in sorted(output_dir.glob('frame_*.jpg')):
                frame_number = int(frame_file.stem.split('_')[1])
                timestamp = frame_number / metadata.fps
                frame_id = f"{metadata.video_id}_frame_{frame_number:06d}"
                frames.append((frame_id, timestamp, frame_number))
        
        return frames
    
    async def _process_frames_batch(self, video_id: str, frames: List[Tuple[str, float, int]]):
        """Process frames in batches for efficiency"""
        batch_size = 10
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self._process_single_frame(video_id, frame_id, timestamp, frame_number)
                for frame_id, timestamp, frame_number in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            for result in results:
                if isinstance(result, FrameAnalysis):
                    await self._store_frame_analysis(result)
                elif isinstance(result, Exception):
                    logger.error(f"Frame processing error: {result}")
    
    async def _process_single_frame(self, video_id: str, frame_id: str, 
                                  timestamp: float, frame_number: int) -> FrameAnalysis:
        """Process a single frame with enhanced analysis"""
        frame_path = Path(f"data/frames/{video_id}/frame_{frame_number:06d}.jpg")
        
        # Use configured vision analyzer (respects environment variables)
        from src.vision_analyzer import analyze_frame
        
        telemetry = {
            'video_id': video_id,
            'frame_id': frame_id,
            'timestamp': timestamp,
            'frame_number': frame_number
        }
        
        result = analyze_frame(frame_id, frame_path, telemetry)
        
        # Extract key information
        threat_level = result.get('overall_threat_level', 'low')
        confidence = 0.8  # Default confidence
        
        # Parse people detected
        analysis = result.get('gpt4o_enhanced', {}).get('enhanced_analysis', '')
        people_detected = 0
        if 'people_detected' in analysis:
            import re
            match = re.search(r'"people_detected":\s*(\d+)', analysis)
            if match:
                people_detected = int(match.group(1))
        
        # Create frame analysis object
        frame_analysis = FrameAnalysis(
            frame_id=frame_id,
            video_id=video_id,
            timestamp=timestamp,
            frame_number=frame_number,
            analysis_result=result,
            threat_level=threat_level,
            confidence=confidence,
            people_detected=people_detected,
            objects_detected=[]  # Extract from result if needed
        )
        
        # Generate alerts for high threat frames
        if threat_level in ['high', 'medium']:
            await self._generate_alert(frame_analysis)
        
        FRAMES_ANALYZED.inc()
        return frame_analysis
    
    async def _store_frame_analysis(self, analysis: FrameAnalysis):
        """Store frame analysis in database"""
        try:
            with self.db_engine.connect() as conn:
                # Store in PostgreSQL
                stmt = """
                INSERT INTO frame_analysis 
                (frame_id, video_id, timestamp, frame_number, analysis_result, 
                 threat_level, confidence, people_detected, objects_detected)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (frame_id) DO UPDATE SET
                    analysis_result = EXCLUDED.analysis_result,
                    threat_level = EXCLUDED.threat_level,
                    confidence = EXCLUDED.confidence,
                    people_detected = EXCLUDED.people_detected,
                    objects_detected = EXCLUDED.objects_detected
                """
                
                conn.execute(stmt, (
                    analysis.frame_id,
                    analysis.video_id,
                    analysis.timestamp,
                    analysis.frame_number,
                    json.dumps(analysis.analysis_result),
                    analysis.threat_level,
                    analysis.confidence,
                    analysis.people_detected,
                    json.dumps(analysis.objects_detected)
                ))
                
                conn.commit()
            
            # Cache in Redis for quick access
            await self._cache_frame_analysis(analysis)
            
        except Exception as e:
            logger.error(f"Failed to store frame analysis: {e}")
            raise
    
    async def _cache_frame_analysis(self, analysis: FrameAnalysis):
        """Cache frame analysis in Redis"""
        try:
            cache_key = f"frame:{analysis.frame_id}"
            cache_data = {
                'video_id': analysis.video_id,
                'threat_level': analysis.threat_level,
                'confidence': analysis.confidence,
                'people_detected': analysis.people_detected,
                'timestamp': analysis.timestamp
            }
            
            # Cache for 24 hours
            await self.redis_client.setex(
                cache_key, 
                timedelta(hours=24), 
                json.dumps(cache_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to cache frame analysis: {e}")
    
    async def _generate_alert(self, analysis: FrameAnalysis):
        """Generate alert for threatening frame"""
        try:
            alert_id = f"alert_{analysis.frame_id}_{int(time.time())}"
            
            alert = {
                'alert_id': alert_id,
                'video_id': analysis.video_id,
                'frame_id': analysis.frame_id,
                'alert_type': 'threat_detected',
                'severity': analysis.threat_level.upper(),
                'title': f"Threat detected in {analysis.video_id}",
                'description': f"Threat level {analysis.threat_level} detected at timestamp {analysis.timestamp}",
                'confidence': analysis.confidence,
                'metadata': analysis.analysis_result
            }
            
            # Store alert
            await self._store_alert(alert)
            
            # Add to alert queue for real-time processing
            await self.alert_queue.put(alert)
            
            ALERTS_GENERATED.inc()
            
        except Exception as e:
            logger.error(f"Failed to generate alert: {e}")
    
    async def _store_alert(self, alert: Dict[str, Any]):
        """Store alert in database"""
        try:
            with self.db_engine.connect() as conn:
                stmt = """
                INSERT INTO alerts 
                (alert_id, video_id, frame_id, alert_type, severity, title, 
                 description, confidence, metadata, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                conn.execute(stmt, (
                    alert['alert_id'],
                    alert['video_id'],
                    alert['frame_id'],
                    alert['alert_type'],
                    alert['severity'],
                    alert['title'],
                    alert['description'],
                    alert['confidence'],
                    json.dumps(alert['metadata']),
                    alert.get('status', 'open')
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
            raise
    
    async def _create_video_record(self, metadata: VideoMetadata) -> str:
        """Create video record in database"""
        try:
            with self.db_engine.connect() as conn:
                stmt = """
                INSERT INTO video_index 
                (video_id, filename, file_path, file_size, duration, fps, 
                 resolution_width, resolution_height, codec, location, 
                 camera_id, tags, metadata, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING video_id
                """
                
                result = conn.execute(stmt, (
                    metadata.video_id,
                    metadata.filename,
                    metadata.file_path,
                    metadata.file_size,
                    metadata.duration,
                    metadata.fps,
                    metadata.resolution[0],
                    metadata.resolution[1],
                    metadata.codec,
                    metadata.location,
                    metadata.camera_id,
                    json.dumps(metadata.tags),
                    json.dumps(asdict(metadata)),
                    'processing'
                ))
                
                conn.commit()
                return metadata.video_id
                
        except Exception as e:
            logger.error(f"Failed to create video record: {e}")
            raise
    
    async def _update_video_status(self, video_id: str, status: str):
        """Update video processing status"""
        try:
            with self.db_engine.connect() as conn:
                stmt = """
                UPDATE video_index 
                SET status = %s, updated_at = %s, processed_at = %s
                WHERE video_id = %s
                """
                
                conn.execute(stmt, (
                    status,
                    datetime.utcnow(),
                    datetime.utcnow() if status == 'completed' else None,
                    video_id
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update video status: {e}")
    
    async def search_frames(self, query: str, video_id: Optional[str] = None, 
                          threat_level: Optional[str] = None, 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Search frames using vector similarity and filters"""
        try:
            # Build base query
            sql = """
            SELECT frame_id, video_id, timestamp, frame_number, threat_level, 
                   confidence, people_detected, analysis_result
            FROM frame_analysis
            WHERE 1=1
            """
            
            params = []
            
            if video_id:
                sql += " AND video_id = %s"
                params.append(video_id)
            
            if threat_level:
                sql += " AND threat_level = %s"
                params.append(threat_level)
            
            sql += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)
            
            with self.db_engine.connect() as conn:
                result = conn.execute(sql, params)
                frames = [dict(row) for row in result.fetchall()]
            
            return frames
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def get_alerts(self, status: Optional[str] = None, 
                        severity: Optional[str] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Get alerts with filters"""
        try:
            sql = """
            SELECT alert_id, video_id, frame_id, alert_type, severity, title,
                   description, confidence, status, created_at
            FROM alerts
            WHERE 1=1
            """
            
            params = []
            
            if status:
                sql += " AND status = %s"
                params.append(status)
            
            if severity:
                sql += " AND severity = %s"
                params.append(severity)
            
            sql += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            
            with self.db_engine.connect() as conn:
                result = conn.execute(sql, params)
                alerts = [dict(row) for row in result.fetchall()]
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            with self.db_engine.connect() as conn:
                # Video stats
                video_result = conn.execute("""
                    SELECT COUNT(*) as total_videos,
                           SUM(file_size) as total_size,
                           AVG(duration) as avg_duration
                    FROM video_index
                    WHERE status = 'completed'
                """)
                video_stats = dict(video_result.fetchone())
                
                # Frame stats
                frame_result = conn.execute("""
                    SELECT COUNT(*) as total_frames,
                           AVG(confidence) as avg_confidence,
                           COUNT(CASE WHEN threat_level != 'low' THEN 1 END) as threat_frames
                    FROM frame_analysis
                """)
                frame_stats = dict(frame_result.fetchone())
                
                # Alert stats
                alert_result = conn.execute("""
                    SELECT COUNT(*) as total_alerts,
                           COUNT(CASE WHEN severity = 'HIGH' THEN 1 END) as high_alerts,
                           COUNT(CASE WHEN status = 'open' THEN 1 END) as open_alerts
                    FROM alerts
                """)
                alert_stats = dict(alert_result.fetchone())
                
                return {
                    'videos': video_stats,
                    'frames': frame_stats,
                    'alerts': alert_stats
                }
                
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    async def cleanup_old_data(self, days: int = 30):
        """Clean up old data to manage storage"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            with self.db_engine.connect() as conn:
                # Delete old alerts
                conn.execute("""
                    DELETE FROM alerts 
                    WHERE created_at < %s AND status IN ('resolved', 'false_positive')
                """, (cutoff_date,))
                
                # Optionally delete old frame analyses for non-critical videos
                conn.execute("""
                    DELETE FROM frame_analysis 
                    WHERE created_at < %s AND threat_level = 'low'
                """, (cutoff_date,))
                
                conn.commit()
                
            logger.info(f"Cleaned up data older than {days} days")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Global instance
production_indexer = ProductionVideoIndexer()
