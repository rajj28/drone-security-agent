"""
api.py — FastAPI backend for Drone Security Analyst Agent.

- Exposes endpoints for health, frames, analysis, alerts, search, session summary, and Q&A
- Video upload and processing capabilities
- Production-ready error handling and logging
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import uuid
import time
import logging
from datetime import datetime
import shutil
import os

from src.config import settings
from src.pinecone_indexer import search_frames
from src.qa_agent import SecurityQAAgent
from src.mongodb_storage import get_mongodb_storage

def get_next_extracted_folder():
    """Get the next available extracted folder number (extracted1, extracted2, etc.)"""
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
    
    # Find all existing extracted folders
    extracted_folders = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.startswith("extracted"):
            try:
                # Extract number from folder name (e.g., "extracted1" -> 1)
                num = int(item.name.replace("extracted", ""))
                extracted_folders.append(num)
            except ValueError:
                continue
    
    # Get next available number
    next_num = max(extracted_folders) + 1 if extracted_folders else 1
    return f"extracted{next_num}"

def get_latest_extracted_folder():
    """Get the most recent extracted folder"""
    data_dir = Path("data")
    if not data_dir.exists():
        return None
    
    # Find all existing extracted folders
    extracted_folders = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.startswith("extracted"):
            try:
                # Extract number from folder name (e.g., "extracted1" -> 1)
                num = int(item.name.replace("extracted", ""))
                extracted_folders.append((num, item))
            except ValueError:
                continue
    
    if not extracted_folders:
        return None
    
    # Return the folder with the highest number
    latest = max(extracted_folders, key=lambda x: x[0])
    return latest[1]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check ffmpeg availability at startup
def check_ffmpeg():
    """Verify ffmpeg is installed."""
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.decode().split('\n')[0]
            logger.info(f"✅ FFmpeg available: {version[:60]}")
            return True
        else:
            logger.error("❌ FFmpeg not working properly")
            return False
    except Exception as e:
        logger.error(f"❌ FFmpeg not found: {e}")
        return False

FFMPEG_AVAILABLE = check_ffmpeg()

app = FastAPI(
    title="Drone Security Analyst API",
    description="Production-ready AI-powered security analysis system",
    version="2.0.0"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {"message": "Drone Security Analyst API", "status": "running"}

@app.get("/api")
def api_info():
    """API information endpoint"""
    return {
        "message": "Drone Security Analyst API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "frames": "/frames",
            "sessions": "/sessions",
            "upload": "/upload-video",
            "search": "/search"
        },
        "documentation": "/docs"
    }

# Global variable to track processing status (fallback if MongoDB not available)
processing_status = {}

# Initialize MongoDB storage for persistent session tracking
mongodb_storage = get_mongodb_storage()
logger.info(f"MongoDB connection status: {mongodb_storage.is_connected()}")

def save_session_status(session_id: str, status_data: dict):
    """Save session status to both memory and MongoDB."""
    processing_status[session_id] = status_data
    if mongodb_storage.is_connected():
        mongodb_storage.save_session(session_id, status_data)

def get_session_status(session_id: str) -> dict:
    """Get session status from MongoDB or memory."""
    # Try MongoDB first (persistent across restarts)
    if mongodb_storage.is_connected():
        db_status = mongodb_storage.get_session(session_id)
        if db_status:
            # Update memory cache
            processing_status[session_id] = db_status
            return db_status
    # Fallback to memory
    return processing_status.get(session_id)

def get_all_session_statuses() -> list:
    """Get all sessions from MongoDB or memory."""
    if mongodb_storage.is_connected():
        return mongodb_storage.get_all_sessions()
    # Fallback to memory
    return [
        {
            "session_id": sid,
            "filename": status.get("filename", "unknown"),
            "status": status.get("status", "unknown"),
            "upload_time": status.get("upload_time"),
            "progress": status.get("progress", 0)
        }
        for sid, status in processing_status.items()
    ]

@app.get("/health")
def health():
    """Health check endpoint with system status."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "system": "drone-security-agent",
        "ffmpeg_available": FFMPEG_AVAILABLE,
        "mongodb_connected": mongodb_storage.is_connected(),
        "persistence": "mongodb" if mongodb_storage.is_connected() else "memory-only"
    }

@app.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    extraction_strategy: str = "hybrid",
    max_frames: int = 100
):
    """
    Upload and process a video file for security analysis with intelligent frame extraction.
    
    Args:
        file: Video file to upload
        session_id: Optional session ID
        extraction_strategy: Frame extraction strategy (uniform, motion_based, scene_change, hybrid)
        max_frames: Maximum number of frames to extract (10-500)
    
    Supports 14+ video formats: .mp4, .avi, .mov, .dav, .mkv, .wmv, .flv, .webm, .mpeg, .3gp, .ts, .m4v, .m2ts
    """
    try:
        # Validate file type
        allowed_extensions = {".mp4", ".avi", ".mov", ".dav", ".mkv", ".wmv", ".flv", ".webm", ".mpeg", ".mpg", ".3gp", ".ts", ".m4v", ".m2ts"}
        
        # Debug: Log file object details
        logger.info(f"File object - filename: {file.filename}, content_type: {file.content_type}")
        
        # Handle case where filename might be None or empty
        if not file.filename:
            logger.error("Filename is None or empty")
            raise HTTPException(
                status_code=400,
                detail="No filename provided. Please select a file."
            )
        
        file_extension = Path(file.filename).suffix.lower()
        
        # Add debugging log
        logger.info(f"Upload attempt - filename: {file.filename}, extension: {file_extension}")
        
        if file_extension not in allowed_extensions:
            logger.error(f"Invalid extension: {file_extension}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{file_extension}'. Allowed types: {', '.join(sorted(allowed_extensions))}"
            )
        
        # Validate extraction strategy
        valid_strategies = ["uniform", "motion_based", "scene_change", "hybrid"]
        if extraction_strategy not in valid_strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid extraction strategy: {extraction_strategy}. Valid options: {', '.join(valid_strategies)}"
            )
        
        # Validate max_frames
        if max_frames < 10 or max_frames > 500:
            raise HTTPException(
                status_code=400,
                detail="max_frames must be between 10 and 500"
            )
        
        # Generate session ID if not provided (for tracking purposes)
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Create session-specific directory for ALL processing
        session_dir = Path("data") / "sessions" / session_id
        extracted_dir = session_dir / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded video in the session folder
        video_path = session_dir / file.filename
        try:
            # Reset file position to beginning
            file.file.seek(0)
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Verify file was saved properly
            if video_path.stat().st_size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file appears to be empty. Please select a valid video file."
                )
            
            logger.info(f"Video saved successfully: {video_path} ({video_path.stat().st_size} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to save video file: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save uploaded file: {str(e)}"
            )
        
        logger.info(f"Video saved to session: {session_id}, extracted dir: {extracted_dir}")
        
        # Initialize processing status (saved to MongoDB if available)
        initial_status = {
            "status": "uploaded",
            "filename": file.filename,
            "upload_time": datetime.now().isoformat(),
            "processing_steps": [],
            "current_step": "queued",
            "progress": 0,
            "error": None,
            "extraction_strategy": extraction_strategy,
            "max_frames": max_frames
        }
        save_session_status(session_id, initial_status)
        logger.info(f"Session {session_id} status saved to {'MongoDB' if mongodb_storage.is_connected() else 'memory'}")
        
        # Start processing in background - use session directory
        background_tasks.add_task(
            process_video_pipeline,
            session_id,
            str(video_path),
            str(extracted_dir),  # Use session extracted folder
            extraction_strategy,
            max_frames
        )
        
        logger.info(f"Video uploaded: {file.filename} (Session: {session_id})")
        
        return {
            "session_id": session_id,
            "message": f"Video uploaded successfully. Processing started with {extraction_strategy} strategy.",
            "filename": file.filename,
            "extraction_strategy": extraction_strategy,
            "max_frames": max_frames,
            "estimated_time": "5-10 minutes"
        }
        
    except Exception as e:
        logger.error(f"Video upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/processing-status/{session_id}")
def get_processing_status(session_id: str):
    """Get the current processing status for a video session from MongoDB or memory."""
    status = get_session_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return status

@app.get("/sessions")
def list_sessions():
    """List all processing sessions from MongoDB or memory."""
    sessions = get_all_session_statuses()
    
    # Format for response
    formatted_sessions = []
    for session in sessions:
        if isinstance(session, dict):
            formatted_sessions.append({
                "session_id": session.get("session_id", "unknown"),
                "filename": session.get("filename", "unknown"),
                "status": session.get("status", "unknown"),
                "upload_time": session.get("upload_time"),
                "progress": session.get("progress", 0)
            })
    
    return {"sessions": formatted_sessions, "source": "mongodb" if mongodb_storage.is_connected() else "memory"}

async def keep_alive_heartbeat(session_id: str, interval: int = 60):
    """
    Keep-alive heartbeat to prevent Render free tier from sleeping during long operations.
    Logs activity every `interval` seconds to keep the service awake.
    """
    import asyncio
    heartbeat_count = 0
    while True:
        status = get_session_status(session_id)
        if not status or status.get("status") != "processing":
            break
        await asyncio.sleep(interval)
        heartbeat_count += 1
        current_step = status.get('current_step', 'unknown')
        logger.info(f"[{session_id}] Keep-alive heartbeat #{heartbeat_count} - service active, status: {current_step}")
        # Update timestamp to show activity
        status_update = get_session_status(session_id) or {}
        status_update["last_heartbeat"] = time.time()
        save_session_status(session_id, status_update)


async def process_video_pipeline(session_id: str, video_path: str, session_dir: str, extraction_strategy: str = "hybrid", max_frames: int = 100):
    """
    Background task to process uploaded video through the complete pipeline with intelligent frame extraction.
    Includes keep-alive heartbeat to prevent service sleep during long operations.
    """
    try:
        import sys
        import subprocess
        import asyncio
        
        # Update status
        status_update = get_session_status(session_id) or {}
        status_update["status"] = "processing"
        status_update["current_step"] = "extracting_frames"
        status_update["start_time"] = time.time()
        save_session_status(session_id, status_update)
        
        # Start keep-alive heartbeat to prevent Render sleep (every 60 seconds)
        heartbeat_task = asyncio.create_task(keep_alive_heartbeat(session_id, interval=60))
        logger.info(f"[{session_id}] Started keep-alive heartbeat for long processing")
        
        # Step 1: Check ffmpeg availability
        if not FFMPEG_AVAILABLE:
            logger.error(f"[{session_id}] ❌ FFmpeg not available - cannot extract frames!")
            status_update = get_session_status(session_id) or {}
            status_update["status"] = "failed"
            status_update["error"] = "FFmpeg not installed"
            save_session_status(session_id, status_update)
            return
        
        logger.info(f"[{session_id}] ✅ FFmpeg available, starting frame extraction")
        
        # Step 2: Extract frames using intelligent extractor
        logger.info(f"[PIPELINE] Starting frame extraction for session {session_id}")
        
        # Update video path in settings for the extractor
        from src.config import settings
        original_video_file = settings.VIDEO_FILE
        settings.VIDEO_FILE = Path(video_path)
        original_extracted_dir = settings.EXTRACTED_DIR
        original_outputs_dir = settings.OUTPUTS_DIR
        
        # session_dir is ALREADY the extracted folder (passed from upload endpoint)
        # e.g., data/sessions/{session_id}/extracted
        extracted_dir = Path(session_dir)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        
        # Also set outputs dir to session outputs for extraction log
        session_root = extracted_dir.parent
        outputs_dir = session_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Update settings
        settings.EXTRACTED_DIR = extracted_dir
        settings.OUTPUTS_DIR = outputs_dir
        
        logger.info(f"[PIPELINE] Extracting frames to: {extracted_dir}")
        logger.info(f"[PIPELINE] Extraction log to: {outputs_dir}")
        
        try:
            # Use intelligent frame extractor
            from src.intelligent_frame_extractor import extract_frames_intelligently
            
            frames = extract_frames_intelligently(
                video_path=Path(video_path),
                output_dir=settings.EXTRACTED_DIR,
                strategy=extraction_strategy,
                max_frames=max_frames
            )
            
            logger.info(f"Successfully extracted {len(frames)} frames using {extraction_strategy} strategy")
            
            # Extract file paths from frame dicts (if dicts) or use as-is (if strings)
            frame_paths = []
            for f in frames:
                if isinstance(f, dict):
                    # If frame is a dict, get the file_path or construct from session_dir
                    if 'file_path' in f:
                        frame_paths.append(f['file_path'])
                    elif 'filename' in f:
                        # Don't add .jpg if filename already has it
                        filename = f['filename']
                        if not filename.endswith('.jpg'):
                            filename += '.jpg'
                        # Frames saved directly to session_dir, not session_dir/extracted/
                        frame_paths.append(f"{session_dir}/{filename}")
                else:
                    # Frame is already a path string
                    frame_paths.append(str(f))
            
            logger.info(f"[STORAGE] Extracted {len(frame_paths)} frame paths for storage")
            
            # Verify files actually exist and find where they are
            existing_files = [p for p in frame_paths if Path(p).exists()]
            logger.info(f"[STORAGE] {len(existing_files)}/{len(frame_paths)} frame files exist on disk")
            
            if len(existing_files) == 0 and len(frame_paths) > 0:
                logger.error(f"[STORAGE] CRITICAL: No frame files found! Looking in: {session_dir}")
                # Try to find where frames actually are
                session_path = Path(session_dir)
                if session_path.exists():
                    jpg_files = list(session_path.glob("*.jpg"))
                    logger.info(f"[STORAGE] Found {len(jpg_files)} .jpg files in {session_dir}")
                    for f in jpg_files[:5]:
                        logger.info(f"[STORAGE]   -> {f.name}")
                
                # Check outputs directory (where extraction log is saved)
                outputs_path = Path("outputs")
                if outputs_path.exists():
                    output_jpgs = list(outputs_path.glob("*.jpg"))
                    if output_jpgs:
                        logger.info(f"[STORAGE] Found {len(output_jpgs)} .jpg files in outputs/")
            
            # Save frame list to processing_status for persistence
            status_update = get_session_status(session_id) or {}
            status_update["extracted_frames"] = frame_paths
            status_update["frame_count"] = len(frame_paths)
            status_update["session_dir"] = str(session_dir)
            save_session_status(session_id, status_update)
            
        except Exception as e:
            # Fallback to basic frame extractor if intelligent one fails
            logger.warning(f"Intelligent extraction failed, falling back to basic extraction: {e}")
            extraction_output_dir = session_dir + "/extracted"
            logger.info(f"[EXTRACTION] Running frame_extractor with output: {extraction_output_dir}")
            result = subprocess.run([
                sys.executable, "-m", "src.frame_extractor",
                "--input", video_path,
                "--output", extraction_output_dir
            ], capture_output=True, text=True)
            
            logger.info(f"[EXTRACTION] Return code: {result.returncode}")
            logger.info(f"[EXTRACTION] stdout: {result.stdout[:500] if result.stdout else 'empty'}")
            logger.info(f"[EXTRACTION] stderr: {result.stderr[:500] if result.stderr else 'empty'}")
            
            if result.returncode != 0:
                raise Exception(f"Frame extraction failed: {result.stderr}")
            
            # Get frames from fallback extraction (only frame_*.jpg, not temp files)
            fallback_frames = sorted([str(f) for f in (Path(session_dir) / "extracted").glob("frame_*.jpg")])
            logger.info(f"[FALLBACK] Found {len(fallback_frames)} frames in {session_dir}/extracted")
            status_update = get_session_status(session_id) or {}
            status_update["extracted_frames"] = fallback_frames
            status_update["frame_count"] = len(fallback_frames)
            status_update["session_dir"] = str(session_dir)
            save_session_status(session_id, status_update)
            logger.info(f"[FALLBACK] Stored {len(fallback_frames)} frames in processing_status")
        
        finally:
            # Restore original settings
            settings.VIDEO_FILE = original_video_file
            settings.EXTRACTED_DIR = original_extracted_dir
            settings.OUTPUTS_DIR = original_outputs_dir
        
        status_update = get_session_status(session_id) or {}
        status_update.setdefault("processing_steps", []).append("frame_extraction")
        status_update["progress"] = 20
        
        # Step 2: Generate telemetry
        status_update["current_step"] = "generating_telemetry"
        save_session_status(session_id, status_update)
        logger.info(f"[PIPELINE] Generating telemetry for session {session_id}")
        
        # session_dir is ALREADY the extracted folder (e.g., data/sessions/{id}/extracted)
        # Derive session root from it
        extracted_dir = Path(session_dir)
        session_root = extracted_dir.parent
        telemetry_dir = session_root / "telemetry"
        
        logger.info(f"[PIPELINE] Using EXTRACTED_DIR: {extracted_dir}")
        logger.info(f"[PIPELINE] Using TELEMETRY_DIR: {telemetry_dir}")
        
        # Ensure telemetry directory exists
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config to use session directory
        original_extracted_dir = settings.EXTRACTED_DIR
        original_telemetry_dir = settings.TELEMETRY_DIR
        
        settings.EXTRACTED_DIR = extracted_dir
        settings.TELEMETRY_DIR = telemetry_dir
        
        # Pass SESSION_ID so telemetry generator uses correct directory
        env = os.environ.copy()
        env["SESSION_ID"] = session_id
        
        result = subprocess.run([
            sys.executable, "-m", "src.telemetry_generator"
        ], capture_output=True, text=True, env=env, timeout=300)  # 5 minute timeout
        
        if result.returncode != 0:
            logger.error(f"Telemetry generation stderr: {result.stderr}")
            raise Exception(f"Telemetry generation failed: {result.stderr}")
        
        status_update = get_session_status(session_id) or {}
        status_update.setdefault("processing_steps", []).append("telemetry_generation")
        status_update["progress"] = 40
        
        # Step 3: Vision analysis
        status_update["current_step"] = "analyzing_frames"
        save_session_status(session_id, status_update)
        logger.info(f"[PIPELINE] Running vision analysis for session {session_id}")
        logger.info(f"[PIPELINE] Looking for frames in: {settings.EXTRACTED_DIR}")
        
        # Verify frames exist before running vision analysis
        frame_files = list(settings.EXTRACTED_DIR.glob("frame_*.jpg"))
        logger.info(f"[PIPELINE] Found {len(frame_files)} frames for analysis")
        
        if len(frame_files) == 0:
            logger.error(f"[PIPELINE] No frames found in {settings.EXTRACTED_DIR}")
            raise FileNotFoundError(f"No frames found in {settings.EXTRACTED_DIR}")
        
        # Set analysis directory
        analysis_dir = session_root / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        settings.ANALYSIS_DIR = analysis_dir
        
        # Run vision analyzer subprocess with session context
        env = os.environ.copy()
        env["SESSION_ID"] = session_id
        env["EXTRACTED_DIR"] = str(settings.EXTRACTED_DIR)
        env["ANALYSIS_DIR"] = str(analysis_dir)
        env["TELEMETRY_DIR"] = str(telemetry_dir)
        
        logger.info(f"[PIPELINE] Running vision analyzer with SESSION_ID={session_id}")
        
        result = subprocess.run([
            sys.executable, "-m", "src.vision_analyzer"
        ], capture_output=True, text=True, env=env, timeout=600)  # 10 minute timeout
        
        logger.info(f"[PIPELINE] Vision analyzer stdout: {result.stdout[:500]}")
        if result.stderr:
            logger.warning(f"[PIPELINE] Vision analyzer stderr: {result.stderr[:500]}")
        
        if result.returncode != 0:
            raise Exception(f"Vision analysis failed: {result.stderr}")
        
        status_update = get_session_status(session_id) or {}
        status_update.setdefault("processing_steps", []).append("vision_analysis")
        status_update["progress"] = 60
        
        # Step 4: Alert generation
        status_update["current_step"] = "generating_alerts"
        save_session_status(session_id, status_update)
        logger.info(f"[PIPELINE] Generating alerts for session {session_id}")
        
        alerts_dir = session_root / "alerts"
        alerts_dir.mkdir(parents=True, exist_ok=True)
        settings.ALERTS_DIR = alerts_dir
        
        result = subprocess.run([
            sys.executable, "-m", "src.alert_engine"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode != 0:
            raise Exception(f"Alert generation failed: {result.stderr}")
        
        status_update = get_session_status(session_id) or {}
        status_update.setdefault("processing_steps", []).append("alert_generation")
        status_update["progress"] = 80
        
        # Step 5: Person tracking
        status_update["current_step"] = "tracking_persons"
        save_session_status(session_id, status_update)
        logger.info(f"[PIPELINE] Running person tracking for session {session_id}")
        
        # Use existing session_root (parent of extracted_dir)
        session_subdir = session_root / "session"
        session_subdir.mkdir(parents=True, exist_ok=True)
        settings.SESSION_DIR = session_subdir
        
        result = subprocess.run([
            sys.executable, "src/person_tracker.py"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        status_update = get_session_status(session_id) or {}
        status_update.setdefault("processing_steps", []).append("person_tracking")
        status_update["progress"] = 90
        
        # Step 6: Session summary
        status_update["current_step"] = "generating_summary"
        save_session_status(session_id, status_update)
        logger.info(f"Generating session summary for session {session_id}")
        
        result = subprocess.run([
            sys.executable, "-m", "src.summarizer"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode != 0:
            raise Exception(f"Session summary failed: {result.stderr}")
        
        # Restore original settings
        settings.EXTRACTED_DIR = original_extracted_dir
        settings.TELEMETRY_DIR = original_telemetry_dir
        settings.OUTPUTS_DIR = original_outputs_dir
        
        status_update = get_session_status(session_id) or {}
        status_update.setdefault("processing_steps", []).append("session_summary")
        status_update["progress"] = 100
        status_update["status"] = "completed"
        status_update["current_step"] = "completed"
        status_update["completion_time"] = datetime.now().isoformat()
        save_session_status(session_id, status_update)
        
        logger.info(f"Processing completed for session {session_id}")
        
        # Cancel keep-alive heartbeat
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        
    except Exception as e:
        logger.error(f"Processing failed for session {session_id}: {str(e)}")
        status_update = get_session_status(session_id) or {}
        status_update["status"] = "failed"
        status_update["error"] = str(e)
        status_update["current_step"] = "failed"
        save_session_status(session_id, status_update)
        
        # Cancel keep-alive heartbeat on failure
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

@app.get("/frames")
def list_frames():
    """Get frames from the latest extracted folder"""
    # Try to get the latest extracted folder
    latest_folder = get_latest_extracted_folder()
    
    if latest_folder and latest_folder.exists():
        frames = []
        for f in sorted(latest_folder.glob("frame_*.jpg")):
            frames.append(f.name)
        if frames:
            return {"frames": frames, "source": "latest_folder", "count": len(frames)}
    
    # Fallback to original extracted directory
    if settings.EXTRACTED_DIR.exists():
        frames = []
        for f in sorted(settings.EXTRACTED_DIR.glob("frame_*.jpg")):
            frames.append(f.name)
        if frames:
            return {"frames": frames, "source": "extracted_dir", "count": len(frames)}
    
    return {"frames": [], "source": "none", "count": 0}

@app.get("/frames/{frame_id}")
def get_frame_analysis(frame_id: str):
    path = settings.ANALYSIS_DIR / f"{frame_id}_analysis.json"
    if not path.exists():
        raise HTTPException(404, "Frame analysis not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/frames/{frame_id}/alert")
def get_frame_alert(frame_id: str):
    path = settings.ALERTS_DIR / f"{frame_id}_alert.json"
    if not path.exists():
        raise HTTPException(404, "Alert not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/sessions/{session_id}/frames")
def get_session_frames(session_id: str):
    """Get frames for a specific session"""
    logger.info(f"[FRAMES API] Request for session: {session_id}")
    
    try:
        # 1. Check persistent storage first (MongoDB or memory)
        session_status = get_session_status(session_id)
        if session_status:
            stored_frames = session_status.get("extracted_frames", [])
            logger.info(f"[FRAMES API] Found session, extracted_frames: {len(stored_frames)}")
            if stored_frames:
                # Extract just the filename from full paths
                try:
                    frame_names = [Path(str(f)).name for f in stored_frames]
                    logger.info(f"[FRAMES API] Returning {len(frame_names)} frames from storage")
                    source = "mongodb" if mongodb_storage.is_connected() else "memory"
                    return {"frames": frame_names, "source": source, "count": len(frame_names)}
                except Exception as e:
                    logger.error(f"[FRAMES API] Error extracting frame names: {e}")
                    # Return the stored frames as-is if path extraction fails
                    source = "mongodb_raw" if mongodb_storage.is_connected() else "memory_raw"
                    return {"frames": [str(f) for f in stored_frames], "source": source, "count": len(stored_frames)}
        else:
            logger.warning(f"[FRAMES API] Session {session_id} not found in storage")
    except Exception as e:
        logger.error(f"[FRAMES API] Error accessing session storage: {e}")
    
    # 2. Session-specific extracted directory (fallback) - only frame_*.jpg
    session_extracted_dir = Path("data") / "sessions" / session_id / "extracted"
    if session_extracted_dir.exists():
        frames = []
        for f in sorted(session_extracted_dir.glob("frame_*.jpg")):
            frames.append(f.name)
        if frames:
            return {"frames": frames, "source": "session_extracted", "count": len(frames)}
    
    # 3. Session-specific session directory - only frame_*.jpg
    session_session_dir = Path("data") / "sessions" / session_id / "session"
    if session_session_dir.exists():
        frames = []
        for f in sorted(session_session_dir.glob("frame_*.jpg")):
            frames.append(f.name)
        if frames:
            return {"frames": frames, "source": "session_dir", "count": len(frames)}
    
    # 4. Check if frames need to be moved from general extracted to session-specific - only frame_*.jpg
    extracted_dir = Path("data/extracted")
    if extracted_dir.exists():
        frames = []
        for f in sorted(extracted_dir.glob("frame_*.jpg")):
            frames.append(f.name)
        if frames:
            # Move frames to session-specific directory
            session_extracted_dir.mkdir(parents=True, exist_ok=True)
            for frame_file in frames:
                src = extracted_dir / frame_file
                dst = session_extracted_dir / frame_file
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
            return {"frames": frames, "source": "extracted", "count": len(frames)}
    
    return {"frames": [], "source": "none", "count": 0}

@app.get("/sessions/{session_id}/alerts")
def get_session_alerts(session_id: str):
    """Get alerts for a specific session - FILTERED to only show MEDIUM, HIGH, CRITICAL"""
    session_dir = Path("data") / "sessions" / session_id
    alerts_file = session_dir / "alerts.json"
    
    if not alerts_file.exists():
        return {
            "session_date": datetime.now().strftime("%Y-%m-%d"),
            "total_alerts": 0,
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
            "alerts": []
        }
    
    with open(alerts_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # FILTER: Only include MEDIUM, HIGH, CRITICAL alerts for alert center
    # Exclude LOW and CLEAR - they don't warrant immediate attention
    SIGNIFICANT_LEVELS = ['MEDIUM', 'HIGH', 'CRITICAL']
    
    filtered_alerts = [
        a for a in data.get("alerts", []) 
        if a.get("severity", "").upper() in SIGNIFICANT_LEVELS
    ]
    
    # Recalculate counts based on filtered alerts
    high_count = sum(1 for a in filtered_alerts if a.get("severity") == "HIGH")
    medium_count = sum(1 for a in filtered_alerts if a.get("severity") == "MEDIUM")
    critical_count = sum(1 for a in filtered_alerts if a.get("severity") == "CRITICAL")
    
    return {
        "session_date": data.get("session_date", datetime.now().strftime("%Y-%m-%d")),
        "total_alerts": len(filtered_alerts),
        "high_severity": high_count,
        "medium_severity": medium_count,
        "critical_severity": critical_count,
        "low_severity": 0,  # Filtered out
        "alerts": filtered_alerts,
        "filter_applied": "MEDIUM+ only - LOW and CLEAR excluded"
    }

@app.get("/sessions/{session_id}/summary")
def get_session_summary(session_id: str):
    """Get summary for a specific session"""
    session_dir = Path("data") / "sessions" / session_id
    summary_file = session_dir / "session_summary.json"
    
    if not summary_file.exists():
        return {
            "session_summary": {
                "session_date": datetime.now().strftime("%Y-%m-%d"),
                "total_frames_analyzed": 0,
                "total_objects_detected": 0,
                "total_alerts": 0,
                "analysis_duration": "0 minutes",
                "key_events": []
            }
        }
    
    with open(summary_file, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/search")
def semantic_search(payload: Dict[str, Any]):
    query = payload.get("query")
    top_k = payload.get("top_k", 5)
    if not query:
        raise HTTPException(400, "Missing query")
    return search_frames(query, top_k)


@app.post("/qa")
def ask_qa(payload: Dict[str, Any]):
    question = payload.get("question")
    if not question:
        raise HTTPException(400, "Missing question")
    agent = SecurityQAAgent()
    return agent.answer(question)

@app.get("/alerts")
def get_all_alerts():
    """Get all alerts - FILTERED to only show MEDIUM, HIGH, CRITICAL (no LOW or CLEAR)"""
    path = Path("outputs/alerts/all_alerts.json")
    if not path.exists():
        # Return empty alerts structure if file doesn't exist
        return {
            "session_date": datetime.now().strftime("%Y-%m-%d"),
            "total_alerts": 0,
            "high_severity": 0,
            "medium_severity": 0,
            "alerts": []
        }
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # FILTER: Only include MEDIUM, HIGH, CRITICAL alerts
    # Exclude LOW and CLEAR from alert center
    SIGNIFICANT_LEVELS = ['MEDIUM', 'HIGH', 'CRITICAL']
    filtered_alerts = [
        a for a in data.get("alerts", []) 
        if a.get("severity", "").upper() in SIGNIFICANT_LEVELS
    ]
    
    # Update counts
    high_count = sum(1 for a in filtered_alerts if a.get("severity") == "HIGH")
    medium_count = sum(1 for a in filtered_alerts if a.get("severity") == "MEDIUM")
    critical_count = sum(1 for a in filtered_alerts if a.get("severity") == "CRITICAL")
    
    return {
        "session_date": data.get("session_date", datetime.now().strftime("%Y-%m-%d")),
        "total_alerts": len(filtered_alerts),
        "high_severity": high_count,
        "medium_severity": medium_count,
        "critical_severity": critical_count,
        "alerts": filtered_alerts,
        "filter_applied": "MEDIUM+ only - LOW and CLEAR excluded"
    }

@app.get("/alerts/high")
def get_high_alerts():
    path = Path("outputs/alerts/all_alerts.json")
    if not path.exists():
        return {"high_alerts": []}
    with open(path, "r", encoding="utf-8") as f:
        all_alerts = json.load(f)
    high = [a for a in all_alerts.get("alerts", []) if a["severity"] == "HIGH"]
    return {"high_alerts": high}


@app.get("/sessions/{session_id}/frame-image/{frame_name}")
def get_session_frame_image(session_id: str, frame_name: str):
    """Serve a frame image for a specific session from MongoDB GridFS or filesystem"""
    logger.info(f"[FRAME IMAGE] Request for session: {session_id}, frame: {frame_name}")
    
    # Try MongoDB GridFS first
    mongo_storage = get_mongodb_storage()
    if mongo_storage.is_connected():
        image_data = mongo_storage.get_frame_image(session_id, frame_name)
        if image_data:
            logger.info(f"[FRAME IMAGE] Serving from MongoDB GridFS: {frame_name}")
            return Response(content=image_data, media_type="image/jpeg")
    
    # Security: ensure frame_name doesn't contain path traversal
    frame_name = Path(frame_name).name
    
    # Try to find the frame in various locations
    possible_paths = [
        # Session-specific extracted directory
        Path("data") / "sessions" / session_id / "extracted" / frame_name,
        # Direct path from stored frames (with extracted/ subfolder - where frames are actually saved)
        Path("data/extracted2") / "extracted" / frame_name,
        Path("data/extracted1") / "extracted" / frame_name,
        Path("data/extracted") / "extracted" / frame_name,
        Path("data/extracted3") / "extracted" / frame_name,
        Path("data/extracted4") / "extracted" / frame_name,
        Path("data/extracted5") / "extracted" / frame_name,
        # Also check direct paths (fallback)
        Path("data/extracted2") / frame_name,
        Path("data/extracted1") / frame_name,
        Path("data/extracted") / frame_name,
        Path("data/extracted3") / frame_name,
        Path("data/extracted4") / frame_name,
        Path("data/extracted5") / frame_name,
        # Session directory
        Path("data") / "sessions" / session_id / "session" / frame_name,
    ]
    
    # Check persistent storage for stored path
    session_status = get_session_status(session_id)
    if session_status:
        stored_frames = session_status.get("extracted_frames", [])
        logger.info(f"[FRAME IMAGE] Stored frames count: {len(stored_frames)}")
        logger.info(f"[FRAME IMAGE] Looking for: {frame_name}")
        for i, frame_path in enumerate(stored_frames[:3]):  # Log first 3
            logger.info(f"[FRAME IMAGE] Stored[{i}]: {frame_path}")
        for frame_path in stored_frames:
            if frame_name in str(frame_path):
                possible_paths.insert(0, Path(frame_path))
                logger.info(f"[FRAME IMAGE] Found stored path: {frame_path}")
                break
    
    # Try each path
    logger.info(f"[FRAME IMAGE] Checking {len(possible_paths)} possible paths for {frame_name}")
    for i, img_path in enumerate(possible_paths):
        logger.info(f"[FRAME IMAGE] Path {i}: {img_path} - exists: {img_path.exists()}")
        if img_path.exists():
            logger.info(f"[FRAME IMAGE] Found frame at: {img_path}")
            return FileResponse(
                img_path,
                media_type="image/jpeg",
                filename=frame_name
            )
    
    logger.error(f"[FRAME IMAGE] Frame not found: {frame_name}")
    raise HTTPException(404, f"Frame {frame_name} not found")
