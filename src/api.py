"""
api.py — FastAPI backend for Drone Security Analyst Agent.

- Exposes endpoints for health, frames, analysis, alerts, search, session summary, and Q&A
- Video upload and processing capabilities
- Production-ready error handling and logging
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
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

# Global variable to track processing status
processing_status = {}

@app.get("/health")
def health():
    """Health check endpoint with system status."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "system": "drone-security-agent"
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
    
    Supports multiple video formats: .mp4, .avi, .mov, .dav, .mkv, .wmv, .flv
    """
    try:
        # Validate file type
        allowed_extensions = {".mp4", ".avi", ".mov", ".dav", ".mkv", ".wmv", ".flv"}
        
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
        
        # Create sequential extracted folder for this video
        new_extracted_folder_name = get_next_extracted_folder()
        new_extracted_dir = Path("data") / new_extracted_folder_name
        new_extracted_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded video in the extracted folder
        video_path = new_extracted_dir / file.filename
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
        
        # Create minimal session dir for status tracking only
        session_dir = Path("data") / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Video saved to: {new_extracted_folder_name}")
        
        # Initialize processing status
        processing_status[session_id] = {
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
        
        # Start processing in background
        background_tasks.add_task(
            process_video_pipeline,
            session_id,
            str(video_path),
            str(new_extracted_dir),  # Use extracted folder, not session folder
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
    """Get the current processing status for a video session."""
    if session_id not in processing_status:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return processing_status[session_id]

@app.get("/sessions")
def list_sessions():
    """List all processing sessions."""
    sessions = []
    for session_id, status in processing_status.items():
        sessions.append({
            "session_id": session_id,
            "filename": status.get("filename", "unknown"),
            "status": status.get("status", "unknown"),
            "upload_time": status.get("upload_time"),
            "progress": status.get("progress", 0)
        })
    
    return {"sessions": sessions}

async def keep_alive_heartbeat(session_id: str, interval: int = 60):
    """
    Keep-alive heartbeat to prevent Render free tier from sleeping during long operations.
    Logs activity every `interval` seconds to keep the service awake.
    """
    import asyncio
    heartbeat_count = 0
    while processing_status.get(session_id, {}).get("status") == "processing":
        await asyncio.sleep(interval)
        heartbeat_count += 1
        logger.info(f"[{session_id}] Keep-alive heartbeat #{heartbeat_count} - service active, status: {processing_status[session_id].get('current_step', 'unknown')}")
        # Update timestamp to show activity
        processing_status[session_id]["last_heartbeat"] = time.time()


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
        processing_status[session_id]["status"] = "processing"
        processing_status[session_id]["current_step"] = "extracting_frames"
        processing_status[session_id]["start_time"] = time.time()
        
        # Start keep-alive heartbeat to prevent Render sleep (every 60 seconds)
        heartbeat_task = asyncio.create_task(keep_alive_heartbeat(session_id, interval=60))
        logger.info(f"[{session_id}] Started keep-alive heartbeat for long processing")
        
        # Step 1: Extract frames using intelligent extractor
        logger.info(f"Extracting frames for session {session_id} using {extraction_strategy} strategy")
        
        # Update video path in settings for the extractor
        from src.config import settings
        original_video_file = settings.VIDEO_FILE
        settings.VIDEO_FILE = Path(video_path)
        original_extracted_dir = settings.EXTRACTED_DIR
        
        # Use the extracted folder that was already created during upload
        # Find the extracted folder that contains the video file
        video_path_obj = Path(video_path)
        extracted_dir = video_path_obj.parent
        
        # Update settings to use the extracted folder
        settings.EXTRACTED_DIR = extracted_dir
        
        logger.info(f"Using extracted folder: {extracted_dir.name}")
        
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
            
            # Save frame list to processing_status for Railway persistence
            processing_status[session_id]["extracted_frames"] = frames
            processing_status[session_id]["frame_count"] = len(frames)
            processing_status[session_id]["session_dir"] = str(session_dir)
            
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
            
            # Get frames from fallback extraction
            fallback_frames = sorted([str(f) for f in (Path(session_dir) / "extracted").glob("*.jpg")])
            logger.info(f"[FALLBACK] Found {len(fallback_frames)} frames in {session_dir}/extracted")
            processing_status[session_id]["extracted_frames"] = fallback_frames
            processing_status[session_id]["frame_count"] = len(fallback_frames)
            processing_status[session_id]["session_dir"] = str(session_dir)
            logger.info(f"[FALLBACK] Stored {len(fallback_frames)} frames in processing_status")
        
        finally:
            # Restore original settings
            settings.VIDEO_FILE = original_video_file
            settings.EXTRACTED_DIR = original_extracted_dir
        
        processing_status[session_id]["processing_steps"].append("frame_extraction")
        processing_status[session_id]["progress"] = 20
        
        # Step 2: Generate telemetry
        processing_status[session_id]["current_step"] = "generating_telemetry"
        logger.info(f"Generating telemetry for session {session_id}")
        
        # Update config to use session directory
        original_extracted_dir = settings.EXTRACTED_DIR
        original_telemetry_dir = settings.TELEMETRY_DIR
        
        settings.EXTRACTED_DIR = Path(session_dir) / "extracted"
        settings.TELEMETRY_DIR = Path(session_dir) / "telemetry"
        settings.TELEMETRY_DIR.mkdir(exist_ok=True)
        
        result = subprocess.run([
            sys.executable, "-m", "src.telemetry_generator"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Telemetry generation failed: {result.stderr}")
        
        processing_status[session_id]["processing_steps"].append("telemetry_generation")
        processing_status[session_id]["progress"] = 40
        
        # Step 3: Vision analysis
        processing_status[session_id]["current_step"] = "analyzing_frames"
        logger.info(f"Running vision analysis for session {session_id}")
        
        settings.ANALYSIS_DIR = Path(session_dir) / "analysis"
        settings.ANALYSIS_DIR.mkdir(exist_ok=True)
        
        result = subprocess.run([
            sys.executable, "-m", "src.vision_analyzer"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Vision analysis failed: {result.stderr}")
        
        processing_status[session_id]["processing_steps"].append("vision_analysis")
        processing_status[session_id]["progress"] = 60
        
        # Step 4: Alert generation
        processing_status[session_id]["current_step"] = "generating_alerts"
        logger.info(f"Generating alerts for session {session_id}")
        
        settings.ALERTS_DIR = Path(session_dir) / "alerts"
        settings.ALERTS_DIR.mkdir(exist_ok=True)
        
        result = subprocess.run([
            sys.executable, "-m", "src.alert_engine"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Alert generation failed: {result.stderr}")
        
        processing_status[session_id]["processing_steps"].append("alert_generation")
        processing_status[session_id]["progress"] = 80
        
        # Step 5: Person tracking
        processing_status[session_id]["current_step"] = "tracking_persons"
        logger.info(f"Running person tracking for session {session_id}")
        
        settings.SESSION_DIR = Path(session_dir) / "session"
        settings.SESSION_DIR.mkdir(exist_ok=True)
        
        result = subprocess.run([
            sys.executable, "src/person_tracker.py"
        ], capture_output=True, text=True)
        
        processing_status[session_id]["processing_steps"].append("person_tracking")
        processing_status[session_id]["progress"] = 90
        
        # Step 6: Session summary
        processing_status[session_id]["current_step"] = "generating_summary"
        logger.info(f"Generating session summary for session {session_id}")
        
        result = subprocess.run([
            sys.executable, "-m", "src.summarizer"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Session summary failed: {result.stderr}")
        
        # Restore original settings
        settings.EXTRACTED_DIR = original_extracted_dir
        settings.TELEMETRY_DIR = original_telemetry_dir
        
        processing_status[session_id]["processing_steps"].append("session_summary")
        processing_status[session_id]["progress"] = 100
        processing_status[session_id]["status"] = "completed"
        processing_status[session_id]["current_step"] = "completed"
        processing_status[session_id]["completion_time"] = datetime.now().isoformat()
        
        logger.info(f"Processing completed for session {session_id}")
        
        # Cancel keep-alive heartbeat
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        
    except Exception as e:
        logger.error(f"Processing failed for session {session_id}: {str(e)}")
        processing_status[session_id]["status"] = "failed"
        processing_status[session_id]["error"] = str(e)
        processing_status[session_id]["current_step"] = "failed"
        
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
    logger.info(f"[FRAMES API] processing_status keys: {list(processing_status.keys())}")
    
    # 1. Check processing_status first (Railway persistence)
    if session_id in processing_status:
        stored_frames = processing_status[session_id].get("extracted_frames", [])
        logger.info(f"[FRAMES API] Found session, extracted_frames: {len(stored_frames)}")
        if stored_frames:
            # Extract just the filename from full paths
            frame_names = [Path(f).name for f in stored_frames]
            return {"frames": frame_names, "source": "memory", "count": len(frame_names)}
    else:
        logger.warning(f"[FRAMES API] Session {session_id} not found in processing_status")
    
    # 2. Session-specific extracted directory (fallback)
    session_extracted_dir = Path("data") / "sessions" / session_id / "extracted"
    if session_extracted_dir.exists():
        frames = []
        for f in sorted(session_extracted_dir.glob("*.jpg")):
            frames.append(f.name)
        if frames:
            return {"frames": frames, "source": "session_extracted", "count": len(frames)}
    
    # 3. Session-specific session directory
    session_session_dir = Path("data") / "sessions" / session_id / "session"
    if session_session_dir.exists():
        frames = []
        for f in sorted(session_session_dir.glob("*.jpg")):
            frames.append(f.name)
        if frames:
            return {"frames": frames, "source": "session_dir", "count": len(frames)}
    
    # 4. Check if frames need to be moved from general extracted to session-specific
    extracted_dir = Path("data/extracted")
    if extracted_dir.exists():
        frames = []
        for f in sorted(extracted_dir.glob("*.jpg")):
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
    """Get alerts for a specific session"""
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
        return json.load(f)

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
    path = Path("outputs/alerts/all_alerts.json")
    if not path.exists():
        # Return empty alerts structure if file doesn't exist
        return {
            "session_date": datetime.now().strftime("%Y-%m-%d"),
            "total_alerts": 0,
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
            "alerts": []
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/alerts/high")
def get_high_alerts():
    path = Path("outputs/alerts/all_alerts.json")
    if not path.exists():
        return {"high_alerts": []}
    with open(path, "r", encoding="utf-8") as f:
        all_alerts = json.load(f)
    high = [a for a in all_alerts.get("alerts", []) if a["severity"] == "HIGH"]
    return {"high_alerts": high}
