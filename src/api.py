"""
api.py — FastAPI backend for Drone Security Analyst Agent.

- Exposes endpoints for health, frames, analysis, alerts, search, session summary, and Q&A
- Video upload and processing capabilities
- Production-ready error handling and logging
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
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
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=15)
        if result.returncode == 0:
            version = result.stdout.decode().split('\n')[0]
            logger.info(f"FFmpeg available: {version[:60]}")
            return True
        else:
            logger.error("FFmpeg not working properly")
            return False
    except Exception as e:
        logger.error(f"FFmpeg not found: {e}")
        return False

FFMPEG_AVAILABLE = check_ffmpeg()

def ffmpeg_available() -> bool:
    """Return ffmpeg availability, re-probing if the boot-time check failed.

    The startup probe can time out on a cold-started machine, and a stale
    False here would permanently block video processing.
    """
    global FFMPEG_AVAILABLE
    if not FFMPEG_AVAILABLE:
        FFMPEG_AVAILABLE = check_ffmpeg()
    return FFMPEG_AVAILABLE

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

# Compress JSON/static responses (big win for session/frame list payloads)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def add_cache_headers(request, call_next):
    """Long-cache immutable assets: Vite-hashed bundles and per-session frame images."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/assets/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif "/frame-image/" in path and response.status_code == 200:
        response.headers["Cache-Control"] = "public, max-age=86400"
    return response

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
            "search": "/search",
            "qa": "/qa",
            "alerts": "/alerts",
            "session_summary": "/session-summary",
        },
        "features": {
            "semantic_search": True,
            "security_qa_agent": True,
            "robust_frame_preprocess": bool(settings.ROBUST_PREPROCESS),
            "cloud_analyzer": os.getenv("USE_CLOUD_ANALYZER", str(settings.USE_CLOUD_ANALYZER)).lower() in ("1", "true", "yes"),
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
        "ffmpeg_available": ffmpeg_available(),
        "mongodb_connected": mongodb_storage.is_connected(),
        "persistence": "mongodb" if mongodb_storage.is_connected() else "memory-only"
    }


@app.get("/debug")
def debug_status():
    """Debug endpoint — checks all API keys, quotas, and system components."""
    checks = []
    
    # 1. Groq API Key
    groq_key = settings.GROQ_API_KEY
    groq_status = "not_configured"
    groq_detail = ""
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            # Light test — just check auth with minimal tokens
            r = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5
            )
            groq_status = "ok"
            groq_detail = "Authenticated and responding"
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err:
                groq_status = "quota_exhausted"
                groq_detail = "Daily token limit reached. Resets in ~24h."
            elif "401" in err or "invalid" in err.lower():
                groq_status = "invalid_key"
                groq_detail = "API key is invalid or expired"
            else:
                groq_status = "error"
                groq_detail = err[:150]
    checks.append({"name": "Groq API (Vision)", "status": groq_status, "detail": groq_detail, "provider": "groq"})
    
    # 2. NVIDIA NIM
    nvidia_key = getattr(settings, 'NVIDIA_API_KEY', '')
    nvidia_status = "not_configured"
    nvidia_detail = ""
    if nvidia_key:
        try:
            from openai import OpenAI
            nvidia_model = getattr(settings, 'NVIDIA_MODEL', 'meta/llama-3.3-70b-instruct')
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_key)
            r = client.chat.completions.create(
                model=nvidia_model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5
            )
            nvidia_status = "ok"
            nvidia_detail = f"Model: {nvidia_model}"
        except Exception as e:
            err = str(e)
            if "503" in err or "DEGRADED" in err:
                nvidia_status = "degraded"
                nvidia_detail = "Model temporarily unavailable on NVIDIA servers"
            elif "400" in err:
                nvidia_status = "degraded"
                nvidia_detail = "Model in degraded state — try again later"
            elif "401" in err:
                nvidia_status = "invalid_key"
                nvidia_detail = "API key is invalid"
            else:
                nvidia_status = "error"
                nvidia_detail = err[:150]
    checks.append({"name": "NVIDIA NIM (Orchestration)", "status": nvidia_status, "detail": nvidia_detail, "provider": "nvidia"})
    
    # 3. Gemini
    gemini_key = settings.GEMINI_API_KEY
    gemini_status = "not_configured"
    gemini_detail = ""
    if gemini_key:
        gemini_status = "configured"
        gemini_detail = f"Model: {settings.GEMINI_MODEL}, Keys: {1 + (1 if settings.GEMINI_API_KEY_2 else 0)}"
    checks.append({"name": "Gemini (Fallback)", "status": gemini_status, "detail": gemini_detail, "provider": "gemini"})
    
    # 4. Pinecone
    pinecone_status = "not_configured"
    pinecone_detail = ""
    if settings.PINECONE_API_KEY:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            idx = pc.describe_index(settings.PINECONE_INDEX_NAME)
            pinecone_status = "ok"
            pinecone_detail = f"Index: {settings.PINECONE_INDEX_NAME}, Dimension: {settings.PINECONE_DIMENSION}"
        except Exception as e:
            pinecone_status = "error"
            pinecone_detail = str(e)[:150]
    checks.append({"name": "Pinecone (Vector Search)", "status": pinecone_status, "detail": pinecone_detail, "provider": "pinecone"})
    
    # 5. MongoDB
    mongo_status = "ok" if mongodb_storage.is_connected() else "disconnected"
    checks.append({"name": "MongoDB (Sessions)", "status": mongo_status, "detail": "Connected" if mongo_status == "ok" else "Not connected", "provider": "mongodb"})
    
    # 6. FFmpeg
    ffmpeg_ok = ffmpeg_available()
    checks.append({"name": "FFmpeg (Video Processing)", "status": "ok" if ffmpeg_ok else "missing", "detail": "Installed" if ffmpeg_ok else "Not found in PATH", "provider": "system"})
    
    # Overall status
    critical_issues = [c for c in checks if c["status"] in ["error", "invalid_key", "missing"]]
    warnings = [c for c in checks if c["status"] in ["quota_exhausted", "degraded", "disconnected"]]
    
    overall = "healthy"
    if critical_issues:
        overall = "critical"
    elif warnings:
        overall = "degraded"
    
    return {
        "overall": overall,
        "timestamp": datetime.now().isoformat(),
        "checks": checks,
        "config": {
            "vision_provider": os.environ.get("VISION_PROVIDER", settings.VISION_PROVIDER if hasattr(settings, 'VISION_PROVIDER') else "unknown"),
            "agent_llm_provider": os.environ.get("AGENT_LLM_PROVIDER", getattr(settings, 'AGENT_LLM_PROVIDER', 'unknown')),
            "max_frames": settings.MAX_FRAMES,
            "max_vision_workers": int(os.environ.get("MAX_VISION_WORKERS", "2")),
        },
        "tips": {
            "quota_exhausted": "Groq free tier: 500K tokens/day. Wait 24h or use a different email account.",
            "degraded": "NVIDIA NIM free models go down occasionally. System falls back to Groq automatically.",
            "vision_slow": "Reduce MAX_FRAMES in .env or use the slider (5-20 frames).",
        }
    }


@app.post("/debug/clear-sessions")
def debug_clear_sessions():
    """Clear all sessions from MongoDB and disk."""
    import shutil
    # Clear MongoDB
    deleted = 0
    if mongodb_storage.is_connected():
        try:
            deleted = mongodb_storage.sessions_collection.delete_many({}).deleted_count
        except Exception:
            pass
    # Clear disk
    sessions_dir = Path("data/sessions")
    if sessions_dir.exists():
        for d in sessions_dir.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
    # Clear memory
    processing_status.clear()
    return {"message": f"Cleared {deleted} sessions from MongoDB and all session data from disk."}


@app.post("/upload-sample-video")
async def upload_sample_video(
    background_tasks: BackgroundTasks,
    extraction_strategy: str = "hybrid",
    max_frames: int = 30,
    use_cloud_enhancers: bool = False
):
    """Upload the bundled sample video for demo/tour purposes."""
    import shutil
    
    sample_path = Path("sample-video.mp4")
    if not sample_path.exists():
        # Try the original filename
        sample_path = Path("Sneaky Thieves Caught Stealing Phones On Camera - Newsflare (1080p, h264) (1).mp4")
    if not sample_path.exists():
        raise HTTPException(404, "Sample video not found on server")
    
    session_id = str(uuid.uuid4())
    session_dir = Path("data") / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy sample video to session directory
    video_dest = session_dir / sample_path.name
    shutil.copy2(str(sample_path), str(video_dest))
    
    extracted_dir = session_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    
    # Save initial session status
    status_data = {
        "session_id": session_id,
        "filename": sample_path.name,
        "status": "processing",
        "current_step": "initializing",
        "progress": 0,
        "upload_time": datetime.now().isoformat(),
        "extraction_strategy": extraction_strategy,
        "max_frames": max_frames,
        "extracted_frames": [],
        "frame_count": 0,
        "session_dir": str(extracted_dir),
    }
    save_session_status(session_id, status_data)
    
    logger.info(f"Sample video session created: {session_id}")
    
    # Start pipeline in background
    background_tasks.add_task(
        process_video_pipeline,
        session_id=session_id,
        video_path=str(video_dest),
        session_dir=str(extracted_dir),
        extraction_strategy=extraction_strategy,
        max_frames=max_frames,
        use_cloud_enhancers=use_cloud_enhancers,
    )
    
    return {
        "session_id": session_id,
        "filename": sample_path.name,
        "status": "processing",
    }


@app.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    extraction_strategy: str = "hybrid",
    max_frames: int = 30,
    use_cloud_enhancers: bool = False
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
            max_frames,
            use_cloud_enhancers
        )
        
        logger.info(f"Video uploaded: {file.filename} (Session: {session_id})")
        
        return {
            "session_id": session_id,
            "message": f"Video uploaded successfully. Processing started with {extraction_strategy} strategy.",
            "filename": file.filename,
            "extraction_strategy": extraction_strategy,
            "max_frames": max_frames,
            "estimated_time": "30-90 seconds"
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

@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str):
    """
    Cancel a running session pipeline (if active), delete the session metadata
    from database, and delete all associated files and results from disk.
    """
    logger.info(f"Received request to delete/cancel session: {session_id}")
    
    # 1. Mark as cancelled in memory so running threads terminate immediately
    from src.cancellation import cancel_session
    cancel_session(session_id)

    # 1b. If this is an active live capture, stop its stream/ingestion first
    from src.live_capture import get_live_session, remove_live_session
    live = get_live_session(session_id)
    if live:
        live.stop()
        remove_live_session(session_id)
        logger.info(f"Stopped active live capture for session {session_id}")
    
    # 2. Try to remove metadata from MongoDB / local memory cache
    db_deleted = False
    if mongodb_storage.is_connected():
        try:
            db_deleted = mongodb_storage.delete_session(session_id)
        except Exception as e:
            logger.error(f"Failed to delete session {session_id} from MongoDB: {e}")
            
    if session_id in processing_status:
        processing_status.pop(session_id)
        
    # 3. Delete session directory from disk
    import shutil
    session_dir = Path("data/sessions") / session_id
    disk_deleted = False
    
    if session_dir.exists():
        try:
            shutil.rmtree(session_dir, ignore_errors=True)
            disk_deleted = True
            logger.info(f"Deleted directory for session {session_id} from disk")
        except Exception as e:
            logger.error(f"Could not immediately delete directory for session {session_id}: {e}")

    return {
        "session_id": session_id,
        "message": "Session cancellation and deletion request processed.",
        "db_deleted": db_deleted,
        "disk_deleted": disk_deleted
    }

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


# Pipeline stages that historically ran as separate Python subprocesses.
# Running them in-process avoids 5x interpreter cold-starts + re-imports + disk IPC,
# which is the single biggest production-level speedup for the pipeline.
_PIPELINE_STAGES = ("telemetry", "vision", "alerts", "tracking", "summary")

# Pipeline stages mutate shared global settings (via apply_session_layout), so concurrent
# runs would race and read/write each other's session directories. This lock serializes
# whole-pipeline execution to keep each run isolated.
import threading
_pipeline_lock = threading.Lock()


def _run_stage_in_process(stage: str, session_id: str) -> None:
    """Run a single pipeline stage in the current process.

    Points the shared settings at the session's directory layout, then calls the
    stage function directly instead of spawning `python -m src.<module>`.
    Raises on failure so the caller can fall back to the subprocess path.
    """
    from src.session_bootstrap import apply_session_layout

    # Make settings.* resolve to data/sessions/{session_id}/... for this stage.
    apply_session_layout(session_id)

    if stage == "telemetry":
        from src.telemetry_generator import generate_telemetry
        meta_path = settings.OUTPUTS_DIR / "extraction_log.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            frame_meta = json.load(f)["frames"]
        generate_telemetry(frame_meta)
    elif stage == "vision":
        from src.vision_analyzer import analyze_all_frames
        analyze_all_frames()
    elif stage == "alerts":
        from src.alert_engine import process_alerts
        process_alerts()
    elif stage == "tracking":
        from src.person_tracker import process_all_frames
        process_all_frames()
    elif stage == "summary":
        from src.summarizer import generate_session_summary
        generate_session_summary()
    else:
        raise ValueError(f"Unknown pipeline stage: {stage}")


def _run_stage(session_id: str, stage: str, subprocess_args: list, *, env=None, timeout: int = 300) -> None:
    """Run a pipeline stage in-process by default, falling back to a subprocess.

    Set PIPELINE_MODE=subprocess to force the legacy subprocess behavior.
    If the in-process call fails for any reason, the subprocess path is used so
    the pipeline degrades gracefully instead of failing outright.
    """
    import subprocess

    mode = os.environ.get("PIPELINE_MODE", getattr(settings, "PIPELINE_MODE", "in_process")).lower()

    if mode != "subprocess":
        try:
            stage_start = time.time()
            _run_stage_in_process(stage, session_id)
            logger.info(f"[PIPELINE] In-process '{stage}' completed in {int((time.time() - stage_start) * 1000)}ms")
            return
        except Exception as exc:
            logger.warning(f"[PIPELINE] In-process '{stage}' failed ({exc}); falling back to subprocess")

    # Legacy subprocess fallback
    sub_env = dict(env) if env else os.environ.copy()
    sub_env.setdefault("PYTHONIOENCODING", "utf-8")
    sub_env.setdefault("PYTHONUTF8", "1")
    result = subprocess.run(
        subprocess_args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=sub_env,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise Exception(f"{stage} stage failed: {result.stderr}")


async def process_video_pipeline(session_id: str, video_path: str, session_dir: str, extraction_strategy: str = "hybrid", max_frames: int = 100, use_cloud_enhancers: bool = False):
    """
    Async wrapper: runs the heavy, blocking pipeline in a worker thread so the API
    event loop stays responsive (status polls, heartbeat) while processing runs.
    """
    import asyncio

    # Keep-alive heartbeat runs on the event loop while the pipeline runs in a thread.
    heartbeat_task = asyncio.create_task(keep_alive_heartbeat(session_id, interval=60))
    logger.info(f"[{session_id}] Started keep-alive heartbeat for long processing")
    try:
        await asyncio.to_thread(
            _run_pipeline_sync, session_id, video_path, session_dir, extraction_strategy, max_frames, use_cloud_enhancers
        )
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


def _run_pipeline_sync(session_id: str, video_path: str, session_dir: str, extraction_strategy: str = "hybrid", max_frames: int = 100, use_cloud_enhancers: bool = False):
    """
    Runs the pipeline body under a global lock. Pipeline stages mutate shared global
    settings (apply_session_layout), so concurrent runs must be serialized to avoid one
    session reading/writing another session's directories.
    """
    with _pipeline_lock:
        # Per-run toggle for the HF CLIP+BLIP cross-check layer. Safe because runs
        # are serialized under the lock; vision_analyzer reads this env at call time.
        os.environ["USE_CLOUD_ANALYZER"] = "true" if use_cloud_enhancers else "false"
        _run_pipeline_body(session_id, video_path, session_dir, extraction_strategy, max_frames)


def _run_pipeline_body(session_id: str, video_path: str, session_dir: str, extraction_strategy: str = "hybrid", max_frames: int = 100):
    """Synchronous pipeline body (runs in a worker thread). Executes all stages in-process."""
    from src.cancellation import is_cancelled, clear_cancellation
    if is_cancelled(session_id):
        raise RuntimeError("Session cancelled by user request.")
    try:
        import sys
        import subprocess
        
        # Update status
        status_update = get_session_status(session_id) or {}
        status_update["status"] = "processing"
        status_update["current_step"] = "extracting_frames"
        status_update["start_time"] = time.time()
        save_session_status(session_id, status_update)
        
        # Step 1: Check ffmpeg availability
        if not ffmpeg_available():
            logger.error(f"[{session_id}] FFmpeg not available - cannot extract frames!")
            status_update = get_session_status(session_id) or {}
            status_update["status"] = "failed"
            status_update["error"] = "FFmpeg not installed"
            save_session_status(session_id, status_update)
            return
        
        logger.info(f"[{session_id}] FFmpeg available, starting frame extraction")
        
        # Step 2: Extract frames using intelligent extractor
        logger.info(f"[PIPELINE] Starting frame extraction for session {session_id}")
        
        # Update video path in settings for the extractor
        from src.config import settings
        original_video_file = settings.VIDEO_FILE
        # Snapshot session-scoped dirs so global settings can be restored after the
        # in-process stages mutate them via apply_session_layout().
        original_session_dir = settings.SESSION_DIR
        original_analysis_dir = settings.ANALYSIS_DIR
        original_alerts_dir = settings.ALERTS_DIR
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
            
            # Verify extraction_log.json was created
            extraction_log_path = settings.OUTPUTS_DIR / "extraction_log.json"
            if not extraction_log_path.exists():
                # Create it from the returned frame data
                logger.warning(f"[EXTRACTION] extraction_log.json not found at {extraction_log_path}, creating it")
                frame_list = []
                for idx, f in enumerate(frames):
                    if isinstance(f, dict):
                        frame_list.append({
                            "filename": f.get("filename", ""),
                            "frame_id": Path(f.get("filename", "")).stem,
                            "timestamp_seconds": f.get("timestamp_seconds", idx * (settings.VIDEO_DURATION_SECONDS / max(len(frames), 1)))
                        })
                log_data = {
                    "total_frames": len(frame_list),
                    "total_frames_extracted": len(frame_list),
                    "video_duration_seconds": settings.VIDEO_DURATION_SECONDS,
                    "fps": settings.VIDEO_FPS,
                    "resolution": "1920x1080",
                    "codec": "hevc",
                    "frames": frame_list
                }
                extraction_log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
                logger.info(f"[EXTRACTION] Created extraction_log.json with {len(frame_list)} frames")
            
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
            extraction_output_dir = session_dir
            logger.info(f"[EXTRACTION] Running frame_extractor with output: {extraction_output_dir}")
            result = subprocess.run([
                sys.executable, "-m", "src.frame_extractor",
                "--input", video_path,
                "--output", extraction_output_dir
            ], capture_output=True, text=True,
               env={**os.environ, "SESSION_ID": session_id, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
            
            logger.info(f"[EXTRACTION] Return code: {result.returncode}")
            logger.info(f"[EXTRACTION] stdout: {result.stdout[:500] if result.stdout else 'empty'}")
            logger.info(f"[EXTRACTION] stderr: {result.stderr[:500] if result.stderr else 'empty'}")
            
            if result.returncode != 0:
                raise Exception(f"Frame extraction failed: {result.stderr}")
            
            # Get frames from fallback extraction (only frame_*.jpg, not temp files)
            fallback_frames = sorted(list(Path(session_dir).glob("frame_*.jpg")))
            logger.info(f"[FALLBACK] Found {len(fallback_frames)} frames in {session_dir}")
            
            # Write extraction_log.json to session outputs directory
            session_root = Path(session_dir).parent
            session_outputs = session_root / "outputs"
            session_outputs.mkdir(parents=True, exist_ok=True)
            extraction_log = {
                "total_frames": len(fallback_frames),
                "total_frames_extracted": len(fallback_frames),
                "video_duration_seconds": settings.VIDEO_DURATION_SECONDS,
                "fps": settings.VIDEO_FPS,
                "resolution": "1920x1080",
                "codec": "hevc",
                "frames": [
                    {"filename": f.name, "frame_id": f.stem, "timestamp_seconds": i * (settings.VIDEO_DURATION_SECONDS / max(len(fallback_frames), 1))}
                    for i, f in enumerate(fallback_frames)
                ]
            }
            log_path = session_outputs / "extraction_log.json"
            log_path.write_text(json.dumps(extraction_log, indent=2), encoding="utf-8")
            logger.info(f"[FALLBACK] Wrote extraction_log.json to {log_path}")
            
            frame_paths = [str(f) for f in fallback_frames]
            status_update = get_session_status(session_id) or {}
            status_update["extracted_frames"] = frame_paths
            status_update["frame_count"] = len(frame_paths)
            status_update["session_dir"] = str(session_dir)
            save_session_status(session_id, status_update)
            logger.info(f"[FALLBACK] Stored {len(frame_paths)} frames in processing_status")
        
        finally:
            # Restore original settings
            settings.VIDEO_FILE = original_video_file
            settings.EXTRACTED_DIR = original_extracted_dir
            settings.OUTPUTS_DIR = original_outputs_dir
        
        status_update = get_session_status(session_id) or {}
        status_update.setdefault("processing_steps", []).append("frame_extraction")
        status_update["progress"] = 20
        from src.cancellation import is_cancelled
        if is_cancelled(session_id):
            raise RuntimeError("Session cancelled by user request.")
        
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
        
        _run_stage(session_id, "telemetry", [
            sys.executable, "-m", "src.telemetry_generator"
        ], env=env, timeout=300)
        
        status_update = get_session_status(session_id) or {}
        status_update.setdefault("processing_steps", []).append("telemetry_generation")
        status_update["progress"] = 40
        if is_cancelled(session_id):
            raise RuntimeError("Session cancelled by user request.")
            
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
        
        _run_stage(session_id, "vision", [
            sys.executable, "-m", "src.vision_analyzer"
        ], env=env, timeout=600)
        
        status_update = get_session_status(session_id) or {}
        status_update.setdefault("processing_steps", []).append("vision_analysis")
        status_update["progress"] = 60
        if is_cancelled(session_id):
            raise RuntimeError("Session cancelled by user request.")
            
        # Step 3b: Index frames in Pinecone for semantic search
        status_update["current_step"] = "indexing_frames"
        save_session_status(session_id, status_update)
        logger.info(f"[PIPELINE] Indexing frames in Pinecone for session {session_id}")
        try:
            from src.pinecone_indexer import index_frames
            os.environ["SESSION_ID"] = session_id
            settings.SESSION_ID = session_id
            from src.session_bootstrap import apply_session_layout
            apply_session_layout(session_id)
            index_frames()
            logger.info(f"[PIPELINE] Pinecone indexing completed for session {session_id}")
        except Exception as exc:
            logger.warning(f"[PIPELINE] Pinecone indexing failed (non-fatal): {exc}")
        if is_cancelled(session_id):
            raise RuntimeError("Session cancelled by user request.")
            
        # Step 4: Alert generation
        status_update["current_step"] = "generating_alerts"
        save_session_status(session_id, status_update)
        logger.info(f"[PIPELINE] Generating alerts for session {session_id}")
        
        alerts_dir = session_root / "alerts"
        alerts_dir.mkdir(parents=True, exist_ok=True)
        settings.ALERTS_DIR = alerts_dir
        
        _run_stage(session_id, "alerts", [
            sys.executable, "-m", "src.alert_engine"
        ], timeout=300)
        
        status_update = get_session_status(session_id) or {}
        status_update.setdefault("processing_steps", []).append("alert_generation")
        status_update["progress"] = 80
        if is_cancelled(session_id):
            raise RuntimeError("Session cancelled by user request.")
            
        # Step 5: Person tracking
        status_update["current_step"] = "tracking_persons"
        save_session_status(session_id, status_update)
        logger.info(f"[PIPELINE] Running person tracking for session {session_id}")
        
        # Use existing session_root (parent of extracted_dir)
        session_subdir = session_root / "session"
        session_subdir.mkdir(parents=True, exist_ok=True)
        settings.SESSION_DIR = session_subdir
        
        # Person tracking is non-fatal — log and continue if it fails.
        try:
            _run_stage(session_id, "tracking", [
                sys.executable, "src/person_tracker.py"
            ], timeout=300)
        except Exception as exc:
            logger.warning(f"[PIPELINE] Person tracking failed (non-fatal): {exc}")
        
        status_update = get_session_status(session_id) or {}
        status_update.setdefault("processing_steps", []).append("person_tracking")
        status_update["progress"] = 90
        if is_cancelled(session_id):
            raise RuntimeError("Session cancelled by user request.")
            
        # Step 6: Session summary
        status_update["current_step"] = "generating_summary"
        save_session_status(session_id, status_update)
        logger.info(f"Generating session summary for session {session_id}")
        
        _run_stage(session_id, "summary", [
            sys.executable, "-m", "src.summarizer"
        ], timeout=300)
        
        # Restore original settings
        settings.EXTRACTED_DIR = original_extracted_dir
        settings.TELEMETRY_DIR = original_telemetry_dir
        settings.OUTPUTS_DIR = original_outputs_dir
        settings.SESSION_DIR = original_session_dir
        settings.ANALYSIS_DIR = original_analysis_dir
        settings.ALERTS_DIR = original_alerts_dir
        # Reset SESSION_ID so it doesn't leak into the next upload/request
        settings.SESSION_ID = ""
        os.environ.pop("SESSION_ID", None)
        
        status_update = get_session_status(session_id) or {}
        status_update.setdefault("processing_steps", []).append("session_summary")
        status_update["progress"] = 100
        status_update["status"] = "completed"
        status_update["current_step"] = "completed"
        status_update["completion_time"] = datetime.now().isoformat()
        save_session_status(session_id, status_update)
        
        logger.info(f"Processing completed for session {session_id}")
        
    except Exception as e:
        error_msg = str(e)
        from src.cancellation import is_cancelled, clear_cancellation
        if is_cancelled(session_id) or "cancelled by user request" in error_msg:
            logger.info(f"Pipeline caught cancellation for session {session_id}. Cleaning up and exiting.")
            # Restore original settings
            settings.EXTRACTED_DIR = original_extracted_dir
            settings.TELEMETRY_DIR = original_telemetry_dir
            settings.OUTPUTS_DIR = original_outputs_dir
            settings.SESSION_DIR = original_session_dir
            settings.ANALYSIS_DIR = original_analysis_dir
            settings.ALERTS_DIR = original_alerts_dir
            settings.SESSION_ID = ""
            os.environ.pop("SESSION_ID", None)
            
            # Clean directory from disk
            import shutil
            session_root = Path("data/sessions") / session_id
            shutil.rmtree(session_root, ignore_errors=True)
            clear_cancellation(session_id)
            return
            
        # Detect common issues and provide user-friendly messages
        if "429" in error_msg or "rate_limit" in error_msg or "tokens per day" in error_msg:
            user_error = "API quota exhausted (Groq daily limit). Please wait 10-15 minutes or check Debug panel for status."
        elif "503" in error_msg or "DEGRADED" in error_msg:
            user_error = "AI model temporarily unavailable (server-side issue). Please retry in a few minutes."
        elif "401" in error_msg or "invalid" in error_msg.lower():
            user_error = "API key invalid or expired. Check your .env configuration."
        elif "timeout" in error_msg.lower():
            user_error = "Processing timed out. Try with fewer frames (reduce slider to 5-10)."
        else:
            user_error = error_msg[:200]
        
        logger.error(f"Processing failed for session {session_id}: {error_msg}")
        status_update = get_session_status(session_id) or {}
        status_update["status"] = "failed"
        status_update["error"] = user_error
        status_update["error_detail"] = error_msg[:500]
        status_update["current_step"] = "failed"
        save_session_status(session_id, status_update)

# ---------------------------------------------------------------------------
# Live capture: drone / mobile camera feeds
# ---------------------------------------------------------------------------

@app.post("/live/start")
def start_live_capture(payload: Optional[Dict[str, Any]] = None):
    """
    Start a live capture session.

    Body:
        source: "browser" (phone/laptop camera pushes frames) or
                "stream" (server pulls a drone/IP-camera RTSP/RTMP/HTTP URL)
        stream_url: required when source == "stream"
        capture_interval: seconds between saved frames (stream mode, default 2)
        max_frames: stop capturing after this many frames (default 60)
    """
    from src.live_capture import start_live_session

    payload = payload or {}
    source = payload.get("source", "browser")
    if source not in ("browser", "stream"):
        raise HTTPException(400, "source must be 'browser' or 'stream'")

    stream_url = (payload.get("stream_url") or "").strip()
    if source == "stream" and not stream_url:
        raise HTTPException(400, "stream_url is required for stream source (e.g. rtsp://... from your drone)")

    max_frames = max(5, min(500, int(payload.get("max_frames", 60))))
    capture_interval = max(0.5, min(60.0, float(payload.get("capture_interval", 2.0))))

    session_id = str(uuid.uuid4())
    session_dir = Path("data") / "sessions" / session_id
    extracted_dir = session_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    try:
        start_live_session(
            session_id,
            extracted_dir,
            source,
            max_frames=max_frames,
            stream_url=stream_url or None,
            capture_interval=capture_interval,
        )
    except Exception as e:
        raise HTTPException(400, f"Could not start live capture: {e}")

    label = "Live Capture (camera)" if source == "browser" else f"Live Stream ({stream_url[:60]})"
    save_session_status(session_id, {
        "session_id": session_id,
        "filename": label,
        "status": "live",
        "current_step": "capturing",
        "progress": 0,
        "upload_time": datetime.now().isoformat(),
        "source": f"live-{source}",
        "max_frames": max_frames,
        "capture_interval": capture_interval,
        "frame_count": 0,
        "extracted_frames": [],
        "session_dir": str(extracted_dir),
    })
    logger.info(f"[LIVE] Started {source} capture session {session_id} (max {max_frames} frames)")

    return {
        "session_id": session_id,
        "status": "live",
        "source": source,
        "max_frames": max_frames,
        "capture_interval": capture_interval,
    }


@app.post("/live/{session_id}/frame")
async def push_live_frame(session_id: str, file: UploadFile = File(...)):
    """Receive one camera frame (JPEG) from the browser during a live session."""
    from src.live_capture import get_live_session

    live = get_live_session(session_id)
    if not live:
        raise HTTPException(404, "No active live session with this id (already stopped?)")
    if live.source != "browser":
        raise HTTPException(400, "This live session captures from a stream URL; frames cannot be pushed")
    if live.is_full():
        return {"accepted": False, "reason": "max_frames reached", "frame_count": live.frame_count, "full": True}

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty frame")

    entry = live.add_frame_bytes(data)
    status_update = get_session_status(session_id) or {}
    status_update["frame_count"] = live.frame_count
    save_session_status(session_id, status_update)

    return {"accepted": True, "frame_count": live.frame_count, "full": live.is_full(), **entry}


@app.get("/live/{session_id}/status")
def live_capture_status(session_id: str):
    """Poll live capture progress (frame count, stream errors, capacity)."""
    from src.live_capture import get_live_session

    live = get_live_session(session_id)
    if not live:
        # Session may already be stopped and analyzing — report from stored status
        stored = get_session_status(session_id)
        if stored:
            return {"session_id": session_id, "capturing": False, "frame_count": stored.get("frame_count", 0),
                    "status": stored.get("status"), "error": stored.get("error")}
        raise HTTPException(404, "Live session not found")

    return {
        "session_id": session_id,
        "capturing": True,
        "source": live.source,
        "frame_count": live.frame_count,
        "max_frames": live.max_frames,
        "full": live.is_full(),
        "elapsed_seconds": round(time.time() - live.start_time, 1),
        "error": live.error,
    }


@app.post("/live/{session_id}/stop")
async def stop_live_capture(session_id: str, background_tasks: BackgroundTasks, analyze: bool = True):
    """Stop a live capture and run the standard analysis pipeline on the captured frames."""
    from src.live_capture import get_live_session, remove_live_session

    live = get_live_session(session_id)
    if not live:
        raise HTTPException(404, "No active live session with this id")

    live.stop()
    remove_live_session(session_id)
    logger.info(f"[LIVE] Stopped session {session_id} with {live.frame_count} frames")

    status_update = get_session_status(session_id) or {}

    if live.frame_count == 0:
        status_update["status"] = "failed"
        status_update["error"] = live.error or "No frames were captured"
        status_update["current_step"] = "failed"
        save_session_status(session_id, status_update)
        return {"session_id": session_id, "frame_count": 0, "analysis_started": False,
                "error": status_update["error"]}

    # Produce the same artifacts an uploaded video would have at this point
    session_root = Path("data") / "sessions" / session_id
    live.write_extraction_log(session_root / "outputs")
    frame_paths = [str(live.extracted_dir / f["filename"]) for f in live.frames]

    status_update["extracted_frames"] = frame_paths
    status_update["frame_count"] = live.frame_count
    status_update["status"] = "processing" if analyze else "captured"
    status_update["current_step"] = "queued" if analyze else "captured"
    status_update["progress"] = 10 if analyze else 0
    if live.error:
        status_update["capture_warning"] = live.error
    save_session_status(session_id, status_update)

    if analyze:
        background_tasks.add_task(process_live_pipeline, session_id)

    return {
        "session_id": session_id,
        "frame_count": live.frame_count,
        "analysis_started": analyze,
        "capture_warning": live.error,
    }


async def process_live_pipeline(session_id: str):
    """Async wrapper for live-session analysis, mirroring process_video_pipeline."""
    import asyncio

    heartbeat_task = asyncio.create_task(keep_alive_heartbeat(session_id, interval=60))
    try:
        await asyncio.to_thread(_run_live_analysis_sync, session_id)
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


def _run_live_analysis_sync(session_id: str):
    with _pipeline_lock:
        _run_live_analysis_body(session_id)


def _run_live_analysis_body(session_id: str):
    """
    Run the analysis stages (telemetry → vision → index → alerts → tracking → summary)
    on frames already captured live. Frame extraction is skipped: the live session
    already wrote frame_*.jpg and extraction_log.json in the standard session layout.
    """
    import sys
    from src.cancellation import is_cancelled, clear_cancellation

    def check_cancelled():
        if is_cancelled(session_id):
            raise RuntimeError("Session cancelled by user request.")

    # Stages mutate global settings via apply_session_layout; snapshot and restore.
    _SETTING_ATTRS = ("SESSION_DIR", "EXTRACTED_DIR", "OUTPUTS_DIR", "TELEMETRY_DIR",
                      "ANALYSIS_DIR", "ALERTS_DIR", "INDEX_DIR", "SESSION_ID")
    snapshot = {attr: getattr(settings, attr, None) for attr in _SETTING_ATTRS}

    env = os.environ.copy()
    env["SESSION_ID"] = session_id

    def update(step: str, progress: int):
        status_update = get_session_status(session_id) or {}
        status_update["status"] = "processing"
        status_update["current_step"] = step
        status_update["progress"] = progress
        save_session_status(session_id, status_update)

    try:
        check_cancelled()

        update("generating_telemetry", 20)
        _run_stage(session_id, "telemetry", [sys.executable, "-m", "src.telemetry_generator"], env=env, timeout=300)
        check_cancelled()

        update("analyzing_frames", 40)
        _run_stage(session_id, "vision", [sys.executable, "-m", "src.vision_analyzer"], env=env, timeout=600)
        check_cancelled()

        update("indexing_frames", 60)
        try:
            from src.pinecone_indexer import index_frames
            from src.session_bootstrap import apply_session_layout
            os.environ["SESSION_ID"] = session_id
            settings.SESSION_ID = session_id
            apply_session_layout(session_id)
            index_frames()
        except Exception as exc:
            logger.warning(f"[LIVE PIPELINE] Pinecone indexing failed (non-fatal): {exc}")
        check_cancelled()

        update("generating_alerts", 75)
        _run_stage(session_id, "alerts", [sys.executable, "-m", "src.alert_engine"], env=env, timeout=300)
        check_cancelled()

        update("tracking_persons", 85)
        try:
            _run_stage(session_id, "tracking", [sys.executable, "src/person_tracker.py"], env=env, timeout=300)
        except Exception as exc:
            logger.warning(f"[LIVE PIPELINE] Person tracking failed (non-fatal): {exc}")
        check_cancelled()

        update("generating_summary", 95)
        _run_stage(session_id, "summary", [sys.executable, "-m", "src.summarizer"], env=env, timeout=300)

        status_update = get_session_status(session_id) or {}
        status_update["status"] = "completed"
        status_update["current_step"] = "completed"
        status_update["progress"] = 100
        status_update["completion_time"] = datetime.now().isoformat()
        save_session_status(session_id, status_update)
        logger.info(f"[LIVE PIPELINE] Analysis completed for live session {session_id}")

    except Exception as e:
        error_msg = str(e)
        if is_cancelled(session_id) or "cancelled by user request" in error_msg:
            logger.info(f"[LIVE PIPELINE] Session {session_id} cancelled; cleaning up.")
            import shutil
            shutil.rmtree(Path("data/sessions") / session_id, ignore_errors=True)
            clear_cancellation(session_id)
            return
        if "429" in error_msg or "rate_limit" in error_msg or "tokens per day" in error_msg:
            user_error = "API quota exhausted (Groq daily limit). Please wait 10-15 minutes or check Debug panel for status."
        elif "503" in error_msg or "DEGRADED" in error_msg:
            user_error = "AI model temporarily unavailable (server-side issue). Please retry in a few minutes."
        elif "401" in error_msg or "invalid" in error_msg.lower():
            user_error = "API key invalid or expired. Check your .env configuration."
        elif "timeout" in error_msg.lower():
            user_error = "Processing timed out. Try capturing fewer frames."
        else:
            user_error = error_msg[:200]
        logger.error(f"[LIVE PIPELINE] Analysis failed for session {session_id}: {error_msg}")
        status_update = get_session_status(session_id) or {}
        status_update["status"] = "failed"
        status_update["error"] = user_error
        status_update["error_detail"] = error_msg[:500]
        status_update["current_step"] = "failed"
        save_session_status(session_id, status_update)

    finally:
        for attr, value in snapshot.items():
            if value is not None:
                setattr(settings, attr, value)
        settings.SESSION_ID = ""
        os.environ.pop("SESSION_ID", None)


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

@app.get("/sessions/{session_id}/frames/{frame_id}/analysis")
def get_session_frame_analysis(session_id: str, frame_id: str):
    """Get frame analysis for a specific session"""
    session_id = Path(session_id).name
    frame_id = Path(frame_id).name
    path = Path("data") / "sessions" / session_id / "analysis" / f"{frame_id}_analysis.json"
    if not path.exists():
        path = settings.ANALYSIS_DIR / f"{frame_id}_analysis.json"
    if not path.exists():
        raise HTTPException(404, "Frame analysis not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/sessions/{session_id}/frames/{frame_id}/telemetry")
def get_session_frame_telemetry(session_id: str, frame_id: str):
    """Get frame telemetry for a specific session"""
    session_id = Path(session_id).name
    frame_id = Path(frame_id).name
    path = Path("data") / "sessions" / session_id / "telemetry" / f"{frame_id}_telemetry.json"
    if not path.exists():
        path = settings.TELEMETRY_DIR / f"{frame_id}_telemetry.json"
    if not path.exists():
        raise HTTPException(404, "Frame telemetry not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/sessions/{session_id}/frames/{frame_id}/alert")
def get_session_frame_alert(session_id: str, frame_id: str):
    """Get frame alert for a specific session"""
    session_id = Path(session_id).name
    frame_id = Path(frame_id).name
    path = Path("data") / "sessions" / session_id / "alerts" / f"{frame_id}_alert.json"
    if not path.exists():
        path = settings.ALERTS_DIR / f"{frame_id}_alert.json"
    if not path.exists():
        raise HTTPException(404, "Frame alert not found")
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
    
    # Check multiple possible alert file locations
    alerts_file = session_dir / "alerts" / "all_alerts.json"
    if not alerts_file.exists():
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
    
    # Try to load existing summary
    summary_data = None
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
    
    # Always enrich with key_events from analysis data
    analysis_file = session_dir / "analysis" / "all_analysis.json"
    alerts_file = session_dir / "alerts" / "all_alerts.json"
    
    key_events = []
    total_frames = 0
    total_alerts = 0
    
    if analysis_file.exists():
        with open(analysis_file, "r", encoding="utf-8") as f:
            analyses = json.load(f)
            total_frames = len([a for a in analyses if a is not None])
            # Extract significant events from analysis
            for a in analyses:
                if a is None:
                    continue
                threat = str(a.get("threat_assessment", "")).upper()
                if threat in ["HIGH", "CRITICAL", "MEDIUM"]:
                    key_events.append({
                        "timestamp": a.get("timestamp", ""),
                        "frame": a.get("frame_id", ""),
                        "description": (a.get("vlm_description") or a.get("activity") or "Suspicious activity detected")[:150],
                        "severity": threat
                    })
    
    if alerts_file.exists():
        with open(alerts_file, "r", encoding="utf-8") as f:
            alerts_data = json.load(f)
            alerts_list = alerts_data.get("alerts", [])
            total_alerts = len([a for a in alerts_list if a.get("severity", "").upper() in ["MEDIUM", "HIGH", "CRITICAL"]])
    
    # Build response
    result = {
        "session_date": datetime.now().strftime("%Y-%m-%d"),
        "total_frames_analyzed": total_frames,
        "total_alerts": total_alerts,
        "analysis_duration": f"{max(1, total_frames * 4 // 60)} mins",
        "key_events": key_events[:10],  # Top 10 events
    }
    
    # Merge with existing summary if available
    if summary_data:
        if isinstance(summary_data, dict):
            ss = summary_data.get("session_summary", summary_data)
            result["session_date"] = ss.get("date", result["session_date"])
            result["narrative"] = ss.get("session_highlights", summary_data.get("narrative", ""))
    
    return result

@app.post("/search")
def semantic_search(payload: Dict[str, Any]):
    query = payload.get("query")
    top_k = payload.get("top_k", 5)
    session_id = payload.get("session_id")
    if not query:
        raise HTTPException(400, "Missing query")
    # Set session context for namespace resolution AND directory layout
    if session_id:
        os.environ["SESSION_ID"] = session_id
        settings.SESSION_ID = session_id
        from src.session_bootstrap import apply_session_layout
        apply_session_layout(session_id)
    try:
        return search_frames(query, top_k)
    except Exception as exc:
        logger.error(f"Semantic search failed: {exc}")
        # Return an empty-but-valid response so the frontend can display a message
        return {
            "query": query,
            "namespace": session_id or settings.PINECONE_NAMESPACE,
            "integrated_inference": settings.PINECONE_USE_INTEGRATED,
            "timestamp": datetime.now().isoformat(),
            "results": [],
            "error": f"Search unavailable: {str(exc)[:200]}",
        }


@app.post("/qa")
def ask_qa(payload: Dict[str, Any]):
    question = payload.get("question")
    session_id = payload.get("session_id")
    if not question:
        raise HTTPException(400, "Missing question")
    # Set session context so QA agent reads correct analysis data
    if session_id:
        os.environ["SESSION_ID"] = session_id
        settings.SESSION_ID = session_id
        from src.session_bootstrap import apply_session_layout
        apply_session_layout(session_id)
    try:
        agent = SecurityQAAgent()
        return agent.answer(question)
    except Exception as exc:
        logger.error(f"QA agent failed: {exc}")
        # Return a graceful fallback so the frontend chat doesn't just say "connection error"
        return {
            "question": question,
            "answer": f"I was unable to process your question due to a backend issue: {str(exc)[:200]}. Please ensure the video has been fully processed and Pinecone is indexed.",
            "sources": [],
            "confidence": 0,
            "error": str(exc)[:200],
        }

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
    # Try MongoDB GridFS first
    mongo_storage = get_mongodb_storage()
    if mongo_storage.is_connected():
        image_data = mongo_storage.get_frame_image(session_id, frame_name)
        if image_data:
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
        for frame_path in stored_frames:
            if frame_name in str(frame_path):
                possible_paths.insert(0, Path(frame_path))
                break

    # Try each path
    for img_path in possible_paths:
        if img_path.exists():
            return FileResponse(
                img_path,
                media_type="image/jpeg",
                filename=frame_name
            )
    
    logger.error(f"[FRAME IMAGE] Frame not found: {frame_name}")
    raise HTTPException(404, f"Frame {frame_name} not found")

# Mount React static frontend if built, else fall back to legacy dashboard
frontend_dist = Path("frontend/dist")
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
else:
    dashboard_dir = Path("dashboard")
    if dashboard_dir.exists():
        app.mount("/legacy", StaticFiles(directory="dashboard", html=True), name="legacy-dashboard")

