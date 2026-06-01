"""
dashboard.py — Production-level Streamlit dashboard for Drone Security Analyst Agent.
Modern UI with smooth interactions, real-time updates, and responsive design.
"""

import streamlit as st
import json
import os
import sys
import requests
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.qa_agent import SecurityQAAgent
from src.pinecone_indexer import search_frames

# API configuration - use environment variable or default to localhost
API_BASE = os.environ.get("API_URL", "http://localhost:8000")

# Streamlit compatibility function for image display
def display_image(image_path, caption=None, width=None, session_id=None):
    """Display image with Streamlit version compatibility.
    
    For Railway deployment, image_path can be a URL or local path.
    If session_id is provided, will construct API URL for the image.
    """
    try:
        # If session_id provided, use API URL
        if session_id and API_BASE:
            # image_path is just the filename
            if not image_path.startswith('http'):
                image_url = f"{API_BASE}/sessions/{session_id}/frame-image/{image_path}"
                st.image(image_url, caption=caption, width=width)
                return
        
        # Try newer parameter first
        if width:
            st.image(image_path, caption=caption, width=width)
        else:
            st.image(image_path, caption=caption, use_column_width=True)
    except TypeError:
        # Fallback to older parameter
        if width:
            st.image(image_path, caption=caption, width=width)
        else:
            st.image(image_path, caption=caption, use_column_width=True)

# Configure page with modern settings
st.set_page_config(
    page_title="🛡️ Drone Security Analyst Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
        margin: 0.5rem 0;
    }
    
    .alert-high {
        border-left-color: #e74c3c;
        background: #fdf2f2;
    }
    
    .alert-medium {
        border-left-color: #f39c12;
        background: #fef9e7;
    }
    
    .alert-low {
        border-left-color: #27ae60;
        background: #e8f8f5;
    }
    
    .status-online {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #27ae60;
        border-radius: 50%;
        margin-right: 5px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .frame-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .frame-card {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .frame-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    
    .search-result {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
        transition: all 0.2s;
    }
    
    .search-result:hover {
        border-color: #2a5298;
        box-shadow: 0 2px 4px rgba(42, 82, 152, 0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_json(path: Path):
    """Load JSON file with caching."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading {path}: {e}")
            return None
    return None

def get_api_data(endpoint: str) -> Dict[str, Any]:
    """Get data from API with retry logic for Render free tier sleep mode."""
    max_retries = 5  # Increased for Render sleep wake-up
    retry_delay = 3  # Longer delay for service wake-up
    
    for attempt in range(max_retries):
        try:
            url = f"{API_BASE}/{endpoint}"
            # Longer timeout for Render free tier wake-up (can take 10-30s)
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 502:
                # Service is waking up, retry without spamming errors
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                # Only show error on final attempt
                st.warning(f"⚠️ API waking up from sleep... (attempt {attempt + 1}/{max_retries})")
            else:
                st.error(f"API returned status {response.status_code} for {endpoint}")
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            st.error(f"⏱️ API timeout - service may be starting up")
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            st.error(f"🔌 API connection refused - service waking up")
        except Exception as e:
            st.error(f"Unexpected error loading {endpoint}: {e}")
    return {}

def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%I:%M %p")
    except:
        return timestamp

def get_alert_color(severity: str) -> str:
    """Get color for alert severity."""
    colors = {
        "HIGH": "#e74c3c",
        "MEDIUM": "#f39c12", 
        "LOW": "#27ae60",
        "NONE": "#95a5a6"
    }
    return colors.get(severity, "#95a5a6")

def create_metric_card(title: str, value: Any, color: str = "#2a5298"):
    """Create a styled metric card."""
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {color};">
        <h4 style="margin: 0; color: #2c3e50;">{title}</h4>
        <p style="margin: 0.5rem 0; font-size: 1.5rem; font-weight: bold; color: {color};">{value}</p>
    </div>
    """, unsafe_allow_html=True)

def extract_from_raw_response(raw: str) -> Dict[str, Any]:
    """Extract partial data from raw_response JSON string."""
    import re
    result = {}
    
    # Extract vlm_description
    vlm_match = re.search(r'"vlm_description"\s*:\s*"([^"]*)"', raw)
    if vlm_match:
        result["vlm_description"] = vlm_match.group(1)
    
    # Extract scene_type
    scene_match = re.search(r'"scene_type"\s*:\s*"([^"]*)"', raw)
    if scene_match:
        result["scene_type"] = scene_match.group(1)
    
    # Extract people_count
    people_match = re.search(r'"people_count"\s*:\s*(\d+)', raw)
    if people_match:
        result["people_count"] = int(people_match.group(1))
    
    # Extract objects_detected
    objects_match = re.search(r'"objects_detected"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
    if objects_match:
        try:
            objects_str = objects_match.group(1)
            objects = re.findall(r'"([^"]*)"', objects_str)
            if objects:
                result["objects_detected"] = objects
        except:
            pass
    
    # Extract activity
    activity_match = re.search(r'"activity"\s*:\s*"([^"]*)"', raw)
    if activity_match:
        result["activity"] = activity_match.group(1)
    
    # Extract person_features (handle partial/truncated arrays)
    person_match = re.search(r'"person_features"\s*:\s*(\[.*?\])(?:,\s*"|$)', raw, re.DOTALL)
    if person_match:
        try:
            person_str = person_match.group(1)
            result["person_features"] = json.loads(person_str)
        except:
            pass
    
    return result

def display_alert_card(alert: Dict[str, Any], compact: bool = False):
    """Display an alert card with proper formatting."""
    severity = alert.get('severity', 'NONE')
    alert_type = alert.get('alert_type', 'Unknown')
    frame_id = alert.get('frame_id', 'N/A')
    timestamp = alert.get('timestamp', 'N/A')
    location = alert.get('location', 'N/A')
    message = alert.get('message', 'No message provided')
    recommended_action = alert.get('recommended_action', '')
    objects_involved = alert.get('objects_involved', [])
    
    alert_color = get_alert_color(severity)
    
    # Format objects
    objects_str = ', '.join(objects_involved) if objects_involved else 'None'
    
    if compact:
        # Compact view for low severity
        st.markdown(f"""
        <div style="padding: 0.5rem; border-left: 3px solid {alert_color}; margin: 0.25rem 0; background: #f8f9fa;">
            <strong>{alert_type}</strong> - {frame_id} at {location} ({timestamp})
        </div>
        """, unsafe_allow_html=True)
    else:
        # Full view for high/medium severity
        st.markdown(f"""
        <div class="metric-card alert-{severity.lower()}" style="margin: 0.75rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="flex: 1;">
                    <h4 style="margin: 0 0 0.5rem 0; color: {alert_color};">🚨 {alert_type}</h4>
                    <p style="margin: 0.25rem 0;"><strong>Frame:</strong> {frame_id}</p>
                    <p style="margin: 0.25rem 0;"><strong>Time:</strong> {timestamp}</p>
                    <p style="margin: 0.25rem 0;"><strong>Location:</strong> {location}</p>
                    <p style="margin: 0.25rem 0;"><strong>Objects:</strong> {objects_str}</p>
                </div>
                <div style="text-align: right; margin-left: 1rem;">
                    <span style="background: {alert_color}; color: white; padding: 0.35rem 1rem; border-radius: 20px; font-size: 0.9rem; font-weight: bold;">
                        {severity}
                    </span>
                </div>
            </div>
            <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #eee;">
                <p style="margin: 0; color: #2c3e50;"><strong>Message:</strong> {message}</p>
                {f'<p style="margin: 0.5rem 0 0 0; color: #27ae60;"><strong>Recommended Action:</strong> {recommended_action}</p>' if recommended_action else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_frame' not in st.session_state:
    st.session_state.selected_frame = None
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'selected_session_id' not in st.session_state:
    st.session_state.selected_session_id = None
if 'active_video_session' not in st.session_state:
    st.session_state.active_video_session = None
if 'active_session_id' not in st.session_state:
    st.session_state.active_session_id = None
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

# Header with status
st.markdown("""
<div class="main-header">
    <h1>🛡️ Drone Security Analyst Dashboard</h1>
    <p>Real-time AI-powered security monitoring and analysis</p>
    <p><span class="status-online"></span>System Online • Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%I:%M:%S %p")), unsafe_allow_html=True)

# Sidebar with system info
with st.sidebar:
    st.markdown("### 📊 System Status")
    
    # API Status
    api_status = get_api_data("health")
    if api_status:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Disconnected")
    
    # Quick Stats
    st.markdown("### 📈 Quick Stats")
    frames_data = get_api_data("frames")
    if frames_data:
        st.metric("Total Frames", len(frames_data.get("frames", [])))
    
    alerts_data = get_api_data("alerts")
    if alerts_data:
        st.metric("Total Alerts", alerts_data.get("total_alerts", 0))
    
    # Refresh button
    if st.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()

# Tab state management
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# Tab selection UI
tab_names = [
    "🎬 Frame Analysis", 
    "🚨 Alert Center", 
    "🔍 Semantic Search", 
    "📊 Session Summary", 
    "💬 Security Agent"
]

# Create tab selection buttons
cols = st.columns(len(tab_names))
for i, tab_name in enumerate(tab_names):
    with cols[i]:
        if st.button(tab_name, type="primary" if st.session_state.active_tab == i else "secondary", key=f"tab_{i}"):
            st.session_state.active_tab = i
            st.rerun()

st.markdown("---")

# Dynamic content based on selected tab
if st.session_state.active_tab == 0:
    st.markdown("### 🎬 Frame Analysis")
    
    # Video upload section
    st.markdown("#### 📹 Upload New Video for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'dav', 'mkv', 'wmv', 'flv'],
        help="Upload a security video file for AI analysis. Supported formats: MP4, AVI, MOV, DAV, MKV, WMV, FLV"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / (1024*1024):.2f} MB")
        with col3:
            file_ext = uploaded_file.name.split('.')[-1].upper()
            st.metric("File Type", file_ext)
        
        # Extraction strategy selection
        st.markdown("##### 🎯 Frame Extraction Strategy")
        col1, col2 = st.columns(2)
        
        with col1:
            extraction_strategy = st.selectbox(
                "Choose extraction strategy:",
                options=["hybrid", "motion_based", "scene_change", "uniform"],
                format_func=lambda x: {
                    "hybrid": "🧠 Hybrid (Recommended)",
                    "motion_based": "🎬 Motion-Based",
                    "scene_change": "🎭 Scene Change",
                    "uniform": "⏱️ Uniform"
                }.get(x, x),
                index=0
            )
        
        with col2:
            max_frames = st.slider(
                "Maximum frames to extract:",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Higher values provide more detailed analysis but take longer"
            )
        
        # Strategy descriptions
        strategy_descriptions = {
            "hybrid": "🧠 Combines motion detection and scene changes for optimal results",
            "motion_based": "🎬 Focuses on frames with significant motion/activity",
            "scene_change": "🎭 Extracts frames when the scene changes significantly",
            "uniform": "⏱️ Extracts frames at regular time intervals"
        }
        
        st.info(strategy_descriptions[extraction_strategy])
        
        # Upload button
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            with st.spinner("Uploading and starting analysis..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    params = {
                        "extraction_strategy": extraction_strategy,
                        "max_frames": max_frames
                    }
                    response = requests.post(
                        f"{API_BASE}/upload-video",
                        files=files,
                        params=params,
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        session_id = result["session_id"]
                        
                        st.write(f"🔍 Debug Upload - Session ID: {session_id}")
                        st.write(f"🔍 Debug Upload - Response: {result}")
                        
                        # Store session ID in session state and set as active video
                        st.session_state.current_session_id = session_id
                        st.session_state.selected_session_id = session_id
                        st.session_state.active_video_session = session_id
                        st.session_state.active_session_id = session_id
                        
                        st.write(f"🔍 Debug Upload - Session states set:")
                        st.write(f"  - current_session_id: {st.session_state.current_session_id}")
                        st.write(f"  - selected_session_id: {st.session_state.selected_session_id}")
                        st.write(f"  - active_video_session: {st.session_state.active_video_session}")
                        
                        st.success(f"✅ Video uploaded successfully! Session ID: {session_id}")
                        st.info("🔄 Processing started... Analysis will appear here when complete.")
                        st.rerun()
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Error uploading video: {str(e)}")
    
    st.markdown("---")
    
    # Video session selector
    st.markdown("#### 📚 Video Sessions")
    
    # Get all sessions
    sessions_data = get_api_data("sessions")
    sessions = sessions_data.get("sessions", []) if sessions_data else []
    
    if sessions:
        # Create session selector
        session_options = []
        for session in sessions:
            session_name = f"{session.get('filename', 'Unknown')} ({session.get('status', 'Unknown')})"
            session_options.append((session["session_id"], session_name))
        
        selected_session_id = st.selectbox(
            "Select video session to analyze:",
            options=[opt[0] for opt in session_options],
            format_func=lambda x: next(opt[1] for opt in session_options if opt[0] == x),
            index=0,
            key="session_selector"
        )
        
        # Store selected session
        st.session_state.selected_session_id = selected_session_id
        st.session_state.active_session_id = selected_session_id
        
        # Show session status
        selected_session = next((s for s in sessions if s["session_id"] == selected_session_id), None)
        if selected_session:
            st.markdown(f"**Status:** {selected_session.get('status', 'Unknown')}")
            if selected_session.get('status') == 'completed':
                st.success("✅ Processing completed - Ready for analysis")
            elif selected_session.get('status') == 'processing':
                st.warning("⏳ Processing in progress...")
            else:
                st.info("📤 Uploaded, waiting for processing")
    
    # Processing status for current session
    if 'current_session_id' in st.session_state and st.session_state.current_session_id:
        st.markdown("#### 📊 Processing Status")
        session_id = st.session_state.current_session_id
        
        try:
            response = requests.get(f"{API_BASE}/processing-status/{session_id}", timeout=60)
            if response.status_code == 200:
                status = response.json()
                
                # Progress bar
                progress = status.get("progress", 0)
                st.progress(progress / 100)
                st.write(f"**Current Step:** {status.get('current_step', 'Unknown')}")
                
                # Processing steps
                if status.get("processing_steps"):
                    st.write("**Completed Steps:**")
                    for step in status["processing_steps"]:
                        st.write(f"✅ {step.replace('_', ' ').title()}")
                
                # Error display
                if status.get("error"):
                    st.error(f"❌ Error: {status['error']}")
                
                # Refresh button
                if status.get("status") in ["processing", "uploaded"]:
                    if st.button("🔄 Refresh Status"):
                        st.rerun()
                
                # View results button
                if status.get("status") == "completed":
                    st.success("🎉 Processing completed!")
                    if st.button("📈 View Analysis Results", type="primary"):
                        # Switch to Frame Analysis tab (already here)
                        st.session_state.active_tab = 0
                        st.rerun()
                        
            else:
                st.error("❌ Could not retrieve processing status")
                
        except Exception as e:
            st.error(f"❌ Error checking status: {str(e)}")
    
    st.markdown("---")
    
    # Frame Analysis Section
    st.markdown("#### 🎬 Frame Analysis")
    
    # Frame loading - use session-specific API if session is selected, otherwise general API
    active_session_id = st.session_state.get('selected_session_id') or st.session_state.get('active_session_id')
    
    if active_session_id:
        # Use session-specific API for Railway persistence
        frames_data = get_api_data(f"sessions/{active_session_id}/frames")
        if not frames_data:
            # Fallback to general frames endpoint
            frames_data = get_api_data("frames")
    else:
        # No active session, use general frames endpoint
        frames_data = get_api_data("frames")
    
    # If API doesn't work, try direct file system access with sequential folders
    if not frames_data or not frames_data.get("frames"):
        # Try to find the latest extracted folder
        data_dir = Path("data")
        latest_folder = None
        latest_num = 0
        
        if data_dir.exists():
            for item in data_dir.iterdir():
                if item.is_dir() and item.name.startswith("extracted"):
                    try:
                        num = int(item.name.replace("extracted", ""))
                        if num > latest_num:
                            latest_num = num
                            latest_folder = item
                    except ValueError:
                        continue
        
        # Use latest folder or fallback to original extracted
        if latest_folder and latest_folder.exists():
            frames = [f.name for f in sorted(latest_folder.glob("*.jpg"))]
            if frames:
                frames_data = {"frames": frames}
        else:
            # Fallback to original extracted directory
            extracted_dir = Path("data/extracted")
            if extracted_dir.exists():
                frames = [f.name for f in sorted(extracted_dir.glob("*.jpg"))]
                if frames:
                    frames_data = {"frames": frames}
    
    if frames_data and frames_data.get("frames"):
        frames = frames_data["frames"]
        
        # Frame selector with preview
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_frame = st.selectbox(
                "Select Frame",
                options=frames,
                format_func=lambda x: x.replace(".jpg", ""),
                key="frame_selector"
            )
            
            if selected_frame != st.session_state.selected_frame:
                st.session_state.selected_frame = selected_frame
        
        with col2:
            if st.session_state.selected_frame:
                frame_id = st.session_state.selected_frame.replace(".jpg", "")
                
                # Display frame image - use sequential extracted folders
                img_path = None
                
                # Try to find the latest extracted folder
                data_dir = Path("data")
                latest_folder = None
                latest_num = 0
                
                if data_dir.exists():
                    for item in data_dir.iterdir():
                        if item.is_dir() and item.name.startswith("extracted"):
                            try:
                                num = int(item.name.replace("extracted", ""))
                                if num > latest_num:
                                    latest_num = num
                                    latest_folder = item
                            except ValueError:
                                continue
                
                # For Railway: Use API URL to display frame images
                # Get session_id from active session or use frame_id for lookup
                session_id = st.session_state.get('active_session_id', '')
                if session_id:
                    # Use API to serve image - just pass filename and session_id
                    display_image(f"{frame_id}.jpg", caption=f"Frame: {frame_id}", session_id=session_id)
                else:
                    st.warning("No active session for frame display")
                
                # Load and display analysis data - use session-specific paths if available
                if st.session_state.get('active_session_id'):
                    telemetry_path = Path("data") / "sessions" / st.session_state.active_session_id / "telemetry" / f"{frame_id}_telemetry.json"
                    analysis_path = Path("data") / "sessions" / st.session_state.active_session_id / "analysis" / f"{frame_id}_analysis.json"
                    alert_path = Path("data") / "sessions" / st.session_state.active_session_id / "alerts" / f"{frame_id}_alert.json"
                    
                    # Fallback to general paths if session-specific don't exist
                    if not telemetry_path.exists():
                        telemetry_path = settings.TELEMETRY_DIR / f"{frame_id}_telemetry.json"
                    if not analysis_path.exists():
                        analysis_path = settings.ANALYSIS_DIR / f"{frame_id}_analysis.json"
                    if not alert_path.exists():
                        alert_path = settings.ALERTS_DIR / f"{frame_id}_alert.json"
                else:
                    telemetry_path = settings.TELEMETRY_DIR / f"{frame_id}_telemetry.json"
                    analysis_path = settings.ANALYSIS_DIR / f"{frame_id}_analysis.json"
                    alert_path = settings.ALERTS_DIR / f"{frame_id}_alert.json"
                
                telemetry = load_json(telemetry_path)
                analysis = load_json(analysis_path)
                alert = load_json(alert_path)
                
                # Analysis in columns
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("#### 📍 Telemetry Data")
                    if telemetry:
                        st.json(telemetry)
                    else:
                        st.info("No telemetry data")
                
                with col_b:
                    st.markdown("#### 🔍 Vision Analysis")
                    if analysis:
                        # Extract data from raw_response if parsing failed
                        display_analysis = dict(analysis)
                        if display_analysis.get("_parsing_failed") and display_analysis.get("raw_response"):
                            # Try to extract partial data
                            raw = display_analysis["raw_response"]
                            extracted = extract_from_raw_response(raw)
                            if extracted:
                                display_analysis.update(extracted)
                                display_analysis.pop("_parsing_failed", None)
                        
                        # Show key fields in a nice format
                        st.markdown(f"**Scene:** {display_analysis.get('scene_type', 'Unknown')}")
                        st.markdown(f"**Description:** {display_analysis.get('vlm_description', 'No description')}")
                        st.markdown(f"**People Count:** {display_analysis.get('people_count', 0)}")
                        st.markdown(f"**Activity:** {display_analysis.get('activity', 'Unknown')}")
                        
                        objects = display_analysis.get('objects_detected', [])
                        if objects:
                            st.markdown(f"**Objects:** {', '.join(objects)}")
                        
                        # Show person features if available
                        person_features = display_analysis.get('person_features', [])
                        if person_features:
                            st.markdown("**👥 People Detected:**")
                            for person in person_features:
                                if isinstance(person, dict):
                                    person_id = person.get('id', 'Unknown')
                                    clothing = person.get('clothing_color', 'unknown')
                                    actions = person.get('actions', [])
                                    action_str = ', '.join(actions) if actions else 'standing'
                                    st.markdown(f"  • {person_id}: {clothing} clothing, {action_str}")
                        
                        # Expand for full raw JSON
                        with st.expander("📄 Raw Analysis Data"):
                            st.json(analysis)
                    else:
                        st.info("No analysis data")
                
                # Alert status - Read from analysis threat fields
                st.markdown("#### 🚨 Alert Status")
                
                # Get threat info from analysis (new system)
                if analysis is None:
                    analysis = {}
                threat_level = analysis.get("threat_level") or analysis.get("overall_threat_level", "CLEAR")
                threat_type = analysis.get("threat_type", "clear")
                alert_reasoning = analysis.get("alert_reasoning") or analysis.get("reasoning", "No reasoning provided")
                security_signals = analysis.get("security_signals", [])
                
                # Determine if there's an active threat
                has_threat = threat_level in ["CRITICAL", "HIGH", "MEDIUM"] and threat_type != "clear"
                
                if has_threat:
                    alert_color = get_alert_color(threat_level)
                    st.markdown(f"""
                    <div class="metric-card alert-{threat_level.lower()}">
                        <h4 style="margin: 0;">🚨 Security Alert</h4>
                        <p style="color: {alert_color}; font-weight: bold; font-size: 1.3rem;">
                            {threat_level}
                        </p>
                        <p style="margin: 0.5rem 0; font-weight: bold;">Type: {threat_type.replace('_', ' ').title()}</p>
                        <p style="margin: 0.3rem 0; color: #e74c3c;">
                            Signals: {', '.join(security_signals) if security_signals else 'None detected'}
                        </p>
                        <p style="margin: 0; font-size: 0.9rem; color: #666;">
                            {alert_reasoning}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif threat_level == "LOW":
                    create_metric_card("Low Priority", "⚠️ Monitor", "#f39c12")
                else:
                    create_metric_card("No Threats", "✅ Clear", "#27ae60")
    else:
        st.warning("No frames available. Please run the analysis pipeline first.")

elif st.session_state.active_tab == 1:
    st.markdown("### 🚨 Alert Center")
    
    # Get alerts for active video session
    active_session_id = st.session_state.get('active_video_session') or st.session_state.get('active_session_id')
    alerts_data = None
    
    if active_session_id:
        # Try API first
        alerts_data = get_api_data(f"sessions/{active_session_id}/alerts")
        
        # Try to load session-specific alerts file directly if API doesn't work
        if not alerts_data or not alerts_data.get("alerts"):
            alerts_file = Path("data") / "sessions" / active_session_id / "alerts.json"
            if alerts_file.exists():
                alerts_data = load_json(alerts_file)
    else:
        alerts_data = get_api_data("alerts")
    
    # Also try fallback to general alerts file
    if not alerts_data or not alerts_data.get("alerts"):
        general_alerts_file = Path("outputs/alerts/all_alerts.json")
        if general_alerts_file.exists():
            alerts_data = load_json(general_alerts_file)
    if alerts_data:
        # Alert metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Total Alerts", alerts_data.get("total_alerts", 0), "#e74c3c")
        
        with col2:
            create_metric_card("High Severity", alerts_data.get("high_severity", 0), "#e74c3c")
        
        with col3:
            create_metric_card("Medium Severity", alerts_data.get("medium_severity", 0), "#f39c12")
        
        with col4:
            create_metric_card("Low Severity", alerts_data.get("low_severity", 0), "#27ae60")
        
        # Alert details
        alerts_list = alerts_data.get("alerts", [])
        # Filter out duplicate alerts for cleaner display
        unique_alerts = [a for a in alerts_list if not a.get('duplicate', False)]
        
        if unique_alerts:
            st.markdown("#### 📋 Alert Details")
            
            # Group alerts by severity
            high_alerts = [a for a in unique_alerts if a.get('severity') == 'HIGH']
            medium_alerts = [a for a in unique_alerts if a.get('severity') == 'MEDIUM']
            low_alerts = [a for a in unique_alerts if a.get('severity') == 'LOW']
            
            # Display HIGH severity alerts first
            if high_alerts:
                st.markdown("##### 🔴 HIGH SEVERITY")
                for alert in high_alerts:
                    display_alert_card(alert)
            
            # Display MEDIUM severity alerts
            if medium_alerts:
                st.markdown("##### 🟡 MEDIUM SEVERITY")
                for alert in medium_alerts:
                    display_alert_card(alert)
            
            # Display LOW severity alerts
            if low_alerts:
                with st.expander("🟢 Low Severity Alerts"):
                    for alert in low_alerts:
                        display_alert_card(alert, compact=True)
        else:
            st.success("🎉 No active alerts detected! System is secure.")
    else:
        st.warning("No alert data available")

elif st.session_state.active_tab == 2:
    st.markdown("### 🔍 Semantic Search")
    
    # Show which video is being searched
    active_session_id = st.session_state.get('active_video_session') or st.session_state.get('active_session_id')
    if active_session_id:
        st.info(f"🔍 Searching within uploaded video session: {active_session_id}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search frames by objects, activities, or descriptions...",
            placeholder="e.g., 'person in red shirt', 'white SUV', 'suspicious activity'"
        )
    
    with col2:
        top_k = st.selectbox("Results", [3, 5, 10, 15], index=1)
    
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("🔍 Searching frames..."):
            try:
                # Try API search first
                search_payload = {"query": query, "top_k": top_k}
                if active_session_id:
                    search_payload["session_id"] = active_session_id
                
                search_response = requests.post(f"{API_BASE}/search", json=search_payload, timeout=30)
                if search_response.status_code == 200:
                    results = search_response.json()
                else:
                    # Fallback to local search function
                    results = search_frames(query, top_k)
                
                if results and results.get("results"):
                    st.markdown(f"#### 📊 Found {len(results['results'])} results for: '{query}'")
                    
                    for i, result in enumerate(results["results"], 1):
                        frame_id = result.get("frame_id", "Unknown")
                        score = result.get("score", 0)
                        
                        st.markdown(f"""
                        <div class="search-result">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h4 style="margin: 0;">Result {i}: {frame_id}</h4>
                                <span style="background: #2a5298; color: white; padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.8rem;">
                                    Score: {score:.2f}
                                </span>
                            </div>
                            <p style="margin: 0.5rem 0; color: #666;">
                                {result.get('description', 'No description available')}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show frame preview
                        img_path = settings.EXTRACTED_DIR / f"{frame_id}.jpg"
                        if img_path.exists():
                            display_image(str(img_path), width=200)
                else:
                    st.info("No results found. Try different keywords.")
                    
            except Exception as e:
                st.error(f"Search error: {e}")
    
    # Popular searches
    st.markdown("#### 🔥 Popular Searches")
    popular_searches = [
        "person in restricted zone",
        "white SUV", 
        "suspicious activity",
        "security guard",
        "vehicle at gate"
    ]
    
    cols = st.columns(3)
    for i, search in enumerate(popular_searches):
        if cols[i % 3].button(search, key=f"pop_{i}"):
            st.session_state.search_query = search
            st.rerun()

elif st.session_state.active_tab == 3:
    st.markdown("### 📊 Session Summary")
    
    # Get summary for active video session
    active_session_id = st.session_state.get('active_video_session') or st.session_state.get('active_session_id')
    summary_data = None
    
    if active_session_id:
        st.info(f"� Showing summary for session: {active_session_id}")
        
        # Try API first
        summary_data = get_api_data(f"sessions/{active_session_id}/summary")
        
        # Try to load session-specific summary file directly if API doesn't work
        if not summary_data or not summary_data.get("session_summary"):
            summary_file = Path("data") / "sessions" / active_session_id / "session_summary.json"
            if summary_file.exists():
                summary_data = load_json(summary_file)
        
        # Also check other possible locations
        if not summary_data or not summary_data.get("session_summary"):
            session_dir = Path("data") / "sessions" / active_session_id
            # Check for summary in different locations
            for possible_name in ["summary.json", "session_summary.json", "analysis.json"]:
                possible_file = session_dir / possible_name
                if possible_file.exists():
                    summary_data = load_json(possible_file)
                    break
    else:
        st.info("📊 Showing general session summary (no active video session)")
        summary_data = get_api_data("session/summary")
    if summary_data and summary_data.get("session_summary"):
        summary = summary_data["session_summary"]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Frames Analyzed", summary.get("total_frames_analyzed", 0), "#3498db")
        
        with col2:
            create_metric_card("Total Alerts", summary.get("total_alerts", 0), "#e74c3c")
        
        with col3:
            create_metric_card("People Detected", summary.get("people_count", 0), "#27ae60")
        
        with col4:
            create_metric_card("Vehicles Detected", summary.get("vehicle_count", 0), "#f39c12")
        
        # Summary details
        st.markdown("#### 📋 Session Overview")
        
        # One-line summary
        if summary.get("one_line_summary"):
            st.info(f"📝 {summary['one_line_summary']}")
        
        # Detailed metrics
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("##### 🎯 Detection Statistics")
            detection_stats = summary.get("detection_statistics", {})
            if detection_stats:
                for stat, value in detection_stats.items():
                    st.metric(stat.replace("_", " ").title(), value)
        
        with col_b:
            st.markdown("##### 📍 Location Coverage")
            locations = summary.get("locations_visited", [])
            if locations:
                for location in locations:
                    st.success(f"📍 {location}")
        
        # Timeline
        st.markdown("#### ⏰ Activity Timeline")
        timeline = summary.get("activity_timeline", [])
        if timeline:
            for event in timeline[:5]:  # Show first 5 events
                timestamp = event.get("timestamp", "Unknown time")
                activity = event.get("activity", "Unknown activity")
                st.markdown(f"- **{format_timestamp(timestamp)}**: {activity}")
        
        # Full summary JSON
        with st.expander("📄 Full Summary Data"):
            st.json(summary)
    else:
        st.warning("No session summary available")

elif st.session_state.active_tab == 4:
    st.markdown("### 💬 Security Assistant")
    
    # Show which video is being analyzed
    active_session_id = st.session_state.get('active_video_session') or st.session_state.get('active_session_id')
    if active_session_id:
        st.info(f"💬 Asking questions about uploaded video session: {active_session_id}")
    else:
        st.info("💬 Ask questions about the current video analysis")
    
    # Initialize QA agent
    try:
        st.write(f"🔍 Debug QA - Initializing agent for session: {active_session_id}")
        qa_agent = SecurityQAAgent()
        st.write("🔍 Debug QA - Agent initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize QA agent: {e}")
        st.write(f"🔍 Debug QA - Error details: {str(e)}")
        qa_agent = None
    
    # Chat interface
    st.markdown("#### 🤖 Ask about today's security monitoring")
    
    # Demo questions
    demo_questions = [
        "How many people were detected today?",
        "Any suspicious activity after 6 PM?",
        "What happened at the restricted zone?",
        "Show all high severity alerts",
        "Give me today's incident timeline",
        "Were any unauthorized vehicles detected?"
    ]
    
    st.markdown("##### 💡 Quick Questions:")
    cols = st.columns(2)
    for i, q in enumerate(demo_questions):
        if cols[i % 2].button(q, key=f"demo_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()
    
    # Chat messages
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    st.caption(f"📚 Sources: {', '.join(message['sources'])}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about today's security monitoring..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        if qa_agent:
            with st.spinner("🤔 Analyzing security data..."):
                try:
                    # Check API connectivity first
                    health_response = requests.get(f"{API_BASE}/health", timeout=5)
                    if health_response.status_code != 200:
                        raise Exception("Security analysis server is not responding")
                    
                    result = qa_agent.answer(prompt)
                    
                    if result and result.get("answer"):
                        response = result.get("answer", "I couldn't process that question.")
                        sources = result.get("sources", [])
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                        
                        with chat_container:
                            with st.chat_message("assistant"):
                                st.markdown(response)
                                if sources:
                                    st.caption(f"📚 Sources: {', '.join(sources)}")
                    else:
                        # Handle empty response
                        error_msg = "I couldn't find relevant information for your question. Please try rephrasing or ensure video data is available."
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                    
                except requests.exceptions.RequestException:
                    error_msg = "🔌 Connection error: Unable to reach the security analysis server. Please ensure the system is running properly."
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
        else:
            error_msg = "⚠️ Security agent is not available. Please check if the system is properly configured."
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })
            
            with chat_container:
                with st.chat_message("assistant"):
                    st.error(error_msg)
        
        st.rerun()

elif st.session_state.active_tab == 5:
    st.markdown("### 📹 Video Upload & Processing")
    
    # Video upload section
    st.markdown("#### Upload New Video for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'dav', 'mkv', 'wmv', 'flv'],
        help="Upload a security video file for AI analysis. Supported formats: MP4, AVI, MOV, DAV, MKV, WMV, FLV"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / (1024*1024):.2f} MB")
        with col3:
            st.metric("File Type", uploaded_file.type)
        
        st.markdown("---")
        st.markdown("#### 🎯 Intelligent Frame Extraction Options")
        
        # Extraction strategy selection
        col1, col2 = st.columns(2)
        with col1:
            extraction_strategy = st.selectbox(
                "Extraction Strategy",
                options=["hybrid", "motion_based", "scene_change", "uniform"],
                index=0,
                help="Choose how frames are extracted from the video"
            )
        
        with col2:
            max_frames = st.slider(
                "Maximum Frames",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Maximum number of frames to extract (higher = more detail but slower processing)"
            )
        
        # Strategy description
        strategy_descriptions = {
            "hybrid": "🔄 Combines uniform, motion-based, and scene change detection for best results",
            "motion_based": "🏃 Extracts frames with significant motion/activity",
            "scene_change": "🎭 Extracts frames when the scene changes significantly",
            "uniform": "⏱️ Extracts frames at regular time intervals"
        }
        
        st.info(strategy_descriptions[extraction_strategy])
        
        # Upload button
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            with st.spinner("Uploading and starting analysis..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    params = {
                        "extraction_strategy": extraction_strategy,
                        "max_frames": max_frames
                    }
                    response = requests.post(
                        f"{API_BASE}/upload-video",
                        files=files,
                        params=params,
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        session_id = result["session_id"]
                        
                        # Store session ID in session state
                        st.session_state.current_session_id = session_id
                        
                        st.success(f"✅ Video uploaded successfully!")
                        st.info(f"Session ID: {session_id}")
                        st.info("Processing started. This may take 5-10 minutes.")
                        
                        # Auto-refresh to show progress
                        st.rerun()
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Error uploading video: {str(e)}")
    
    # Processing status section
    if 'current_session_id' in st.session_state and st.session_state.current_session_id:
        st.markdown("#### 📊 Processing Status")
        session_id = st.session_state.current_session_id
        
        # Get processing status
        try:
            response = requests.get(f"{API_BASE}/processing-status/{session_id}", timeout=60)
            if response.status_code == 200:
                status = response.json()
                
                # Display status
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Status", status.get("status", "unknown"))
                    st.metric("Current Step", status.get("current_step", "unknown"))
                with col2:
                    st.metric("Progress", f"{status.get('progress', 0)}%")
                    if status.get("completion_time"):
                        st.metric("Completed", status.get("completion_time"))
                
                # Progress bar
                progress = status.get("progress", 0) / 100
                st.progress(progress)
                
                # Processing steps
                if status.get("processing_steps"):
                    st.markdown("**Completed Steps:**")
                    for step in status["processing_steps"]:
                        st.write(f"✅ {step.replace('_', ' ').title()}")
                
                # Error display
                if status.get("error"):
                    st.error(f"❌ Error: {status['error']}")
                
                # Refresh button
                if status.get("status") in ["processing", "uploaded"]:
                    if st.button("🔄 Refresh Status"):
                        st.rerun()
                
                # View results button
                if status.get("status") == "completed":
                    st.success("🎉 Processing completed!")
                    if st.button("📈 View Analysis Results", type="primary"):
                        # Switch to Frame Analysis tab
                        st.session_state.active_tab = 0
                        st.rerun()
                        
            else:
                st.error("❌ Could not retrieve processing status")
                
        except Exception as e:
            st.error(f"❌ Error checking status: {str(e)}")
    
    # Sessions history
    st.markdown("#### 📚 Processing History")
    try:
        response = requests.get(f"{API_BASE}/sessions")
        if response.status_code == 200:
            sessions_data = response.json()
            sessions = sessions_data.get("sessions", [])
            
            if sessions:
                # Create dataframe for better display
                import pandas as pd
                df = pd.DataFrame(sessions)
                
                # Style the dataframe
                styled_df = df.style.map(
                    lambda x: 'color: green' if x == 'completed' else 'color: orange' if x == 'processing' else 'color: red',
                    subset=['status']
                )
                
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.info("No processing sessions found.")
                
    except Exception as e:
        st.error(f"❌ Error loading sessions: {str(e)}")
    
    # Instructions
    with st.expander("📖 Instructions & Tips"):
        st.markdown("""
        **How to use the video upload feature:**
        
        1. **Upload**: Choose a video file from your device
        2. **Process**: Click "Start Processing" to begin AI analysis
        3. **Monitor**: Watch the progress bar and status updates
        4. **Results**: Once complete, view results in other tabs
        
        **Supported Video Formats:**
        - MP4, AVI, MOV, DAV, MKV, WMV, FLV
        
        **Processing Steps:**
        1. Frame extraction from video
        2. Telemetry data generation
        3. AI vision analysis with GPT-4o
        4. Security alert generation
        5. Person tracking across frames
        6. Session summary creation
        
        **Tips:**
        - Maximum file size: 500MB
        - Processing time: 5-10 minutes depending on video length
        - Results are saved and can be accessed anytime
        - Multiple videos can be processed simultaneously
        """)

# Footer
st.markdown("""
---
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🛡️ Drone Security Analyst Dashboard • Powered by AI • Real-time Monitoring</p>
    <p style='font-size: 0.8rem;'>Last refresh: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
