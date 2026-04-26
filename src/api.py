"""
api.py — FastAPI backend for Drone Security Analyst Agent.

- Exposes endpoints for health, frames, analysis, alerts, search, session summary, and Q&A
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from pathlib import Path
import json
from src.config import settings
from src.pinecone_indexer import search_frames
from src.qa_agent import SecurityQAAgent

app = FastAPI(title="Drone Security Analyst API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/frames")
def list_frames():
    frames = []
    for f in sorted(settings.EXTRACTED_DIR.glob("frame_*.jpg")):
        frames.append(f.name)
    return {"frames": frames}

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

@app.post("/search")
def semantic_search(payload: Dict[str, Any]):
    query = payload.get("query")
    top_k = payload.get("top_k", 5)
    if not query:
        raise HTTPException(400, "Missing query")
    return search_frames(query, top_k)

@app.get("/session/summary")
def get_session_summary():
    path = settings.SESSION_DIR / "session_summary.json"
    if not path.exists():
        raise HTTPException(404, "Session summary not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/qa")
def ask_qa(payload: Dict[str, Any]):
    question = payload.get("question")
    if not question:
        raise HTTPException(400, "Missing question")
    agent = SecurityQAAgent()
    return agent.answer(question)

@app.get("/alerts")
def get_all_alerts():
    path = settings.ALERTS_DIR / "all_alerts.json"
    if not path.exists():
        raise HTTPException(404, "Alerts not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/alerts/high")
def get_high_alerts():
    path = settings.ALERTS_DIR / "all_alerts.json"
    if not path.exists():
        raise HTTPException(404, "Alerts not found")
    with open(path, "r", encoding="utf-8") as f:
        all_alerts = json.load(f)
    high = [a for a in all_alerts.get("alerts", []) if a["severity"] == "HIGH"]
    return {"high_alerts": high}
