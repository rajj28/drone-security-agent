"""
dashboard.py — Streamlit dashboard for Drone Security Analyst Agent.

- 5 tabs: Frame Analysis, Alert Center, Semantic Search, Session Summary, Ask the Agent
- Uses st.cache_resource and st.session_state
"""

import streamlit as st
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.qa_agent import SecurityQAAgent
from src.pinecone_indexer import search_frames

st.set_page_config(page_title="Drone Security Analyst Dashboard", layout="wide")

@st.cache_resource
def load_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

qa_agent = SecurityQAAgent()

TABS = ["🎬 Frame Analysis", "🚨 Alert Center", "🔍 Semantic Search", "📊 Session Summary", "💬 Ask the Agent"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(TABS)

with tab1:
    st.header("🎬 Frame Analysis")
    frames = sorted([f.name for f in settings.EXTRACTED_DIR.glob("frame_*.jpg")])
    frame_id = st.selectbox("Select Frame", [f.replace(".jpg", "") for f in frames])
    img_path = settings.EXTRACTED_DIR / f"{frame_id}.jpg"
    st.image(str(img_path), caption=frame_id)
    telemetry = load_json(settings.TELEMETRY_DIR / f"{frame_id}_telemetry.json")
    analysis = load_json(settings.ANALYSIS_DIR / f"{frame_id}_analysis.json")
    alert = load_json(settings.ALERTS_DIR / f"{frame_id}_alert.json")
    st.subheader("Telemetry")
    st.json(telemetry)
    st.subheader("Vision Analysis")
    st.json(analysis)
    st.subheader("Alert Status")
    if alert:
        color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "yellow", "NONE": "gray"}.get(alert["severity"], "gray")
        st.markdown(f"<span style='color:{color};font-weight:bold'>Severity: {alert['severity']}</span>", unsafe_allow_html=True)
        st.json(alert)
    if st.button("Analyze This Frame"):
        st.info("Frame analysis re-run not implemented in dashboard stub.")

with tab2:
    st.header("🚨 Alert Center")
    all_alerts = load_json(settings.ALERTS_DIR / "all_alerts.json")
    if all_alerts:
        st.metric("Total Alerts", all_alerts["total_alerts"])
        st.metric("High Severity", all_alerts["high_severity"])
        st.metric("Medium Severity", all_alerts["medium_severity"])
        st.metric("Low Severity", all_alerts["low_severity"])
        df = all_alerts["alerts"]
        st.dataframe(df)

with tab3:
    st.header("🔍 Semantic Search")
    query = st.text_input("Search frames by object or activity")
    top_k = st.slider("Top K Results", 1, 10, 5)
    if st.button("Search") and query:
        results = search_frames(query, top_k)
        st.json(results)

with tab4:
    st.header("📊 Session Summary")
    summary = load_json(settings.SESSION_DIR / "session_summary.json")
    if summary:
        st.metric("Total Frames", summary.get("total_frames_analyzed", 0))
        st.metric("Total Alerts", summary.get("statistics", {}).get("total_alerts", 0))
        st.metric("People Detected", summary.get("statistics", {}).get("total_people_detected", 0))
        st.metric("Vehicles", summary.get("statistics", {}).get("total_vehicles_detected", 0))
        st.markdown(f"### 📋 {summary.get('one_line_summary', '')}")
        st.json(summary)

with tab5:
    st.header("💬 Ask the Agent")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("current_question", "")
    demo_questions = [
        "How many people were detected?",
        "Any suspicious activity after hours?",
        "What happened at the restricted zone?",
        "Show all high severity alerts",
        "Give me today's incident timeline"
    ]
    st.write("### 💡 Try these questions:")
    cols = st.columns(2)
    for i, q in enumerate(demo_questions):
        if cols[i % 2].button(q, key=f"demo_{i}"):
            st.session_state.current_question = q
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption(f"Sources: {', '.join(message['sources'])}")
    if prompt := st.chat_input("Ask about today's security monitoring..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("🤔 Agent analyzing..."):
            result = qa_agent.answer(prompt)
        st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
        st.rerun()
