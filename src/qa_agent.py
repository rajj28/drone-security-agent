"""
qa_agent.py — Conversational Q&A agent for drone security session.

- Uses Pinecone search, session context, and Gemini for answers
- Maintains conversation history and saves to qa_log.json
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
from src.config import settings
from src.gemini_client import generate_text
from src.pinecone_indexer import search_frames

QA_LOG_PATH = settings.SESSION_DIR / "qa_log.json"
DEMO_QUESTIONS = [
    "How many people were detected during monitoring?",
    "Were there any vehicles detected? Describe them.",
    "What happened at the restricted zone?",
    "Show me all high severity alerts",
    "What was the most suspicious event today?",
    "Which location had the most activity?",
    "Was anything detected after hours?",
    "Give me a complete timeline of incidents today",
    "What objects appeared most frequently?",
    "Should I be concerned about today's monitoring?",
]

class SecurityQAAgent:
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_context = self._load_session_context()
        self.all_analyses = self._load_analysis_list()
        self.all_alerts = self._load_alerts_summary()

    def _load_json(self, path: Path, default: Any):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if data is not None else default
        return default

    def _load_session_context(self) -> Dict[str, Any]:
        data = self._load_json(settings.SESSION_DIR / "session_context.json", {})
        return data if isinstance(data, dict) else {}

    def _load_analysis_list(self) -> List[Dict[str, Any]]:
        data = self._load_json(settings.ANALYSIS_DIR / "all_analysis.json", [])
        return data if isinstance(data, list) else []

    def _load_alerts_summary(self) -> Dict[str, Any]:
        data = self._load_json(settings.ALERTS_DIR / "all_alerts.json", {"alerts": []})
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"alerts": data}
        return {"alerts": []}

    def answer(self, question: str) -> Dict[str, Any]:
        if not self.all_analyses and not self.session_context:
            return {
                "question": question,
                "answer": (
                    "No processed session data is available yet. Upload a video and wait for the "
                    "pipeline to finish (frame extraction, analysis, and Pinecone indexing) before asking questions."
                ),
                "sources": [],
                "confidence": 0,
            }

        relevant = search_frames(question, top_k=5)
        frame_ids = [r["frame_id"] for r in relevant["results"]]
        
        # Always include suspicious frames (e.g. ELEVATED, UNCLEAR, MEDIUM, HIGH, CRITICAL) in context
        suspicious_fids = [
            a.get("frame_id") for a in self.all_analyses
            if a is not None and a.get("frame_id") and str(a.get("threat_level", "CLEAR")).upper() not in ["CLEAR", "LOW", "UNKNOWN"]
        ]
        for fid in suspicious_fids:
            if fid not in frame_ids:
                frame_ids.append(fid)
                
        frame_details = [a for a in self.all_analyses if a is not None and a.get("frame_id") in frame_ids]
        context = {
            "question": question,
            "session_stats": self.session_context,
            "relevant_frames": frame_details,
            "conversation_history": self.conversation_history,
            "all_alerts_summary": self.all_alerts
        }
        prompt = (
            "You are a concise drone security analyst AI. "
            "Answer questions in 2-4 sentences maximum using ONLY the provided frame data below — "
            "do not invent people, objects, locations, or incidents that aren't in the data. "
            "Be specific with times, locations, and counts drawn from the actual context. "
            "Use bullet points for lists. If the data shows nothing suspicious, say so plainly "
            "instead of describing an incident that didn't happen. "
            "If theft/suspicious activity IS present in the data, state it directly.\n"
            f"Context: {json.dumps(context, indent=2)}"
        )
        start = time.time()
        answer = generate_text(prompt, max_output_tokens=512).strip()
        elapsed = int((time.time() - start) * 1000)
        qa_entry = {
            "qa_id": len(self.conversation_history) // 2 + 1,
            "question": question,
            "pinecone_query": question,
            "frames_retrieved": frame_ids,
            "similarity_scores": [r["similarity_score"] for r in relevant["results"]],
            "answer": answer,
            "sources_used": frame_ids,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "processing_time_ms": elapsed
        }
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer, "sources": frame_ids})
        self._save_qa_log(qa_entry)
        return {
            "question": question,
            "answer": answer,
            "sources": frame_ids,
            "confidence": relevant["results"][0]["similarity_score"] if relevant["results"] else 0
        }

    def _save_qa_log(self, qa_entry: Dict[str, Any]):
        log = {
            "session_id": f"SESSION_{time.strftime('%Y%m%d_%H%M%S')}",
            "total_questions": len(self.conversation_history) // 2,
            "conversations": self.conversation_history,
        }
        if QA_LOG_PATH.exists():
            with open(QA_LOG_PATH, "r", encoding="utf-8") as f:
                prev = json.load(f)
            log["conversations"] = prev.get("conversations", []) + [qa_entry]
            log["total_questions"] = len(log["conversations"])
        with open(QA_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

def run_demo_questions() -> None:
    """Runs predefined demo questions and stores results in qa_log.json."""
    print("\nRunning demo Q&A session...")
    agent = SecurityQAAgent()
    for i, question in enumerate(DEMO_QUESTIONS, start=1):
        try:
            result = agent.answer(question)
            print(f"Q{i}: {question}")
            print(f"   Sources: {', '.join(result.get('sources', [])) if result.get('sources') else 'none'}")
        except Exception as exc:
            print(f"Q{i} failed: {exc}")

    print(f"QA log saved to {QA_LOG_PATH}")


if __name__ == "__main__":
    run_demo_questions()
