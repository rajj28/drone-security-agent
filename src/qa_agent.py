"""
qa_agent.py — Conversational Q&A agent for drone security session.

- Uses Pinecone search, session context, and GPT-4o for answers
- Maintains conversation history and saves to qa_log.json
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from src.config import settings
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
        self.session_context = self._load_json(settings.SESSION_DIR / "session_context.json")
        self.all_analyses = self._load_json(settings.ANALYSIS_DIR / "all_analysis.json")
        self.all_alerts = self._load_json(settings.ALERTS_DIR / "all_alerts.json")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def _load_json(self, path: Path):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def answer(self, question: str) -> Dict[str, Any]:
        relevant = search_frames(question, top_k=5)
        frame_ids = [r["frame_id"] for r in relevant["results"]]
        frame_details = [a for a in self.all_analyses if a is not None and a["frame_id"] in frame_ids]
        context = {
            "question": question,
            "session_stats": self.session_context,
            "relevant_frames": frame_details,
            "conversation_history": self.conversation_history,
            "all_alerts_summary": self.all_alerts
        }
        prompt = (
            "You are an expert drone security analyst AI with access to a full day's monitoring data. "
            "Answer questions accurately using the provided frame analyses and session statistics. "
            "Be specific with times, locations, and counts. If information is not available say so clearly.\n"
            f"Context: {json.dumps(context, indent=2)}"
        )
        start = time.time()
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )
        elapsed = int((time.time() - start) * 1000)
        answer = response.choices[0].message.content.strip()
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
