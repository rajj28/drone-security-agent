"""
pinecone_indexer.py — Embeds frame descriptions, indexes in Pinecone, and provides semantic search.

- Embeds vlm_description from each frame analysis using text-embedding-3-large
- Stores vectors and metadata in Pinecone
- Saves indexing log and query results as JSON
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from src.config import settings

EMBED_MODEL = settings.OPENAI_EMBEDDING_MODEL
EMBED_DIM = settings.OPENAI_EMBEDDING_DIMENSION
INDEX_NAME = settings.PINECONE_INDEX_NAME


def _safe_meta_value(value: Any, default: Any) -> Any:
    """Normalizes metadata values to Pinecone-supported primitive types."""
    if value is None:
        return default
    return value


def init_pinecone():
    """Initializes Pinecone v3 client and returns a connected index handle."""
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric=settings.PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION,
            ),
        )

    return pc.Index(INDEX_NAME)

def embed_text(text: str, client: OpenAI) -> List[float]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
        dimensions=EMBED_DIM,
    )
    return resp.data[0].embedding

def index_frames():
    print("\nIndexing frame descriptions in Pinecone...")
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    index = init_pinecone()
    analysis_path = settings.ANALYSIS_DIR / "all_analysis.json"
    with open(analysis_path, "r", encoding="utf-8") as f:
        all_analysis = json.load(f)
    frames = []
    success_count = 0
    for frame in all_analysis:
        # Skip None entries (from vision analyzer skipping missing frames)
        if frame is None:
            continue
        frame_id = frame["frame_id"]
        desc = frame.get("vlm_description", "")
        meta = {
            "frame_id": frame_id,
            "timestamp": _safe_meta_value(frame.get("timestamp"), ""),
            "location": _safe_meta_value(frame.get("location"), ""),
            "threat_assessment": _safe_meta_value(frame.get("threat_assessment"), "unknown"),
            "people_count": int(_safe_meta_value(frame.get("people_count"), 0)),
            "activity": _safe_meta_value(frame.get("activity"), "unknown"),
            "is_after_hours": bool(_safe_meta_value(frame.get("is_after_hours"), False)),
            "is_restricted_zone": bool(_safe_meta_value(frame.get("is_restricted_zone"), False)),
            "objects": ",".join(frame.get("objects_detected", [])) if frame.get("objects_detected") else ""
        }
        try:
            start = time.time()
            embedding = embed_text(desc, client)
            index.upsert(
                vectors=[
                    {
                        "id": frame_id,
                        "values": embedding,
                        "metadata": meta,
                    }
                ]
            )
            elapsed = int((time.time() - start) * 1000)
            success_count += 1
            frames.append({"frame_id": frame_id, "status": "success", "processing_time_ms": elapsed})
            print(f"Indexed {frame_id}")
        except Exception as e:
            frames.append({"frame_id": frame_id, "status": f"error: {e}"})
            print(f"Failed to index {frame_id}: {e}")
    log = {
        "total_indexed": success_count,
        "index_name": INDEX_NAME,
        "embedding_model": EMBED_MODEL,
        "dimensions": EMBED_DIM,
        "frames": frames
    }
    log_path = settings.INDEX_DIR / "indexing_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"\nIndexing log saved to {log_path}")
    return log

def search_frames(query: str, top_k: int = 5) -> Dict[str, Any]:
    print(f"\nSearching Pinecone for: '{query}' (top_k={top_k})")
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    index = init_pinecone()
    embedding = embed_text(query, client)
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    hits = []
    for match in results.matches:
        meta = match.metadata or {}
        hits.append({
            "frame_id": meta.get("frame_id", match.id),
            "similarity_score": round(match.score, 4),
            "timestamp": meta.get("timestamp"),
            "location": meta.get("location"),
            "description": meta.get("objects", ""),
            "threat_assessment": meta.get("threat_assessment")
        })
    out = {
        "query": query,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": hits
    }
    # Save query result
    qdir = settings.INDEX_DIR / "query_results"
    qdir.mkdir(exist_ok=True)
    qfile = qdir / f"query_{int(time.time())}.json"
    with open(qfile, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Query results saved to {qfile}")
    return out

if __name__ == "__main__":
    index_frames()
    # Example search
    # search_frames("person loitering at night", top_k=3)
