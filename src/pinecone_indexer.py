"""
pinecone_indexer.py — Index frame descriptions in Pinecone and run semantic search.

Integrated mode (default): index ``flytbase`` with Pinecone server-side embedding
(llama-text-embed-v2, 768 dim). Send text via upsert_records / search — no Gemini vectors.

Legacy mode (PINECONE_USE_INTEGRATED=false): embed with Gemini and upsert raw vectors.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from pinecone import Pinecone

from src.config import settings

INDEX_NAME = settings.PINECONE_INDEX_NAME
TEXT_FIELD = settings.PINECONE_TEXT_FIELD


def _namespace() -> str:
    if settings.SESSION_ID and settings.SESSION_ID.strip():
        return settings.SESSION_ID.strip()
    return settings.PINECONE_NAMESPACE


def _safe_meta_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def _get_client() -> Pinecone:
    return Pinecone(api_key=settings.PINECONE_API_KEY)


def init_pinecone():
    """Return a data-plane Index client. Never auto-creates the index."""
    pc = _get_client()
    if settings.PINECONE_HOST:
        return pc.index(host=settings.PINECONE_HOST)
    if not pc.indexes.exists(INDEX_NAME):
        raise RuntimeError(
            f"Pinecone index '{INDEX_NAME}' not found. Create it in the Pinecone console "
            "(llama-text-embed-v2, 768 dimensions) or set PINECONE_INDEX_NAME."
        )
    return pc.index(name=INDEX_NAME)


def _build_record(frame: Dict[str, Any]) -> Dict[str, Any]:
    frame_id = frame["frame_id"]
    desc = (frame.get("vlm_description") or frame.get("description") or "").strip()
    if not desc:
        desc = f"Frame {frame_id}: {frame.get('activity', 'no description')}"

    record: Dict[str, Any] = {
        "_id": frame_id,
        TEXT_FIELD: desc,
        "frame_id": frame_id,
        "timestamp": _safe_meta_value(frame.get("timestamp"), ""),
        "location": _safe_meta_value(frame.get("location"), ""),
        "threat_assessment": _safe_meta_value(frame.get("threat_assessment"), "unknown"),
        "people_count": int(_safe_meta_value(frame.get("people_count"), 0)),
        "activity": _safe_meta_value(frame.get("activity"), "unknown"),
        "is_after_hours": bool(_safe_meta_value(frame.get("is_after_hours"), False)),
        "is_restricted_zone": bool(_safe_meta_value(frame.get("is_restricted_zone"), False)),
        "objects": ",".join(frame.get("objects_detected", []))
        if frame.get("objects_detected")
        else "",
    }
    return record


def _index_integrated(index, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    namespace = _namespace()
    log_entries: List[Dict[str, Any]] = []
    batch: List[Dict[str, Any]] = []
    batch_size = 50

    for frame in frames:
        if frame is None:
            continue
        batch.append(_build_record(frame))
        if len(batch) >= batch_size:
            log_entries.extend(_flush_integrated_batch(index, namespace, batch))
            batch = []

    if batch:
        log_entries.extend(_flush_integrated_batch(index, namespace, batch))

    return log_entries


def _flush_integrated_batch(index, namespace: str, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    start = time.time()
    try:
        response = index.upsert_records(namespace=namespace, records=batch)
        elapsed = int((time.time() - start) * 1000)
        count = getattr(response, "record_count", len(batch))
        for rec in batch:
            fid = rec.get("frame_id", rec.get("_id", "?"))
            entries.append(
                {"frame_id": fid, "status": "success", "processing_time_ms": elapsed}
            )
            print(f"Indexed {fid} (integrated, n={count})")
    except Exception as exc:
        for rec in batch:
            fid = rec.get("frame_id", rec.get("_id", "?"))
            entries.append({"frame_id": fid, "status": f"error: {exc}"})
            print(f"Failed to index {fid}: {exc}")
    return entries


def _index_legacy(index, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from src.gemini_client import embed_text as gemini_embed_text

    log_entries: List[Dict[str, Any]] = []
    for frame in frames:
        if frame is None:
            continue
        frame_id = frame["frame_id"]
        desc = (frame.get("vlm_description") or "").strip()
        meta = {
            "frame_id": frame_id,
            "timestamp": _safe_meta_value(frame.get("timestamp"), ""),
            "location": _safe_meta_value(frame.get("location"), ""),
            "threat_assessment": _safe_meta_value(frame.get("threat_assessment"), "unknown"),
            "people_count": int(_safe_meta_value(frame.get("people_count"), 0)),
            "activity": _safe_meta_value(frame.get("activity"), "unknown"),
            "is_after_hours": bool(_safe_meta_value(frame.get("is_after_hours"), False)),
            "is_restricted_zone": bool(_safe_meta_value(frame.get("is_restricted_zone"), False)),
            "objects": ",".join(frame.get("objects_detected", []))
            if frame.get("objects_detected")
            else "",
        }
        try:
            start = time.time()
            embedding = gemini_embed_text(desc or f"frame {frame_id}")
            index.upsert(vectors=[{"id": frame_id, "values": embedding, "metadata": meta}])
            elapsed = int((time.time() - start) * 1000)
            log_entries.append(
                {"frame_id": frame_id, "status": "success", "processing_time_ms": elapsed}
            )
            print(f"Indexed {frame_id} (legacy vectors)")
        except Exception as exc:
            log_entries.append({"frame_id": frame_id, "status": f"error: {exc}"})
            print(f"Failed to index {frame_id}: {exc}")
    return log_entries


def index_frames() -> Dict[str, Any]:
    print("\nIndexing frame descriptions in Pinecone...")
    index = init_pinecone()
    analysis_path = settings.ANALYSIS_DIR / "all_analysis.json"
    with open(analysis_path, "r", encoding="utf-8") as handle:
        all_analysis = json.load(handle)

    if settings.PINECONE_USE_INTEGRATED:
        frames_log = _index_integrated(index, all_analysis)
        embed_model = "llama-text-embed-v2 (Pinecone integrated)"
    else:
        frames_log = _index_legacy(index, all_analysis)
        embed_model = settings.GEMINI_EMBEDDING_MODEL

    success_count = sum(1 for f in frames_log if f.get("status") == "success")
    log = {
        "total_indexed": success_count,
        "index_name": INDEX_NAME,
        "namespace": _namespace(),
        "integrated_inference": settings.PINECONE_USE_INTEGRATED,
        "embedding_model": embed_model,
        "dimensions": settings.PINECONE_DIMENSION,
        "text_field": TEXT_FIELD,
        "frames": frames_log,
    }
    log_path = settings.INDEX_DIR / "indexing_log.json"
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2)
    print(f"\nIndexing log saved to {log_path}")
    return log


def _search_integrated(index, query: str, top_k: int) -> List[Dict[str, Any]]:
    namespace = _namespace()
    response = index.search(
        namespace=namespace,
        top_k=top_k,
        inputs={"text": query},
        fields=[TEXT_FIELD, "frame_id", "timestamp", "location", "threat_assessment", "activity"],
    )
    hits = []
    for hit in response.result.hits:
        fields = hit.fields or {}
        hits.append(
            {
                "frame_id": fields.get("frame_id", hit.id),
                "similarity_score": round(hit.score, 4),
                "timestamp": fields.get("timestamp"),
                "location": fields.get("location"),
                "description": fields.get(TEXT_FIELD, fields.get("text", "")),
                "threat_assessment": fields.get("threat_assessment"),
            }
        )
    return hits


def _search_legacy(index, query: str, top_k: int) -> List[Dict[str, Any]]:
    from src.gemini_client import embed_text as gemini_embed_text

    embedding = gemini_embed_text(query)
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    hits = []
    for match in results.matches:
        meta = match.metadata or {}
        hits.append(
            {
                "frame_id": meta.get("frame_id", match.id),
                "similarity_score": round(match.score, 4),
                "timestamp": meta.get("timestamp"),
                "location": meta.get("location"),
                "description": meta.get("objects", ""),
                "threat_assessment": meta.get("threat_assessment"),
            }
        )
    return hits


def search_frames(query: str, top_k: int = 5) -> Dict[str, Any]:
    print(f"\nSearching Pinecone for: '{query}' (top_k={top_k})")
    index = init_pinecone()
    if settings.PINECONE_USE_INTEGRATED:
        hits = _search_integrated(index, query, top_k)
    else:
        hits = _search_legacy(index, query, top_k)

    out = {
        "query": query,
        "namespace": _namespace(),
        "integrated_inference": settings.PINECONE_USE_INTEGRATED,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": hits,
    }
    qdir = settings.INDEX_DIR / "query_results"
    qdir.mkdir(exist_ok=True)
    qfile = qdir / f"query_{int(time.time())}.json"
    with open(qfile, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"Query results saved to {qfile}")
    return out


if __name__ == "__main__":
    index_frames()
