"""Deterministic pytest suite for Drone Security Analyst artifacts and API."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
import src.api as api_module


def _read_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def all_telemetry() -> List[Dict[str, Any]]:
    path = settings.TELEMETRY_DIR / "all_telemetry.json"
    assert path.exists(), f"Missing telemetry artifact: {path}"
    payload = _read_json(path)
    assert isinstance(payload, list)
    assert payload, "Telemetry artifact is empty"
    return payload


@pytest.fixture(scope="module")
def all_analysis() -> List[Dict[str, Any]]:
    path = settings.ANALYSIS_DIR / "all_analysis.json"
    assert path.exists(), f"Missing analysis artifact: {path}"
    payload = _read_json(path)
    assert isinstance(payload, list)
    assert payload, "Analysis artifact is empty"
    return payload


@pytest.fixture(scope="module")
def all_alerts() -> Dict[str, Any]:
    path = settings.ALERTS_DIR / "all_alerts.json"
    assert path.exists(), f"Missing alerts artifact: {path}"
    payload = _read_json(path)
    assert isinstance(payload, dict)
    assert isinstance(payload.get("alerts", []), list)
    return payload


def test_frame_extraction_artifacts_exist() -> None:
    meta_path = settings.OUTPUTS_DIR / "extraction_log.json"
    assert meta_path.exists(), "Missing extraction_log.json"
    meta = _read_json(meta_path)
    assert isinstance(meta, dict)
    assert int(meta.get("total_frames") or meta.get("total_frames_extracted") or 0) >= 1

    frames = meta.get("frames", [])
    assert isinstance(frames, list)
    assert len(frames) >= 1
    for frame in frames:
        filename = frame.get("filename")
        assert filename, "Frame entry missing filename"
        assert (settings.EXTRACTED_DIR / filename).exists(), f"Missing extracted frame: {filename}"


def test_telemetry_schema(all_telemetry: List[Dict[str, Any]]) -> None:
    for row in all_telemetry:
        assert isinstance(row.get("frame_id"), str)
        assert isinstance(row.get("timestamp"), str)
        assert isinstance(row.get("location"), str)
        assert isinstance(row.get("drone"), dict)
        assert isinstance(row.get("is_after_hours"), bool)
        assert isinstance(row.get("is_restricted_zone"), bool)


def test_vision_analysis_schema(all_analysis: List[Dict[str, Any]]) -> None:
    valid_threats = {"none", "low", "medium", "high", "critical"}
    rows_with_text = 0
    rows_with_valid_threat = 0
    rows_with_valid_confidence = 0
    for row in all_analysis:
        assert isinstance(row.get("frame_id"), str)
        description = row.get("vlm_description") or row.get("activity")
        if isinstance(description, str) and description.strip():
            rows_with_text += 1

        threat = row.get("threat_assessment")
        if isinstance(threat, str) and threat.lower() in valid_threats:
            rows_with_valid_threat += 1

        confidence = row.get("confidence")
        if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
            rows_with_valid_confidence += 1

    assert rows_with_text / len(all_analysis) >= 0.9
    assert rows_with_valid_threat / len(all_analysis) >= 0.9
    assert rows_with_valid_confidence / len(all_analysis) >= 0.9


def test_indexing_log_consistency(all_analysis: List[Dict[str, Any]]) -> None:
    log_path = settings.INDEX_DIR / "indexing_log.json"
    assert log_path.exists(), "Missing indexing_log.json"
    log = _read_json(log_path)
    assert isinstance(log, dict)
    indexed = int(log.get("total_indexed", 0))
    assert indexed >= 1
    assert indexed == len(all_analysis)


def test_alert_integrity(all_alerts: Dict[str, Any]) -> None:
    alerts = all_alerts.get("alerts", [])
    assert len(alerts) >= 1
    allowed_severity = {"NONE", "LOW", "MEDIUM", "HIGH"}

    for alert in alerts:
        assert isinstance(alert.get("alert_id"), str)
        assert isinstance(alert.get("frame_id"), str)
        assert alert.get("severity") in allowed_severity
        assert isinstance(alert.get("alert_triggered"), bool)
        if alert.get("severity") == "HIGH":
            assert alert.get("alert_triggered") is True


def test_session_summary_exists() -> None:
    summary_path = settings.SESSION_DIR / "session_summary.json"
    assert summary_path.exists(), "Missing session_summary.json"
    summary = _read_json(summary_path)
    assert isinstance(summary, dict)
    one_line = summary.get("one_line_summary", "")
    assert isinstance(one_line, str)
    assert one_line.strip(), "one_line_summary should not be empty"


def test_qa_log_exists() -> None:
    qa_path = settings.SESSION_DIR / "qa_log.json"
    assert qa_path.exists(), "Missing qa_log.json"
    qa = _read_json(qa_path)
    assert isinstance(qa, dict)
    assert int(qa.get("total_questions", 0)) >= 0


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    class FakeQAAgent:
        def answer(self, question: str) -> Dict[str, Any]:
            return {
                "question": question,
                "answer": "stubbed answer",
                "sources": ["frame_001"],
                "confidence": 0.9,
            }

    def fake_search_frames(query: str, top_k: int = 5) -> Dict[str, Any]:
        return {
            "query": query,
            "top_k": top_k,
            "results": [
                {
                    "frame_id": "frame_001",
                    "similarity_score": 0.91,
                    "timestamp": "16:00:00",
                    "location": "Main Gate",
                }
            ],
        }

    monkeypatch.setattr(api_module, "SecurityQAAgent", FakeQAAgent)
    monkeypatch.setattr(api_module, "search_frames", fake_search_frames)
    try:
        return TestClient(api_module.app)
    except TypeError as exc:
        pytest.skip(f"TestClient/httpx compatibility issue in local environment: {exc}")


def test_api_core_endpoints(api_client: TestClient) -> None:
    health = api_client.get("/health")
    assert health.status_code == 200
    assert health.json().get("status") == "ok"

    frames = api_client.get("/frames")
    assert frames.status_code == 200
    assert isinstance(frames.json().get("frames", []), list)

    alerts = api_client.get("/alerts")
    assert alerts.status_code == 200
    assert isinstance(alerts.json().get("alerts", []), list)

    sessions = api_client.get("/sessions")
    assert sessions.status_code == 200
    assert isinstance(sessions.json(), (dict, list))


def test_api_search_and_qa(api_client: TestClient) -> None:
    search = api_client.post("/search", json={"query": "person at gate", "top_k": 3})
    assert search.status_code == 200
    search_payload = search.json()
    assert search_payload.get("query") == "person at gate"
    assert len(search_payload.get("results", [])) == 1

    qa = api_client.post("/qa", json={"question": "What is the top incident?"})
    assert qa.status_code == 200
    assert qa.json().get("answer") == "stubbed answer"


def test_end_to_end_artifact_consistency(
    all_telemetry: List[Dict[str, Any]],
    all_analysis: List[Dict[str, Any]],
    all_alerts: Dict[str, Any],
) -> None:
    telemetry_frames = {row.get("frame_id") for row in all_telemetry}
    analysis_frames = {row.get("frame_id") for row in all_analysis}
    alert_frames = {row.get("frame_id") for row in all_alerts.get("alerts", [])}

    assert telemetry_frames, "No telemetry frame IDs found"
    assert analysis_frames, "No analysis frame IDs found"
    assert telemetry_frames == analysis_frames
    assert alert_frames.issubset(analysis_frames)
