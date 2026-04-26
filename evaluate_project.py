"""
Hard evaluation harness for FlytBase Drone Security Analyst assignment.

Runs evidence-based checks against:
- Assignment requirements
- Submission expectations
- Weighted assessment metrics

Outputs:
- outputs/evaluation/hard_evaluation_report.json
- outputs/evaluation/hard_evaluation_report.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
EVAL_DIR = OUTPUTS_DIR / "evaluation"


@dataclass
class CheckResult:
    id: str
    title: str
    status: str  # pass | partial | fail | not_applicable
    points_earned: float
    points_max: float
    severity: str  # critical | major | minor | info
    evidence: str
    why_it_matters: str
    recommendation: str


def _safe_read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _score(status: str, max_points: float) -> float:
    if status == "pass":
        return max_points
    if status == "partial":
        return round(max_points * 0.5, 2)
    if status == "not_applicable":
        return max_points
    return 0.0


def _status_from_ratio(ratio: float) -> str:
    if ratio >= 0.85:
        return "pass"
    if ratio >= 0.5:
        return "partial"
    return "fail"


def run_hard_evaluation() -> Dict[str, Any]:
    results: List[CheckResult] = []

    readme_path = ROOT / "README.md"
    architecture_path = ROOT / "docs" / "architecture.md"
    tests_path = ROOT / "tests" / "test_agent.py"
    pinecone_indexer_path = ROOT / "src" / "pinecone_indexer.py"
    agent_path = ROOT / "src" / "agent.py"
    alert_engine_path = ROOT / "src" / "alert_engine.py"
    summarizer_path = ROOT / "src" / "summarizer.py"
    qa_agent_path = ROOT / "src" / "qa_agent.py"
    telemetry_all_path = ROOT / "outputs" / "telemetry" / "all_telemetry.json"
    analysis_all_path = ROOT / "outputs" / "analysis" / "all_analysis.json"
    alerts_all_path = ROOT / "outputs" / "alerts" / "all_alerts.json"
    session_context_path = ROOT / "outputs" / "session" / "session_context.json"
    context_summaries_path = ROOT / "outputs" / "session" / "context_summaries.json"
    session_summary_path = ROOT / "outputs" / "session" / "session_summary.json"
    indexing_log_path = ROOT / "outputs" / "index" / "indexing_log.json"
    agent_memory_log_path = ROOT / "outputs" / "session" / "agent_memory_log.json"
    alert_memory_log_path = ROOT / "outputs" / "session" / "alert_memory_log.json"

    readme_text = _safe_read_text(readme_path)
    architecture_text = _safe_read_text(architecture_path)
    tests_text = _safe_read_text(tests_path)
    pinecone_indexer_text = _safe_read_text(pinecone_indexer_path)
    agent_text = _safe_read_text(agent_path)
    alert_engine_text = _safe_read_text(alert_engine_path)

    telemetry_all = _safe_read_json(telemetry_all_path, [])
    analysis_all = _safe_read_json(analysis_all_path, [])
    alerts_all = _safe_read_json(alerts_all_path, {"alerts": []})
    session_context = _safe_read_json(session_context_path, {})
    context_summaries = _safe_read_json(context_summaries_path, {"summaries": []})
    session_summary = _safe_read_json(session_summary_path, {})
    indexing_log = _safe_read_json(indexing_log_path, {})
    agent_memory_log = _safe_read_json(agent_memory_log_path, {"events": []})
    alert_memory_log = _safe_read_json(alert_memory_log_path, {"events": []})

    extracted_frames = list((ROOT / "data" / "extracted").glob("frame_*.jpg"))

    # ---- Requirement checks ----
    feature_spec_keywords = ["value", "property", "security", "requirement"]
    has_feature_spec = all(k in readme_text.lower() for k in feature_spec_keywords)
    feature_status = "pass" if has_feature_spec else "fail"
    results.append(
        CheckResult(
            id="R1",
            title="Feature spec exists with value + key requirements",
            status=feature_status,
            points_earned=_score(feature_status, 5),
            points_max=5,
            severity="major" if feature_status != "pass" else "info",
            evidence="README lacks explicit feature-spec section with 2-3 numbered requirements." if feature_status != "pass" else "Feature spec language found in README.",
            why_it_matters="Assignment explicitly asks for a short feature spec.",
            recommendation="Add a dedicated 'Feature Spec' section: value proposition + 3 measurable requirements.",
        )
    )

    has_architecture_doc = bool(architecture_text.strip())
    arch_status = "pass" if has_architecture_doc else "fail"
    results.append(
        CheckResult(
            id="R2",
            title="Architecture/design artifact present",
            status=arch_status,
            points_earned=_score(arch_status, 5),
            points_max=5,
            severity="critical" if arch_status != "pass" else "info",
            evidence="docs/architecture.md is empty." if arch_status != "pass" else "Architecture document is populated.",
            why_it_matters="Design/architecture carries report and documentation weight.",
            recommendation="Populate architecture with data flow, storage, alert flow, and reasoning loop.",
        )
    )

    development_checks = [
        bool(telemetry_all),
        bool(analysis_all),
        isinstance(alerts_all, dict) and isinstance(alerts_all.get("alerts", []), list),
        bool(extracted_frames),
    ]
    dev_ratio = sum(1 for x in development_checks if x) / len(development_checks)
    dev_status = _status_from_ratio(dev_ratio)
    results.append(
        CheckResult(
            id="R3",
            title="Prototype development pipeline implemented",
            status=dev_status,
            points_earned=_score(dev_status, 5),
            points_max=5,
            severity="major" if dev_status != "pass" else "info",
            evidence=(
                f"extracted_frames={len(extracted_frames)}, telemetry_records={len(telemetry_all)}, "
                f"analysis_records={len(analysis_all)}, alerts={len(alerts_all.get('alerts', []))}"
            ),
            why_it_matters="Core pipeline functionality is foundational to correctness.",
            recommendation="Ensure all stages are reproducible in a single command/script.",
        )
    )

    has_indexing_code = "def index_frames" in pinecone_indexer_text and "def search_frames" in pinecone_indexer_text
    indexed_total = int(indexing_log.get("total_indexed", 0) or 0)
    index_status = "pass" if has_indexing_code and indexed_total > 0 else "partial" if has_indexing_code else "fail"
    results.append(
        CheckResult(
            id="R4",
            title="Cross-domain frame indexing implemented",
            status=index_status,
            points_earned=_score(index_status, 5),
            points_max=5,
            severity="major" if index_status == "fail" else "info",
            evidence=f"index_frames/search_frames found={has_indexing_code}, total_indexed={indexed_total}",
            why_it_matters="Cross-domain requirement is explicitly mandatory in assignment.",
            recommendation="Keep query examples in report showing object/time retrieval.",
        )
    )

    has_stub_tests = "assert True" in tests_text
    has_api_live_dependency = "http://localhost:8000" in tests_text
    qa_status = "fail" if has_stub_tests else "partial" if has_api_live_dependency else "pass"
    results.append(
        CheckResult(
            id="R5",
            title="QA test cases are robust and reproducible",
            status=qa_status,
            points_earned=_score(qa_status, 5),
            points_max=5,
            severity="critical" if qa_status == "fail" else "major" if qa_status == "partial" else "info",
            evidence="tests/test_agent.py contains stub tests and hard dependency on live localhost API." if qa_status != "pass" else "Test suite appears robust and self-contained.",
            why_it_matters="Assignment requires QA scenarios; weak tests reduce confidence and score.",
            recommendation="Replace stubs with assertive tests and use FastAPI TestClient instead of external server.",
        )
    )

    # ---- Metric 1: Correctness & Model Performance (25) ----
    analysis_count = len(analysis_all) if isinstance(analysis_all, list) else 0
    telemetry_count = len(telemetry_all) if isinstance(telemetry_all, list) else 0
    alerts_count = len(alerts_all.get("alerts", [])) if isinstance(alerts_all, dict) else 0

    analysis_schema_hits = 0
    valid_confidence = 0
    if isinstance(analysis_all, list):
        for row in analysis_all:
            if all(k in row for k in ["frame_id", "vlm_description", "threat_assessment", "confidence"]):
                analysis_schema_hits += 1
            conf = row.get("confidence")
            if isinstance(conf, (int, float)) and 0 <= conf <= 1:
                valid_confidence += 1

    schema_ratio = (analysis_schema_hits / analysis_count) if analysis_count else 0.0
    confidence_ratio = (valid_confidence / analysis_count) if analysis_count else 0.0
    count_alignment = 1.0 if analysis_count and telemetry_count and abs(analysis_count - telemetry_count) <= 2 else 0.0
    correctness_ratio = (schema_ratio + confidence_ratio + count_alignment) / 3
    correctness_points = round(25 * correctness_ratio, 2)

    # ---- Metric 2: Reasoning & Scalability (25) ----
    has_context_summaries = isinstance(context_summaries, dict) and isinstance(context_summaries.get("summaries", []), list)
    summaries_count = len(context_summaries.get("summaries", [])) if has_context_summaries else 0
    has_agent_memory = len(agent_memory_log.get("events", [])) > 0
    has_alert_memory = len(alert_memory_log.get("events", [])) > 0
    has_dedupe_fields = "agent_processed_frames" in agent_text and "alert_processed_frames" in alert_engine_text

    reasoning_ratio = (
        (1.0 if summaries_count > 0 else 0.0)
        + (1.0 if has_agent_memory else 0.0)
        + (1.0 if has_alert_memory else 0.0)
        + (1.0 if has_dedupe_fields else 0.0)
    ) / 4
    reasoning_points = round(25 * reasoning_ratio, 2)

    # ---- Metric 3: Experimentation & Innovation (30) ----
    has_bonus_summary = summarizer_path.exists()
    has_bonus_qa = qa_agent_path.exists()
    has_llm_reasoning_layer = "_generate_reasoning" in agent_text and "_reason_about_alert" in alert_engine_text
    has_cross_domain = has_indexing_code and indexed_total > 0

    innovation_ratio = (
        (1.0 if has_cross_domain else 0.0)
        + (1.0 if has_bonus_summary else 0.0)
        + (1.0 if has_bonus_qa else 0.0)
        + (1.0 if has_llm_reasoning_layer else 0.0)
    ) / 4
    innovation_points = round(30 * innovation_ratio, 2)

    # ---- Metric 4: Documentation & Code Quality (20) ----
    readme_has_setup = "setup" in readme_text.lower() and "run" in readme_text.lower()
    readme_has_ai_tooling = any(k in readme_text.lower() for k in ["claude", "cursor", "windsurf", "ai-assisted", "ai tools"])
    architecture_ready = has_architecture_doc
    tests_not_stubbed = not has_stub_tests

    doc_ratio = (
        (1.0 if readme_has_setup else 0.0)
        + (1.0 if readme_has_ai_tooling else 0.0)
        + (1.0 if architecture_ready else 0.0)
        + (1.0 if tests_not_stubbed else 0.0)
    ) / 4
    doc_points = round(20 * doc_ratio, 2)

    total_score = round(correctness_points + reasoning_points + innovation_points + doc_points, 2)

    # ---- Additional delivery checks ----
    report_pdf_exists = bool(list(ROOT.glob("*.pdf"))) or bool(list((ROOT / "docs").glob("*.pdf")))
    video_exists = bool(list(ROOT.glob("*.mp4"))) or bool(list((ROOT / "demo").glob("*.mp4")))

    delivery_findings: List[Tuple[str, str]] = []
    if not report_pdf_exists:
        delivery_findings.append(("critical", "Submission report PDF is missing from repository."))
    if not video_exists:
        delivery_findings.append(("major", "Demo video artifact not found in repository (can be link-only at submission time)."))
    if not readme_has_ai_tooling:
        delivery_findings.append(("major", "README does not document AI tools used and their impact."))

    # ---- High-priority findings ----
    findings: List[Dict[str, str]] = []
    for check in results:
        if check.status != "pass":
            findings.append(
                {
                    "severity": check.severity,
                    "check_id": check.id,
                    "issue": check.title,
                    "evidence": check.evidence,
                    "improvement": check.recommendation,
                }
            )

    for sev, msg in delivery_findings:
        findings.append(
            {
                "severity": sev,
                "check_id": "D1",
                "issue": msg,
                "evidence": "Repository artifact audit",
                "improvement": "Include missing deliverable before submission or provide accessible link.",
            }
        )

    severity_rank = {"critical": 0, "major": 1, "minor": 2, "info": 3}
    findings.sort(key=lambda x: severity_rank.get(x["severity"], 9))

    top_improvements: List[str] = []
    if isinstance(session_summary, dict) and "raw_response" in session_summary:
        top_improvements.append(
            "Fix summarizer JSON reliability (raw_response fallback still appears); enforce schema + retry parse repair."
        )
    if not report_pdf_exists:
        top_improvements.append(
            "Convert docs/final_report.md to PDF and include it in submission artifacts."
        )
    if not video_exists:
        top_improvements.append(
            "Record voiceover demo video and include a shareable link with correct access permissions."
        )
    if not top_improvements:
        top_improvements.append("Maintain current quality and add performance benchmarking for bonus polish.")

    has_critical = any(item.get("severity") == "critical" for item in findings)
    if has_critical:
        readiness = "ready_with_major_gaps"
    elif total_score >= 85:
        readiness = "mostly_ready"
    elif total_score >= 70:
        readiness = "ready_with_minor_gaps"
    else:
        readiness = "not_ready"

    report: Dict[str, Any] = {
        "evaluation_type": "hard_assignment_audit",
        "rubric_source": "FlytBase AI Engineer assignment requirements + weighted assessment metrics",
        "scores": {
            "correctness_and_model_performance_25": correctness_points,
            "reasoning_and_scalability_25": reasoning_points,
            "experimentation_and_innovation_30": innovation_points,
            "documentation_and_code_quality_20": doc_points,
            "total_100": total_score,
        },
        "requirement_checks": [r.__dict__ for r in results],
        "artifact_snapshot": {
            "extracted_frames": len(extracted_frames),
            "telemetry_records": telemetry_count,
            "analysis_records": analysis_count,
            "alerts_records": alerts_count,
            "indexed_total": indexed_total,
            "context_summaries": summaries_count,
            "session_summary_has_raw_response_only": isinstance(session_summary, dict) and "raw_response" in session_summary,
        },
        "findings_ordered": findings,
        "top_improvements": top_improvements,
        "submission_readiness": readiness,
    }

    return report


def write_reports(report: Dict[str, Any]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    json_path = EVAL_DIR / "hard_evaluation_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    scores = report.get("scores", {})
    findings = report.get("findings_ordered", [])

    lines: List[str] = []
    lines.append("# Hard Evaluation Report")
    lines.append("")
    lines.append("## Overall Score")
    lines.append("")
    lines.append(f"- Total: **{scores.get('total_100', 0)}/100**")
    lines.append(f"- Correctness & Model Performance: {scores.get('correctness_and_model_performance_25', 0)}/25")
    lines.append(f"- Reasoning & Scalability: {scores.get('reasoning_and_scalability_25', 0)}/25")
    lines.append(f"- Experimentation & Innovation: {scores.get('experimentation_and_innovation_30', 0)}/30")
    lines.append(f"- Documentation & Code Quality: {scores.get('documentation_and_code_quality_20', 0)}/20")
    lines.append("")
    lines.append("## Critical and Major Findings")
    lines.append("")

    if not findings:
        lines.append("- No blocking findings detected.")
    else:
        for item in findings:
            if item.get("severity") in {"critical", "major"}:
                lines.append(
                    f"- [{item.get('severity').upper()}] {item.get('issue')} | Evidence: {item.get('evidence')} | Improve: {item.get('improvement')}"
                )

    lines.append("")
    lines.append("## Prioritized Improvement Plan")
    lines.append("")
    for idx, rec in enumerate(report.get("top_improvements", []), start=1):
        lines.append(f"{idx}. {rec}")

    md_path = EVAL_DIR / "hard_evaluation_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report = run_hard_evaluation()
    write_reports(report)
    print(json.dumps(report.get("scores", {}), indent=2))
    print("Saved reports to outputs/evaluation/")


if __name__ == "__main__":
    main()
