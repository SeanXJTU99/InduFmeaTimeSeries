"""Report generator node — assembles the final FMEA diagnostic report.

Takes the ranked FMEA matches from the reasoner and produces:
1. A structured JSON diagnostic report.
2. A human-readable summary for the SCADA/MES operator console.
3. Passes the report through the Guardrails gateway before output.

All output is in the format expected by the Pydantic validator and
constrained decoder (see ``src/safety/``).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from src.agent.state import AgentState


def report_generator_node(state: AgentState) -> Dict[str, Any]:
    """Generate the final FMEA diagnostic report.

    Args:
        state: AgentState with ``fmea_matched``, ``device_context``,
            ``alarm_signal``, and ``confidence``.

    Returns:
        Partial state with ``diagnostic_report`` and
        ``requires_fallback``.
    """
    alarm = state.get("alarm_signal", {})
    matches = state.get("fmea_matched", [])
    confidence = state.get("confidence", 0.0)
    tag = alarm.get("tag", "UNKNOWN")

    # --- Build the report ---
    report_entries: List[Dict[str, Any]] = []
    for m in matches:
        report_entries.append({
            "failure_mode": m.get("failure_mode", "unknown"),
            "severity": m.get("severity", 0),
            "occurrence": m.get("occurrence", 1),
            "detection": m.get("detection", 1),
            "rpn": m.get("rpn", 0),
            "confidence": m.get("confidence", 0.0),
            "cause": m.get("cause", ""),
            "recommended_action": m.get("recommended_action", ""),
        })

    requires_fallback = confidence < 0.6 or len(report_entries) == 0

    summary_lines = [
        f"FMEA Diagnostic Report — {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Trigger: {tag} (value: {alarm.get('value', 'N/A')})",
        f"Matches found: {len(report_entries)}",
        f"Overall confidence: {confidence:.2f}",
    ]
    if requires_fallback:
        summary_lines.append(
            "⚠ CONFIDENCE BELOW THRESHOLD — manual inspection recommended."
        )
    for i, entry in enumerate(report_entries):
        summary_lines.append(
            f"  [{i+1}] {entry['failure_mode']} "
            f"(S={entry['severity']}, O={entry['occurrence']}, D={entry['detection']}, "
            f"RPN={entry['rpn']}, conf={entry['confidence']:.2f})"
        )

    report = {
        "alarm_tag": tag,
        "matched_fmea": report_entries,
        "diagnostic_summary": "\n".join(summary_lines),
        "requires_manual_inspection": requires_fallback,
        "citation_sources": state.get("citation_sources", []),
    }

    return {
        "diagnostic_report": report,
        "requires_fallback": requires_fallback,
    }
