"""System fallback node — safe degradation when confidence is insufficient.

Invoked when:
1. The FMEA reasoner confidence is below the operational threshold.
2. The guardrails gateway rejects the generated report.
3. No FMEA entries match the alarm tag.

This node produces a conservative fallback report that:
- Explicitly states the diagnostic uncertainty.
- Recommends manual inspection by a qualified operator.
- Logs the incident for retrospective FMEA knowledge base enrichment.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from src.agent.state import AgentState


def system_fallback_node(state: AgentState) -> Dict[str, Any]:
    """Generate a safe fallback report when AI diagnosis is insufficient.

    Args:
        state: AgentState with ``alarm_signal``, ``confidence``,
            ``fmea_matched``.

    Returns:
        Partial state with ``diagnostic_report`` set to the fallback,
        ``requires_fallback`` = True.
    """
    alarm = state.get("alarm_signal", {})
    tag = alarm.get("tag", "UNKNOWN")
    confidence = state.get("confidence", 0.0)
    attempted_matches = state.get("fmea_matched", [])

    fallback_report: Dict[str, Any] = {
        "alarm_tag": tag,
        "matched_fmea": [],
        "diagnostic_summary": (
            f"⚠ FALLBACK MODE — {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Alarm tag {tag} triggered but AI confidence ({confidence:.2f}) "
            f"is below the operational threshold (0.60).\n"
            f"Attempted FMEA matches: {len(attempted_matches)}.\n\n"
            f"**Recommended action:** Dispatch a qualified operator to "
            f"perform manual inspection of {tag} and associated equipment.\n"
            f"Check the SCADA trend for the preceding 60 minutes for "
            f"any correlated deviations in upstream/downstream tags.\n\n"
            f"This incident will be logged for retrospective FMEA "
            f"knowledge base enrichment."
        ),
        "requires_manual_inspection": True,
        "citation_sources": [],
    }

    return {
        "diagnostic_report": fallback_report,
        "requires_fallback": True,
    }
