"""AgentState TypedDict — the shared state object flowing through the LangGraph.

Every node reads from and writes to this typed dictionary.  LangGraph
serialises it between node invocations, enabling checkpointing, replay,
and human-in-the-loop interruption.

All field values are fictitious examples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state for the FMEA diagnostic agent graph.

    Attributes:
        alarm_signal: the raw alarm payload from PLC / serial stream.
        device_context: resolved equipment info from long-context / RAG.
        fmea_matched: FMEA entries matched by the hybrid search.
        diagnostic_report: the generated FMEA diagnostic report (JSON-serialisable dict).
        engineer_feedback: human operator feedback for closed-loop learning.
        intent: classified intent (``'fmea_query'``, ``'status_check'``, ``'unknown'``).
        confidence: overall LLM confidence in the diagnosis [0, 1].
        requires_fallback: set to True when guardrails or confidence trigger safe fallback.
        citation_sources: list of source IDs cited in the report.
        history: condensed conversation/message history for multi-turn context.
    """

    # --- Input ---
    alarm_signal: Dict[str, Any]
    intent: str

    # --- Intermediate ---
    device_context: str
    fmea_matched: List[Dict[str, Any]]
    confidence: float
    citation_sources: List[str]

    # --- Output ---
    diagnostic_report: Dict[str, Any]
    requires_fallback: bool

    # --- Feedback loop ---
    engineer_feedback: str

    # --- Context ---
    history: List[Dict[str, str]]
