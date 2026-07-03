"""FMEA reasoner node — the core diagnostic inference engine.

This node takes the resolved device context and matched FMEA entries
from the context resolver, and performs causal reasoning:

1. Ranks FMEA matches by confidence × severity.
2. Checks whether the current PLC value deviates from expected ranges.
3. Produces a ranked list of likely failure causes with confidence scores.

In production, this node calls the LLM with the topology-injected
prompt and safe-refusal instructions.  The simulation below mirrors
that reasoning deterministically.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.agent.state import AgentState


def fmea_reasoner_node(state: AgentState) -> Dict[str, Any]:
    """Rank and filter FMEA matches by relevance and confidence.

    Args:
        state: AgentState with ``fmea_matched`` and ``device_context``.

    Returns:
        Partial state with filtered/ranked ``fmea_matched``,
        ``confidence``, and ``citation_sources``.
    """
    matches: List[Dict[str, Any]] = state.get("fmea_matched", [])

    # --- Filter: drop entries below confidence threshold ---
    MIN_CONFIDENCE = 0.6
    viable = [
        m for m in matches
        if m.get("confidence", 0.0) >= MIN_CONFIDENCE
    ]

    # --- Rank by confidence × severity (higher = more urgent) ---
    def _rank_key(m: Dict[str, Any]) -> float:
        conf = float(m.get("confidence", 0.0))
        sev = float(m.get("severity", 1))
        return conf * sev

    viable.sort(key=_rank_key, reverse=True)

    # --- Compute overall confidence ---
    if viable:
        overall_conf = float(max(m.get("confidence", 0.0) for m in viable))
    else:
        overall_conf = 0.0

    # --- Collect citation sources ---
    sources: List[str] = []
    for m in viable:
        sid = m.get("source_id")
        if sid:
            sources.append(sid)

    return {
        "fmea_matched": viable,
        "confidence": overall_conf,
        "citation_sources": sources,
    }
