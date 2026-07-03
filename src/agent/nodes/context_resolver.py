"""Context resolver node — the entry point of the agent graph.

Given an alarm signal (PLC tag + value + timestamp), this node:
1. Normalises the alarm payload.
2. Queries the hybrid RAG search for matching FMEA entries.
3. Resolves the equipment context from the topology graph.
4. Populates ``device_context`` and ``fmea_matched`` in AgentState.

All tag names and values are fictitious.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.agent.state import AgentState
from src.prompt.topology_injector import TopologyInjector


def context_resolver_node(state: AgentState) -> Dict[str, Any]:
    """Resolve equipment and FMEA context from an alarm signal.

    This is the first node in the graph.  In production, it calls the
    hybrid searcher, metadata filter, and reranker from the RAG module.
    Here we provide a deterministic simulation path for portfolio demo.

    Args:
        state: current AgentState with at least ``alarm_signal``.

    Returns:
        Partial state update with ``device_context`` and ``fmea_matched``.
    """
    alarm = state.get("alarm_signal", {})
    tag = alarm.get("tag", "UNKNOWN")
    value = alarm.get("value", "N/A")
    source = alarm.get("source", "UNKNOWN")

    # --- Resolve topology context ---
    injector = TopologyInjector()
    neighbors = injector.get_neighbors(tag)
    context_lines = [
        f"Alarm source: {source}",
        f"Triggering tag: {tag}",
        f"Current value: {value}",
    ]
    if neighbors["upstream"]:
        context_lines.append(f"Upstream: {', '.join(neighbors['upstream'])}")
    if neighbors["downstream"]:
        context_lines.append(f"Downstream: {', '.join(neighbors['downstream'])}")

    # --- Simulated hybrid search (in production: HybridSearcher.search) ---
    fmea_matched = _simulate_fmea_search(tag, value)
    context_lines.append(f"\nFMEA matches retrieved: {len(fmea_matched)}")

    return {
        "device_context": "\n".join(context_lines),
        "fmea_matched": fmea_matched,
    }


def _simulate_fmea_search(tag: str, value: Any) -> List[Dict[str, Any]]:
    """Deterministic simulation of RAG hybrid search for demo purposes.

    In production, this is replaced by::

        from src.rag.hybrid_search import HybridSearcher
        searcher = HybridSearcher()
        results = searcher.search(query, metadata_filter={"tag": tag})

    All entries below are fictitious.
    """
    # Simulated FMEA knowledge base keyed by tag prefix
    simulated_fmea: Dict[str, List[Dict[str, Any]]] = {
        "TE": [
            {
                "failure_mode": "bearing_overheat",
                "severity": 8,
                "occurrence": 4,
                "detection": 6,
                "rpn": 192,
                "confidence": 0.85,
                "cause": "Lubricant degradation due to extended high-load operation",
                "recommended_action": "Check lubricant level and quality. Schedule bearing inspection within 24 hours.",
                "source_id": "fmea_042",
            },
            {
                "failure_mode": "sensor_drift",
                "severity": 3,
                "occurrence": 5,
                "detection": 4,
                "rpn": 60,
                "confidence": 0.62,
                "cause": "Thermocouple aging causing zero-point drift",
                "recommended_action": "Cross-validate with redundant sensor. Recalibrate if deviation > 2°C.",
                "source_id": "fmea_017",
            },
        ],
        "PT": [
            {
                "failure_mode": "flooding",
                "severity": 9,
                "occurrence": 3,
                "detection": 5,
                "rpn": 135,
                "confidence": 0.82,
                "cause": "Excessive vapor load causing liquid entrainment",
                "recommended_action": "Reduce feed flow by 10%. Increase reflux ratio. Monitor ΔP trend for 30 min.",
                "source_id": "fmea_018",
            },
        ],
        "FT": [
            {
                "failure_mode": "valve_stiction",
                "severity": 6,
                "occurrence": 4,
                "detection": 7,
                "rpn": 168,
                "confidence": 0.78,
                "cause": "Control valve stem friction due to packing degradation",
                "recommended_action": "Check valve positioner feedback. Perform stroke test during next maintenance window.",
                "source_id": "fmea_055",
            },
        ],
        "FV": [
            {
                "failure_mode": "valve_stiction",
                "severity": 7,
                "occurrence": 3,
                "detection": 6,
                "rpn": 126,
                "confidence": 0.80,
                "cause": "Valve packing friction or actuator diaphragm leak",
                "recommended_action": "Inspect actuator diaphragm. Schedule packing replacement.",
                "source_id": "fmea_056",
            },
        ],
        "LT": [
            {
                "failure_mode": "dry_bed",
                "severity": 8,
                "occurrence": 2,
                "detection": 5,
                "rpn": 80,
                "confidence": 0.71,
                "cause": "Insufficient liquid level due to downstream valve malfunction",
                "recommended_action": "Verify downstream valve positions. Check for level transmitter calibration drift.",
                "source_id": "fmea_031",
            },
        ],
    }

    prefix = tag[:2]  # e.g. "TE" from "TE-301"
    return simulated_fmea.get(prefix, [{"failure_mode": "unknown", "confidence": 0.0, "cause": "No FMEA entry found for this tag prefix.", "source_id": "fmea_000"}])
