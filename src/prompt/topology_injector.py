"""Topology injector — injects distillation column connectivity graph into prompts.

Multi-stage cryogenic distillation columns are cascaded: T-301's top
vapor feeds T-302's inlet, T-302's bottoms feed T-303, etc.  When the
LLM sees an alarm on T-302, it needs to know its upstream (T-301) and
downstream (T-303) neighbors to perform meaningful causal reasoning.

This module injects the column topology as a set of RDF-like triples
into the system prompt header, along with a stern instruction: "If you
cannot derive the FMEA cause from the topology and retrieved context,
answer 'Unknown — manual inspection required'.  Never speculate beyond
the provided context."

All topology data is fictitious.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Fictitious cryogenic distillation column topology.
# Format: (subject, predicate, object)
DEFAULT_TOPOLOGY: List[tuple[str, str, str]] = [
    # Feed → T-301
    ("Feed_System_F-101", "feeds", "Cryogenic_Column_T-301"),
    # T-301 internals
    ("Cryogenic_Column_T-301", "has_top_pressure_tag", "PT-301"),
    ("Cryogenic_Column_T-301", "has_bottom_pressure_tag", "PT-302"),
    ("Cryogenic_Column_T-301", "has_top_temp_tag", "TE-301"),
    ("Cryogenic_Column_T-301", "has_bottom_temp_tag", "TE-302"),
    ("Cryogenic_Column_T-301", "has_feed_valve", "FV-301"),
    ("Cryogenic_Column_T-301", "has_reflux_valve", "FV-302"),
    ("Cryogenic_Column_T-301", "has_feed_flow_tag", "FT-301"),
    # T-301 → T-302 cascade
    ("Cryogenic_Column_T-301", "vapor_to", "Cryogenic_Column_T-302"),
    ("Cryogenic_Column_T-301", "liquid_to", "Reboiler_H-301"),
    # T-302 internals
    ("Cryogenic_Column_T-302", "has_top_pressure_tag", "PT-303"),
    ("Cryogenic_Column_T-302", "has_top_temp_tag", "TE-303"),
    ("Cryogenic_Column_T-302", "has_bottom_temp_tag", "TE-304"),
    ("Cryogenic_Column_T-302", "has_feed_valve", "FV-303"),
    # T-302 → T-303 cascade
    ("Cryogenic_Column_T-302", "vapor_to", "Cryogenic_Column_T-303"),
    # Heat exchanger linkage
    ("Heat_Exchanger_E-301", "cools", "Cryogenic_Column_T-301"),
    ("Heat_Exchanger_E-301", "has_temp_tag", "TE-305"),
    # Reboiler
    ("Reboiler_H-301", "heats", "Cryogenic_Column_T-302"),
    ("Reboiler_H-301", "has_temp_tag", "TE-306"),
    # Condenser
    ("Condenser_C-301", "condenses", "Cryogenic_Column_T-303"),
    ("Condenser_C-301", "has_pressure_tag", "PT-304"),
]


class TopologyInjector:
    """Build topology-enhanced system prompts for the FMEA Agent.

    Usage::

        injector = TopologyInjector()
        prompt = injector.build_prompt(alarm_tag="TE-301", rag_context=chunks)
    """

    SAFETY_PREAMBLE = (
        "You are an industrial FMEA diagnostic agent for a multi-stage "
        "cryogenic distillation system.  Below is the column topology "
        "(equipment connectivity graph).  ALL claims MUST be backed by "
        "the provided RAG context and topology.  If you cannot derive a "
        "cause from the given information, respond: "
        "'Unknown — manual inspection required.'  "
        "NEVER speculate beyond the provided context.\n\n"
    )

    def __init__(
        self, topology: List[tuple[str, str, str]] | None = None
    ) -> None:
        self._topology = topology or DEFAULT_TOPOLOGY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        alarm_tag: str,
        rag_context: str,
        plc_values: Dict[str, float] | None = None,
    ) -> str:
        """Construct a full system prompt with topology + RAG context.

        Args:
            alarm_tag: the PLC tag that triggered the alarm.
            rag_context: concatenated RAG retrieval results.
            plc_values: optional current PLC readings for context.

        Returns:
            Complete system prompt string.
        """
        parts = [self.SAFETY_PREAMBLE]

        # --- topology ---
        neighbors = self.get_neighbors(alarm_tag)
        parts.append("## Column Topology (relevant neighbors)\n")
        for subj, pred, obj in self._topology:
            if alarm_tag.replace("-", "_") in subj or alarm_tag.replace("-", "_") in obj:
                parts.append(f"- ({subj}) --[{pred}]--> ({obj})\n")
        if neighbors["upstream"]:
            parts.append(f"\nUpstream equipment: {', '.join(neighbors['upstream'])}\n")
        if neighbors["downstream"]:
            parts.append(f"Downstream equipment: {', '.join(neighbors['downstream'])}\n")

        # --- current PLC values ---
        if plc_values:
            parts.append("\n## Current PLC Readings\n")
            for tag, val in plc_values.items():
                parts.append(f"- {tag}: {val}\n")

        # --- RAG context ---
        parts.append("\n## Retrieved FMEA Knowledge (MUST cite sources)\n")
        parts.append(rag_context)

        # --- final instruction ---
        parts.append(
            "\n## Instructions\n"
            "1. Analyse the alarm in the context of the topology and RAG results.\n"
            "2. Match against FMEA entries. Cite each match with [Source: <id>].\n"
            "3. If no match is found, state: 'Unknown — manual inspection required.'\n"
            "4. Output in valid JSON matching the FMEA report schema.\n"
        )
        return "".join(parts)

    def get_neighbors(self, tag: str) -> Dict[str, List[str]]:
        """Return upstream/downstream equipment for a given tag.

        Args:
            tag: PLC tag or equipment ID.

        Returns:
            Dict with ``upstream`` and ``downstream`` lists.
        """
        normalized = tag.replace("-", "_")
        upstream: List[str] = []
        downstream: List[str] = []
        for subj, pred, obj in self._topology:
            if normalized in obj:
                upstream.append(subj)
            if normalized in subj:
                downstream.append(obj)
        return {"upstream": list(set(upstream)), "downstream": list(set(downstream))}

    def to_triples_text(self) -> str:
        """Return the full topology as a triple-text block."""
        lines = ["## Full Topology"]
        for s, p, o in self._topology:
            lines.append(f"- ({s}) --[{p}]--> ({o})")
        return "\n".join(lines)


def inject_topology(
    alarm_tag: str,
    rag_context: str,
    plc_values: Dict[str, float] | None = None,
) -> str:
    """Convenience: build a topology-injected prompt in one call."""
    return TopologyInjector().build_prompt(alarm_tag, rag_context, plc_values)
