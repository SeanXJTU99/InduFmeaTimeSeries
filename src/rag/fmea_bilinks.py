#!/usr/bin/env python3
"""
FMEA Bilinks Causal Graph Retrieval.

Builds a bidirectional link graph from the FMEA matrix where nodes represent
sensors, failure modes, root causes, and mitigation actions. Edges encode
causal relationships: observes, caused_by, mitigated_by.

Retrieval walks the graph via BFS from an alarming sensor, constrained to
the bilinks topology. This guarantees that only causally-related entries
are returned, unlike BM25 semantic search which can match semantically
similar but causally unrelated entries (e.g., "Tower 1 top temperature
high" matching "Tower 2 bottom reboiler temperature high").

BM25+BGE vector search is retained as a fallback for novel failure modes
not yet encoded in the bilinks graph.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BilinkNode:
    """A node in the FMEA causal graph.

    Attributes:
        node_id: Unique integer identifier.
        node_type: "sensor", "failure_mode", "root_cause", or "mitigation".
        tag: Human-readable label (e.g. "TE-101", "Bearing overheat").
        system: Parent system name (e.g. "Centrifugal Pump #1").
        metadata: Arbitrary dict (severity, rpn, etc.).
    """
    node_id: int
    node_type: str
    tag: str
    system: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Bilink:
    """A bidirectional causal link between two FMEA entities.

    Attributes:
        from_node: Source node ID.
        to_node: Target node ID.
        relation: "observes", "caused_by", "mitigated_by".
        weight: Edge weight (confidence or RPN contribution).
    """
    from_node: int
    to_node: int
    relation: str
    weight: float = 1.0


class FMEABilinksGraph:
    """Causal graph for FMEA knowledge retrieval.

    Usage::

        graph = FMEABilinksGraph()
        te101 = graph.add_node("sensor", "TE-101", "Centrifugal Pump #1")
        fm = graph.add_node("failure_mode", "Bearing overheat", "Centrifugal Pump #1",
                            metadata={"severity": 8})
        rc = graph.add_node("root_cause", "Oil deficiency", "Centrifugal Pump #1")
        mt = graph.add_node("mitigation", "Check oil level", "Centrifugal Pump #1")

        graph.add_bilink(te101, fm, "observes")
        graph.add_bilink(fm, rc, "caused_by")
        graph.add_bilink(rc, mt, "mitigated_by")

        results = graph.causal_search(seed_tag="TE-101", max_depth=3)
    """

    def __init__(self):
        self.nodes: dict[int, BilinkNode] = {}
        self._tag_index: dict[str, int] = {}
        self._adjacency: dict[int, list[Bilink]] = {}
        self._next_id: int = 0

    #
    # --- Graph construction -------------------------------------------------
    #

    def add_node(
        self,
        node_type: str,
        tag: str,
        system: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """Add a node and return its ID."""
        node_id = self._next_id
        self._next_id += 1
        self.nodes[node_id] = BilinkNode(
            node_id=node_id,
            node_type=node_type,
            tag=tag,
            system=system,
            metadata=metadata or {},
        )
        self._tag_index[tag] = node_id
        self._adjacency.setdefault(node_id, [])
        return node_id

    def add_bilink(
        self,
        from_id: int,
        to_id: int,
        relation: str,
        weight: float = 1.0,
    ) -> None:
        """Add a bidirectional causal link (two directed edges)."""
        if from_id not in self.nodes or to_id not in self.nodes:
            raise ValueError(
                f"Bilink nodes must be registered first: {from_id}<->{to_id}"
            )
        self._adjacency[from_id].append(
            Bilink(from_node=from_id, to_node=to_id, relation=relation, weight=weight)
        )
        reverse_relation = {
            "observes": "observed_by",
            "caused_by": "causes",
            "mitigated_by": "mitigates",
        }.get(relation, f"reverse_{relation}")
        self._adjacency[to_id].append(
            Bilink(from_node=to_id, to_node=from_id, relation=reverse_relation, weight=weight)
        )

    def load_from_fmea_matrix(self, fmea_rows: list[dict]) -> int:
        """Bulk-load FMEA rows into the bilinks graph.

        Each row expects: tag, system, failure_mode, root_cause, mitigation,
        severity, rpn.

        Returns number of bilinks created.
        """
        bilink_count = 0
        for row in fmea_rows:
            tag = row["tag"]
            system = row.get("system", "Unknown")
            fm = row.get("failure_mode", "")
            rc = row.get("root_cause", "")
            mt = row.get("mitigation", "")
            severity = row.get("severity", 1)

            sensor_id = self._get_or_create("sensor", tag, system, row)
            fm_id = self._get_or_create("failure_mode", fm, system, row)
            rc_id = self._get_or_create("root_cause", rc, system, row)
            mt_id = self._get_or_create("mitigation", mt, system, row)

            self.add_bilink(sensor_id, fm_id, "observes", weight=severity / 10.0)
            bilink_count += 1
            self.add_bilink(fm_id, rc_id, "caused_by", weight=0.9)
            bilink_count += 1
            self.add_bilink(rc_id, mt_id, "mitigated_by", weight=0.8)
            bilink_count += 1

        return bilink_count

    def _get_or_create(
        self, node_type: str, tag: str, system: str, metadata: dict
    ) -> int:
        if tag in self._tag_index:
            return self._tag_index[tag]
        return self.add_node(node_type, tag, system, metadata)

    #
    # --- Causal Search (BFS on Bilinks) ------------------------------------
    #

    def causal_search(
        self,
        seed_tag: str,
        max_depth: int = 3,
        min_weight: float = 0.0,
    ) -> list[dict]:
        """BFS from a sensor tag, constrained to bilinks topology.

        Args:
            seed_tag: Tag of the alarming sensor (e.g. "TE-101").
            max_depth: Maximum BFS depth from seed.
            min_weight: Minimum bilink weight threshold.

        Returns:
            List of dicts with node info and bfs_depth. Empty list if the
            seed tag is unknown (caller should fall back to BM25).
        """
        if seed_tag not in self._tag_index:
            return []

        seed_id = self._tag_index[seed_tag]
        visited: set[int] = {seed_id}
        queue: deque[tuple[int, int]] = deque([(seed_id, 0)])
        results: list[dict] = []

        while queue:
            current_id, depth = queue.popleft()
            if depth > 0:
                node = self.nodes[current_id]
                results.append({
                    "node_id": current_id,
                    "node_type": node.node_type,
                    "tag": node.tag,
                    "system": node.system,
                    "bfs_depth": depth,
                    "metadata": node.metadata,
                })
            if depth >= max_depth:
                continue
            for bilink in self._adjacency.get(current_id, []):
                if bilink.to_node not in visited and bilink.weight >= min_weight:
                    visited.add(bilink.to_node)
                    queue.append((bilink.to_node, depth + 1))

        return results

    def causal_search_with_context(self, seed_tag: str, max_depth: int = 3) -> str:
        """Return a context string for LLM prompt injection."""
        results = self.causal_search(seed_tag, max_depth)

        if not results:
            return (
                f"[FMEA Bilinks] No causal entries found for sensor {seed_tag}. "
                f"This may be a newly-added asset. Manual inspection recommended."
            )

        by_type: dict[str, list[dict]] = {}
        for r in results:
            by_type.setdefault(r["node_type"], []).append(r)

        lines = [
            f"[FMEA Bilinks Causal Retrieval] Seed: {seed_tag}, BFS depth <= {max_depth}:"
        ]
        for ntype in ["failure_mode", "root_cause", "mitigation"]:
            if ntype in by_type:
                type_label = {
                    "failure_mode": "Potential Failure Mode",
                    "root_cause": "Potential Root Cause",
                    "mitigation": "Recommended Mitigation",
                }.get(ntype, ntype)
                lines.append(f"\n  {type_label}:")
                for r in by_type[ntype]:
                    sev = r["metadata"].get("severity", "")
                    sev_str = f" [S={sev}]" if sev else ""
                    lines.append(f"    - {r['tag']}{sev_str} ({r['system']})")

        return "\n".join(lines)

    def get_subgraph_tags(self, seed_tag: str, max_depth: int = 3) -> set[str]:
        """Return all tags in the causal subgraph of a seed tag.

        Used to validate BM25 results: if a BM25 hit's tag is NOT in this set,
        it is a semantic false positive and should be dropped.
        """
        results = self.causal_search(seed_tag, max_depth)
        return {r["tag"] for r in results}

    #
    # --- Diagnostics --------------------------------------------------------
    #

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def bilink_count(self) -> int:
        return sum(len(links) for links in self._adjacency.values())

    def connected_components(self) -> list[set[int]]:
        """Find isolated causal subsystems via DFS.

        Each component represents an independent causal subsystem
        (e.g., separate distillation towers do not share bilinks).
        """
        visited: set[int] = set()
        components: list[set[int]] = []

        for node_id in self.nodes:
            if node_id not in visited:
                component: set[int] = set()
                stack = [node_id]
                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    component.add(current)
                    for bilink in self._adjacency.get(current, []):
                        if bilink.to_node not in visited:
                            stack.append(bilink.to_node)
                components.append(component)

        return components
