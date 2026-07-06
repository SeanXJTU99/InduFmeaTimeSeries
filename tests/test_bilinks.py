"""Tests for FMEA Bilinks Causal Graph Retrieval."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.rag.fmea_bilinks import FMEABilinksGraph


@pytest.fixture
def sample_graph():
    """Minimal FMEA bilinks graph with two isolated subsystems."""
    g = FMEABilinksGraph()

    # System 1: Centrifugal pump bearing
    te101 = g.add_node("sensor", "TE-101", "Centrifugal Pump #1", {"severity": 8})
    fm_heat = g.add_node("failure_mode", "Bearing overheat", "Pump #1", {"severity": 8, "rpn": 192})
    rc_oil = g.add_node("root_cause", "Oil deficiency", "Pump #1")
    mt_oil = g.add_node("mitigation", "Check and refill oil", "Pump #1")

    g.add_bilink(te101, fm_heat, "observes", weight=0.8)
    g.add_bilink(fm_heat, rc_oil, "caused_by", weight=0.9)
    g.add_bilink(rc_oil, mt_oil, "mitigated_by", weight=0.8)

    # System 2: Distillation tower (isolated from System 1)
    te201 = g.add_node("sensor", "TE-201", "Distillation Tower #2 Top", {"severity": 6})
    fm_flood = g.add_node("failure_mode", "Packing flood", "Tower #2", {"severity": 9})
    rc_valve = g.add_node("root_cause", "FV-201 valve stuck", "Tower #2")
    mt_valve = g.add_node("mitigation", "Switch to backup valve", "Tower #2")

    g.add_bilink(te201, fm_flood, "observes", weight=0.9)
    g.add_bilink(fm_flood, rc_valve, "caused_by", weight=0.85)
    g.add_bilink(rc_valve, mt_valve, "mitigated_by", weight=0.7)

    return g


class TestBilinksGraph:

    def test_node_count(self, sample_graph):
        assert sample_graph.node_count == 8

    def test_bilink_count(self, sample_graph):
        # 3 bilinks x 2 directions x 2 systems = 12 directed edges
        assert sample_graph.bilink_count == 12

    def test_causal_search_cross_system_isolation(self, sample_graph):
        """TE-101 search must NOT return TE-201's failure modes."""
        results = sample_graph.causal_search("TE-101", max_depth=3)
        tags = {r["tag"] for r in results}

        assert "Bearing overheat" in tags
        assert "Oil deficiency" in tags
        assert "Check and refill oil" in tags
        # Cross-system contamination
        assert "Packing flood" not in tags
        assert "FV-201 valve stuck" not in tags

    def test_max_depth_respected(self, sample_graph):
        results = sample_graph.causal_search("TE-101", max_depth=1)
        tags = {r["tag"] for r in results}
        assert "Bearing overheat" in tags       # depth 1
        assert "Oil deficiency" not in tags     # depth 2
        assert "Check and refill oil" not in tags  # depth 3

    def test_unknown_sensor_returns_empty(self, sample_graph):
        results = sample_graph.causal_search("TE-999", max_depth=3)
        assert results == []

    def test_context_string_contains_causal_chain(self, sample_graph):
        ctx = sample_graph.causal_search_with_context("TE-101", max_depth=3)
        assert "TE-101" in ctx
        assert "Bearing overheat" in ctx
        assert "Oil deficiency" in ctx
        assert "Check and refill oil" in ctx
        assert "BFS" in ctx

    def test_context_string_unknown_sensor(self, sample_graph):
        ctx = sample_graph.causal_search_with_context("TE-999")
        assert "No causal entries found" in ctx
        assert "Manual inspection" in ctx

    def test_subgraph_tags_isolated(self, sample_graph):
        tags1 = sample_graph.get_subgraph_tags("TE-101", max_depth=3)
        tags2 = sample_graph.get_subgraph_tags("TE-201", max_depth=3)
        assert len(tags1 & tags2) == 0

    def test_connected_components(self, sample_graph):
        components = sample_graph.connected_components()
        assert len(components) == 2
        for comp in components:
            assert len(comp) == 4

    def test_load_from_fmea_matrix(self):
        g = FMEABilinksGraph()
        rows = [
            {
                "tag": "PT-301",
                "system": "Reactor #3",
                "failure_mode": "Pressure spike",
                "root_cause": "Cooling valve failure",
                "mitigation": "Check cooling loop",
                "severity": 9,
                "rpn": 270,
            },
            {
                "tag": "TT-302",
                "system": "Reactor #3",
                "failure_mode": "Temperature runaway",
                "root_cause": "Heater element failure",
                "mitigation": "Switch heater",
                "severity": 10,
                "rpn": 300,
            },
        ]
        count = g.load_from_fmea_matrix(rows)
        assert count == 6  # 3 bilinks x 2 rows
        assert g.node_count == 8  # 4 unique tags per row


class TestBilinksVsBM25:
    """Verify bilinks avoids BM25 semantic false positives."""

    def test_cross_system_no_contamination(self, sample_graph):
        te101_results = sample_graph.causal_search("TE-101", max_depth=3)
        te101_tags = {r["tag"] for r in te101_results}

        te201_results = sample_graph.causal_search("TE-201", max_depth=3)
        te201_tags = {r["tag"] for r in te201_results}

        assert "Packing flood" in te201_tags
        assert "Packing flood" not in te101_tags
        assert "Bearing overheat" in te101_tags
        assert "Bearing overheat" not in te201_tags
