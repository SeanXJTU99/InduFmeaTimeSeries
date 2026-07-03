"""LangGraph StateGraph construction — the orchestrator for FMEA diagnostics.

Builds a directed graph with conditional routing:

    ContextResolver → FMEAReasoner → ReportGenerator
                                          ↓
                                    [confidence check]
                                     ↓           ↓
                                  Reflection   SystemFallback
                                     ↓
                                   (end)

The graph supports checkpointing for human-in-the-loop interruption
and resumption.
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes.context_resolver import context_resolver_node
from src.agent.nodes.fmea_reasoner import fmea_reasoner_node
from src.agent.nodes.report_generator import report_generator_node
from src.agent.nodes.reflection import reflection_node
from src.agent.nodes.system_fallback import system_fallback_node
from src.agent.routing.conditional_edges import (
    confidence_gate,
    feedback_gate,
)


class FMEAAgentGraph:
    """Encapsulates the compiled LangGraph for FMEA diagnostics.

    Usage::

        graph = FMEAAgentGraph()
        app = graph.compile()
        result = app.invoke({
            "alarm_signal": {"tag": "TE-301", "value": 85.0, "source": "PLC"},
            "intent": "fmea_query",
        })
    """

    def __init__(self, checkpointer: object | None = None) -> None:
        """Initialise the graph builder.

        Args:
            checkpointer: optional LangGraph checkpointer (e.g.
                MemorySaver or SqliteSaver) for state persistence
                and human-in-the-loop support.
        """
        self._checkpointer = checkpointer

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def compile(self) -> StateGraph:
        """Build and compile the FMEA agent StateGraph.

        Returns:
            A compiled LangGraph application (runnable).
        """
        workflow = StateGraph(AgentState)

        # --- Add nodes ---
        workflow.add_node("ContextResolver", context_resolver_node)
        workflow.add_node("FMEAReasoner", fmea_reasoner_node)
        workflow.add_node("ReportGenerator", report_generator_node)
        workflow.add_node("Reflection", reflection_node)
        workflow.add_node("SystemFallback", system_fallback_node)

        # --- Edges ---
        workflow.set_entry_point("ContextResolver")
        workflow.add_edge("ContextResolver", "FMEAReasoner")
        workflow.add_edge("FMEAReasoner", "ReportGenerator")

        # Conditional edge: high confidence → end; low → fallback; feedback → reflection
        workflow.add_conditional_edges(
            "ReportGenerator",
            confidence_gate,
            {
                "end": END,
                "fallback": "SystemFallback",
                "reflect": "Reflection",
            },
        )
        workflow.add_edge("SystemFallback", END)
        workflow.add_conditional_edges(
            "Reflection",
            feedback_gate,
            {
                "end": END,
                "retry": "ContextResolver",  # re-enter with updated context
            },
        )

        # Compile with optional checkpointer
        if self._checkpointer is not None:
            return workflow.compile(checkpointer=self._checkpointer)
        return workflow.compile()


def build_graph(checkpointer: object | None = None) -> StateGraph:
    """Convenience: compile the FMEA agent graph in one call.

    Args:
        checkpointer: optional LangGraph checkpointer.

    Returns:
        Compiled StateGraph application.
    """
    return FMEAAgentGraph(checkpointer).compile()
