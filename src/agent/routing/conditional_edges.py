"""Conditional edge functions for the LangGraph agent.

These functions are called at decision points in the graph to route
the state to the next node.  They return a string key that maps to
one of the possible next nodes defined in the graph.
"""

from __future__ import annotations

from typing import Any, Dict

from src.agent.state import AgentState


def confidence_gate(state: AgentState) -> str:
    """Route after ReportGenerator based on confidence and feedback.

    Returns:
        ``'end'`` — report is ready, terminate.
        ``'fallback'`` — confidence too low, route to safe degradation.
        ``'reflect'`` — engineer feedback is pending, enter reflection.
    """
    if state.get("requires_fallback", False):
        return "fallback"

    feedback = state.get("engineer_feedback", "")
    if feedback:
        return "reflect"

    return "end"


def feedback_gate(state: AgentState) -> str:
    """Route after Reflection — retry or terminate.

    Returns:
        ``'end'`` — feedback processed, terminate.
        ``'retry'`` — re-enter the graph for an updated diagnosis.
    """
    feedback = state.get("engineer_feedback", "")
    if feedback:
        # Fresh feedback was just processed — could retry.
        # For now, end after one reflection cycle.
        return "end"
    return "end"
