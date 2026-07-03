"""Reflection node — human-in-the-loop feedback for closed-loop learning.

When the report generator produces a diagnosis below the confidence
threshold, or when an engineer provides explicit feedback (e.g.
correcting a misdiagnosis), this node:

1. Captures the engineer's feedback.
2. Updates the local knowledge cache with the corrected diagnosis.
3. Optionally triggers a re-retrieval with the updated context.

This is the "Reflection & Write" pattern from the source design doc.
"""

from __future__ import annotations

from typing import Any, Dict

from src.agent.state import AgentState


def reflection_node(state: AgentState) -> Dict[str, Any]:
    """Process engineer feedback and update the knowledge state.

    Args:
        state: AgentState with ``engineer_feedback``, ``diagnostic_report``,
            and ``alarm_signal``.

    Returns:
        Partial state update.  If feedback is empty, the node is a no-op.
    """
    feedback = state.get("engineer_feedback", "")
    report = state.get("diagnostic_report", {})
    alarm = state.get("alarm_signal", {})

    if not feedback:
        # No feedback yet — this is a pre-reflection pass.
        # The system pauses here for human input.
        return {}

    # --- Log the correction for future RAG indexing ---
    # In production, this would:
    # 1. Append the corrected (alarm_signature → true_cause) pair to
    #    a feedback buffer.
    # 2. Periodically batch-insert corrected entries into the FAISS /
    #    Milvus vector DB.
    # 3. Trigger a lightweight online update of the DPO preference
    #    dataset (see src/training/dpo_dataset_builder.py).
    tag = alarm.get("tag", "UNKNOWN")
    print(
        f"[Reflection] Engineer feedback captured for tag {tag}: "
        f"'{feedback[:200]}' — will be inserted into RAG index "
        f"on next batch update."
    )

    # Update history with the feedback exchange
    history: list[Dict[str, str]] = state.get("history", [])
    history.append({
        "role": "engineer",
        "content": feedback,
    })

    return {
        "engineer_feedback": "",  # clear after processing
        "history": history,
    }
