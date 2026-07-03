"""Context summarizer — compresses accumulated agent state for long sessions.

When the agent runs for many turns (e.g. an 8-hour shift with hundreds of
alarms), the full message history exceeds the LLM context window.  This
module provides a lightweight extractive summarizer that condenses older
interactions into a compact digest while preserving key diagnostic facts.

In production, this calls the LLM itself for abstractive summarization.
The implementation here uses a deterministic extractive fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SummaryConfig:
    """Summarizer parameters."""

    max_summary_chars: int = 800
    keep_keywords: List[str] = (
        "alarm", "fault", "failure", "FMEA", "RPN", "severity",
        "shutdown", "manual", "inspection", "confidence",
    )


class ContextSummarizer:
    """Extractive conversation summarizer for agent history.

    Usage::

        summarizer = ContextSummarizer()
        digest = summarizer.summarize(message_history)
    """

    def __init__(self, config: SummaryConfig | None = None) -> None:
        self._cfg = config or SummaryConfig()

    def summarize(self, messages: List[Dict[str, str]]) -> str:
        """Condense a message history into a short digest.

        Args:
            messages: list of role/content dicts.

        Returns:
            A paragraph-form summary string.
        """
        if not messages:
            return "No prior history."

        key_sentences: List[str] = []
        for msg in messages:
            content = msg.get("content", "")
            for sentence in content.replace("\n", ". ").split(". "):
                sentence = sentence.strip()
                if not sentence:
                    continue
                for kw in self._cfg.keep_keywords:
                    if kw.lower() in sentence.lower():
                        key_sentences.append(sentence)
                        break

        if not key_sentences:
            # fallback: keep last 3 message snippets
            key_sentences = [
                m.get("content", "")[:200]
                for m in messages[-3:]
            ]

        digest = " | ".join(key_sentences)
        if len(digest) > self._cfg.max_summary_chars:
            digest = digest[: self._cfg.max_summary_chars - 3] + "…"
        return digest

    def summarize_state(self, state: Dict[str, Any]) -> str:
        """Create a one-paragraph digest of the current agent state.

        Args:
            state: current AgentState dict.

        Returns:
            Compact summary string.
        """
        parts: List[str] = []
        alarm = state.get("alarm_signal", {})
        if alarm.get("tag"):
            parts.append(f"Last alarm: {alarm['tag']}={alarm.get('value', '?')}")
        matches = state.get("fmea_matched", [])
        if matches:
            parts.append(f"Top FMEA: {matches[0].get('failure_mode', '?')} "
                         f"(conf={matches[0].get('confidence', 0):.2f})")
        if state.get("confidence"):
            parts.append(f"Confidence: {state['confidence']:.2f}")
        if state.get("requires_fallback"):
            parts.append("⚠ Fallback triggered")
        return " | ".join(parts) if parts else "No active diagnosis."


def summarize_history(messages: List[Dict[str, str]]) -> str:
    """Convenience: summarize message history in one call."""
    return ContextSummarizer().summarize(messages)
