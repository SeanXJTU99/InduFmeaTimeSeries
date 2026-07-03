"""Intent classifier — routes incoming messages to the correct agent mode.

The FMEA agent can handle multiple intent types:
- ``fmea_query`` — alarm-driven diagnostic request (main path).
- ``status_check`` — operator asking about current system state.
- ``knowledge_query`` — general question about FMEA or SOP content.
- ``unknown`` — fallback; triggers safe refusal.

In production this would be a lightweight fine-tuned classifier or an
LLM few-shot call.  The simulation below uses simple keyword heuristics.
"""

from __future__ import annotations

from typing import Any, Dict, List


# Keywords associated with each intent (fictitious examples).
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "fmea_query": [
        "alarm", "anomaly", "abnormal", "fault", "failure",
        "exceed", "threshold", "trip", "warning", "high",
        "low", "deviation", "spike", "drop", "drift",
    ],
    "status_check": [
        "status", "current", "reading", "value", "what is",
        "show", "display", "report", "overview", "summary",
    ],
    "knowledge_query": [
        "what does", "how to", "explain", "procedure", "manual",
        "definition", "SOP", "standard", "guideline",
    ],
}


class IntentClassifier:
    """Lightweight keyword-based intent classifier.

    Usage::

        ic = IntentClassifier()
        intent, score = ic.classify("TE-301 bearing temperature alarm high")
        # → ("fmea_query", 0.67)
    """

    def __init__(
        self, keywords: Dict[str, List[str]] | None = None
    ) -> None:
        self._keywords = keywords or INTENT_KEYWORDS

    def classify(self, text: str) -> tuple[str, float]:
        """Classify the intent of a natural-language query.

        Args:
            text: the user/operator query string.

        Returns:
            ``(intent_label, confidence)`` tuple.  Confidence is the
            fraction of matched keywords relative to the best intent.
        """
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for intent, kws in self._keywords.items():
            scores[intent] = sum(1 for kw in kws if kw.lower() in text_lower)

        total_hits = sum(scores.values())
        if total_hits == 0:
            return "unknown", 0.0

        best_intent = max(scores, key=lambda k: scores[k])  # type: ignore[arg-type]
        confidence = scores[best_intent] / max(total_hits, 1)
        return best_intent, confidence


def classify_intent(text: str) -> tuple[str, float]:
    """Convenience: classify query intent in one call."""
    return IntentClassifier().classify(text)
