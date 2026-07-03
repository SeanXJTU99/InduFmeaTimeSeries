"""Counterfactual advisor — "what if" reasoning for operator decision support.

After MCTS simulates thousands of fault-hypothesis trajectories, this
module generates human-readable counterfactual advice:

    "If the root cause is lubricant degradation, the bearing temperature
     will rise another 5 °C in 60 minutes.  If instead the cause is
     sensor drift, the temperature will oscillate with ±2 °C amplitude.
     Recommend dispatch vibration analysis to disambiguate."

All values and scenarios are fictitious.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CounterfactualConfig:
    """Counterfactual advisor parameters."""

    max_hypotheses: int = 3
    prediction_horizon_min: int = 45  # minutes ahead
    confidence_threshold: float = 0.6


class CounterfactualAdvisor:
    """Generate counterfactual control advice from MCTS simulation results.

    Usage::

        advisor = CounterfactualAdvisor()
        advice = advisor.advise([
            {"hypothesis": "lubricant_degradation", "confidence": 0.85, "predicted_dT": 5.0},
            {"hypothesis": "sensor_drift", "confidence": 0.62, "predicted_dT": 2.0},
        ])
    """

    def __init__(self, config: CounterfactualConfig | None = None) -> None:
        self._cfg = config or CounterfactualConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advise(self, hypotheses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate counterfactual advice from ranked hypotheses.

        Args:
            hypotheses: list of dicts with ``hypothesis``, ``confidence``,
                and optional ``predicted_*`` fields.

        Returns:
            Dict with ``summary`` (str), ``recommendations`` (list of str),
            and ``disambiguation_action`` (str).
        """
        # Sort by confidence descending
        ranked = sorted(
            hypotheses, key=lambda h: h.get("confidence", 0.0), reverse=True
        )[: self._cfg.max_hypotheses]

        summary_lines: List[str] = [
            f"Counterfactual analysis — {len(ranked)} hypotheses ranked:"
        ]
        for i, h in enumerate(ranked):
            summary_lines.append(
                f"  [{i + 1}] {h['hypothesis']} "
                f"(confidence: {h.get('confidence', 0):.2f})"
            )

        recommendations: List[str] = []
        if ranked:
            top = ranked[0]
            if top.get("confidence", 0) >= self._cfg.confidence_threshold:
                recommendations.append(
                    f"Primary hypothesis '{top['hypothesis']}' exceeds confidence "
                    f"threshold.  Recommend executing associated FMEA action: "
                    f"{top.get('fmea_action', 'manual inspection')}."
                )
            else:
                recommendations.append(
                    "No hypothesis exceeds confidence threshold.  "
                    "Recommend additional sensor data collection before action."
                )

        # Disambiguation: if top 2 are close, suggest a discriminating test
        if len(ranked) >= 2:
            conf_gap = ranked[0].get("confidence", 0) - ranked[1].get("confidence", 0)
            if conf_gap < 0.15:
                recommendations.append(
                    f"Hypotheses '{ranked[0]['hypothesis']}' and "
                    f"'{ranked[1]['hypothesis']}' are close in confidence "
                    f"(gap={conf_gap:.2f}).  Suggested disambiguation: "
                    f"dispatch technician for on-site inspection of "
                    f"associated equipment within 2 hours."
                )

        return {
            "summary": "\n".join(summary_lines),
            "recommendations": recommendations,
            "disambiguation_action": (
                recommendations[-1] if len(recommendations) > 1
                else recommendations[0] if recommendations
                else "Insufficient data for advice."
            ),
        }


def generate_advice(hypotheses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convenience: generate counterfactual advice in one call."""
    return CounterfactualAdvisor().advise(hypotheses)
