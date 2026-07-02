"""Physics-informed anomaly detector for cryogenic distillation columns.

Rather than feeding raw process variables (PVs) directly to an ML model,
this module computes physically meaningful derived features — ΔP, reflux
ratio, cascade temperature gradients — that encode distillation-column
health.  Anomalies are scored by comparing these features against their
expected ranges under normal operating conditions.

All tag names and thresholds are fictitious.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class PhysicsDetectorConfig:
    """Configuration for physics-informed anomaly detection."""

    # Nominal operating ranges (fictitious column T-301)
    dp_nominal: Tuple[float, float] = (0.5, 2.0)  # bar — top-bottom ΔP
    reflux_ratio_nominal: Tuple[float, float] = (2.5, 6.0)  # L/D
    temp_gradient_nominal: Tuple[float, float] = (-5.0, -1.0)  # °C/stage

    # Tolerances (how many standard deviations before flagging)
    n_sigma: float = 3.0

    # Feature weights in the combined anomaly score
    weights: Dict[str, float] = field(default_factory=lambda: {
        "dp_deviation": 0.35,
        "reflux_deviation": 0.25,
        "temp_gradient_deviation": 0.25,
        "flooding_index": 0.15,
    })


class PhysicsInformedDetector:
    """Score anomalies using distillation physics constraints.

    Computes four key indicators:

    1. **ΔP deviation** — pressure drop across the column.  Elevated ΔP
       at low flow signals flooding.
    2. **Reflux ratio deviation** — L/D ratio outside design bounds.
    3. **Temperature gradient anomaly** — cascade gradient (ΔT/stage)
       deviating from the tight cryogenic profile.
    4. **Flooding index** — combined indicator: high ΔP + low flow →
       potential flooding precursor.

    Usage::

        det = PhysicsInformedDetector()
        score, breakdown = det.score(features)
        if score > det.config.n_sigma:
            raise_alarm()
    """

    def __init__(self, config: PhysicsDetectorConfig | None = None) -> None:
        self.config = config or PhysicsDetectorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Compute the combined physics-informed anomaly score.

        Args:
            features: dict with keys ``dp`` (bar), ``reflux_ratio``,
                ``temp_gradient`` (°C/stage), ``flow_rate`` (L/min).

        Returns:
            (combined_score, breakdown) where *breakdown* maps each
            indicator name to its individual z-score.
        """
        breakdown = {
            "dp_deviation": self._zscore(
                features.get("dp", 1.0), self.config.dp_nominal
            ),
            "reflux_deviation": self._zscore(
                features.get("reflux_ratio", 4.0), self.config.reflux_ratio_nominal
            ),
            "temp_gradient_deviation": self._zscore(
                features.get("temp_gradient", -3.0), self.config.temp_gradient_nominal
            ),
            "flooding_index": self._flooding_index(features),
        }
        combined = sum(
            self.config.weights[k] * abs(v) for k, v in breakdown.items()
        )
        return combined, breakdown

    def is_anomalous(self, features: Dict[str, float]) -> bool:
        """Return True if the combined score exceeds the sigma threshold."""
        score, _ = self.score(features)
        return score > self.config.n_sigma

    def explain(self, features: Dict[str, float]) -> List[str]:
        """Return human-readable explanations for top contributing features.

        Args:
            features: same dict as :meth:`score`.

        Returns:
            List of explanation strings, most significant first.
        """
        _, breakdown = self.score(features)
        sorted_items = sorted(breakdown.items(), key=lambda x: abs(x[1]), reverse=True)
        explanations: List[str] = []
        for name, z in sorted_items:
            if abs(z) > 1.0:
                direction = "high" if z > 0 else "low"
                explanations.append(f"{name}: {direction} deviation (z={z:.2f})")
        return explanations

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _zscore(value: float, nominal_range: Tuple[float, float]) -> float:
        lo, hi = nominal_range
        center = (lo + hi) / 2.0
        half_span = (hi - lo) / 2.0
        if half_span == 0:
            return 0.0
        return (value - center) / half_span

    @staticmethod
    def _flooding_index(features: Dict[str, float]) -> float:
        """Empirical flooding precursor indicator.

        High ΔP combined with low (or falling) flow rate is a classic
        sign of incipient flooding in a packed distillation column.
        """
        dp = features.get("dp", 1.0)
        flow = features.get("flow_rate", 50.0)
        # Normalised: high dp + low flow → high index
        dp_norm = dp / 2.0  # divide by nominal max
        flow_norm = flow / 50.0  # divide by nominal
        if flow_norm < 0.01:
            flow_norm = 0.01
        return dp_norm / flow_norm - 1.0
