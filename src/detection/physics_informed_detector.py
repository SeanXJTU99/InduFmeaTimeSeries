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
    flow_nominal: float = 50.0  # L/min — nominal feed flow
    dp_max: float = 2.0  # bar — nominal dp upper bound for flooding index

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
        self._energy_func = None  # lazy init

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

        Uses the energy function gradient (if available) to attribute
        the anomaly score to individual physics constraints.  Falls back
        to z-score breakdown when the energy function is unavailable.

        Args:
            features: same dict as :meth:`score`.

        Returns:
            List of explanation strings, most significant first.
        """
        # Prefer energy-based attribution.
        try:
            grad_f = self._energy_gradient_wrt_features(features)
            if grad_f:
                sorted_items = sorted(
                    grad_f.items(), key=lambda x: abs(x[1]), reverse=True
                )
                explanations: List[str] = []
                for name, g in sorted_items[:5]:
                    if abs(g) > 0.0:
                        direction = "high" if g < 0 else "low"
                        explanations.append(
                            f"{name}: {direction} deviation (energy grad={g:.4f})"
                        )
                return explanations or self._zscore_explain(features)
        except Exception:
            pass
        return self._zscore_explain(features)

    def energy_score(
        self,
        features: Dict[str, float],
        predictions: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute anomaly score using the physics residual energy function.

        Maps the raw energy K(x; F) ∈ (-∞, 0] to an anomaly score in [0, ∞)
        via score = -K (higher → more physically inconsistent).

        Args:
            features: PLC measurements.
            predictions: optional model outputs (``abundance_pct``,
                ``anomaly_score``).  Uses nominal defaults if omitted.

        Returns:
            (energy_score, energy_gradient_dict).
        """
        if self._energy_func is None:
            from src.detection.energy_function import (
                DistillationEnergyFunction,
            )
            self._energy_func = DistillationEnergyFunction()

        preds: Dict[str, float] = {
            "abundance_pct": 50.0,
            "anomaly_score": 0.0,
        }
        if predictions is not None:
            preds.update(predictions)

        energy = self._energy_func.energy(features, preds)
        grad = self._energy_func.gradient(features, preds)
        return float(-energy), grad

    def _zscore_explain(self, features: Dict[str, float]) -> List[str]:
        _, breakdown = self.score(features)
        sorted_items = sorted(breakdown.items(), key=lambda x: abs(x[1]), reverse=True)
        explanations: List[str] = []
        for name, z in sorted_items:
            if abs(z) > 1.0:
                direction = "high" if z > 0 else "low"
                explanations.append(f"{name}: {direction} deviation (z={z:.2f})")
        return explanations

    def _energy_gradient_wrt_features(
        self, features: Dict[str, float]
    ) -> Dict[str, float]:
        if self._energy_func is None:
            from src.detection.energy_function import (
                DistillationEnergyFunction,
            )
            self._energy_func = DistillationEnergyFunction()
        return self._energy_func.gradient_wrt_features(
            features,
            {"abundance_pct": 50.0, "anomaly_score": 0.0},
        )

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

    def _flooding_index(self, features: Dict[str, float]) -> float:
        """Empirical flooding precursor indicator.

        High ΔP combined with low (or falling) flow rate is a classic
        sign of incipient flooding in a packed distillation column.
        """
        dp = features.get("dp", 1.0)
        flow = features.get("flow_rate", self.config.flow_nominal)
        dp_norm = dp / self.config.dp_max
        flow_norm = flow / self.config.flow_nominal
        if flow_norm < 0.01:
            flow_norm = 0.01
        return dp_norm / flow_norm - 1.0
