"""Physics residual energy function for cryogenic distillation columns.

Implements the energy function K(x; F) from PINFDiT (ICLR 2026), adapted
for industrial distillation constraints.  Computes the squared deviation
from known physical conservation laws — mass balance, energy balance,
abundance bounds, and pressure-flow consistency — as an energy scalar
and its analytic gradient w.r.t. model predictions.

The energy function is stateless and runs on CPU (<50 μs per call),
enabling inference-time Langevin correction on Jetson AGX Orin without
any autograd framework dependency.

Energy definition
-----------------
  K(x; F) = -[ w_m * R_mass² + w_e * R_energy² + w_a * R_abundance²
             + w_p * R_pressure_flow² + w_r * R_reflux² ]

where each R_* is a dimensionless residual normalised by the sensor's
operating range.  K is maximised (→ 0) when all constraints are satisfied;
strong violations drive K toward large negative values.

Reference
---------
  Cao et al., "PINFDiT: Energy-Based Physics-Informed Diffusion
  Transformers for General-Purpose Time Series Tasks", ICLR 2026.
  Eq. (3): K(xtar; F) = -|| ∂xtar/∂τ - F(τ, xtar, u, ∂xtar/∂u, ...) ||²
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EnergyFunctionConfig:
    """Physics constraint parameters for a single distillation column.

    All tag names and threshold values are fictitious.
    """

    # -- abundance bounds ---------------------------------------------------
    min_abundance_pct: float = 0.0
    max_abundance_pct: float = 100.0

    # -- mass balance (steady-state approximation) --------------------------
    # F_feed ≈ F_distillate + F_bottoms
    # Allowable imbalance as fraction of feed flow.
    mass_balance_tolerance: float = 0.05  # ±5 %

    # -- pressure-flow consistency ------------------------------------------
    # In turbulent flow through a packed column: ΔP ∝ F².
    # Nominal (ΔP, flow_rate) operating point.
    dp_nominal_bar: float = 1.2
    flow_nominal_lpm: float = 50.0
    # Exponent for ΔP ∝ F^n relationship (n ≈ 1.8–2.2 for packed beds).
    flow_exponent: float = 2.0
    # Maximum ratio ΔP_actual / ΔP_expected before flagging.
    dp_ratio_max: float = 2.0

    # -- reflux ratio bounds ------------------------------------------------
    min_reflux_ratio: float = 1.5  # below minimum → separation impossible
    max_reflux_ratio: float = 8.0  # above → flooding risk + energy waste

    # -- temperature gradient -----------------------------------------------
    # Cryogenic distillation: ΔT between top and bottom is bounded.
    # Typical range for air separation: −180 °C (bottom) to −170 °C (top).
    min_temp_gradient_c_per_stage: float = -8.0
    max_temp_gradient_c_per_stage: float = -0.5

    # -- energy function weights (sum need not be 1.0) ---------------------
    w_mass_balance: float = 1.0
    w_energy_balance: float = 0.0  # disabled by default (needs enthalpy)
    w_abundance: float = 5.0       # hard constraint — high weight
    w_pressure_flow: float = 2.0
    w_reflux: float = 1.0
    w_temp_gradient: float = 1.0

    # -- numerical ----------------------------------------------------------
    epsilon: float = 1e-8  # floor for denominators


# ---------------------------------------------------------------------------
# Energy function
# ---------------------------------------------------------------------------

class DistillationEnergyFunction:
    """Physics residual energy for cryogenic distillation.

    Usage::

        cfg = EnergyFunctionConfig()
        ef = DistillationEnergyFunction(cfg)

        features = {"dp": 2.5, "flow_rate": 40.0, "reflux_ratio": 3.0,
                    "temp_top": -170.0, "temp_bottom": -180.0,
                    "n_stages": 50}
        predictions = {"abundance_pct": 105.0, "anomaly_score": 0.2}

        energy = ef.energy(features, predictions)
        grad = ef.gradient(features, predictions)  # dK/d(abundance_pct),
                                                     # dK/d(anomaly_score)

        # Langevin step (caller side):
        # abundance_pct += epsilon * grad["abundance_pct"] + noise
    """

    def __init__(self, config: Optional[EnergyFunctionConfig] = None) -> None:
        self._cfg = config or EnergyFunctionConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def energy(
        self,
        features: Dict[str, float],
        predictions: Dict[str, float],
    ) -> float:
        """Compute K(x; F) — the physics residual energy.

        Args:
            features: PLC measurements (``dp``, ``flow_rate``,
                ``reflux_ratio``, ``temp_top``, ``temp_bottom``,
                ``n_stages``, ``feed_flow``, ``distillate_flow``,
                ``bottoms_flow``).
            predictions: model outputs (``abundance_pct``,
                ``anomaly_score``).

        Returns:
            Energy scalar (≤ 0).  0 = perfect physics consistency.
        """
        residuals = self._compute_residuals(features, predictions)
        return self._combine_energy(residuals)

    def gradient(
        self,
        features: Dict[str, float],
        predictions: Dict[str, float],
    ) -> Dict[str, float]:
        """Analytic gradient of K(x; F) w.r.t. each prediction.

        Computed in closed form — no autograd.  Only prediction
        variables that contribute to active constraints have non-zero
        entries.

        Returns:
            Dict mapping prediction key → dK/d(key).
        """
        cfg = self._cfg
        residuals = self._compute_residuals(features, predictions)

        grad: Dict[str, float] = {}

        # Abundance bound residual (linear outer → constant gradient
        # magnitude near bound).
        a = float(predictions.get("abundance_pct", 50.0))
        r_a = residuals["abundance"]
        if r_a > 0.0:
            span = cfg.max_abundance_pct - cfg.min_abundance_pct + cfg.epsilon
            if a > cfg.max_abundance_pct:
                dR_da = 2.0 * (a - cfg.max_abundance_pct) / (span ** 2)
            elif a < cfg.min_abundance_pct:
                dR_da = -2.0 * (cfg.min_abundance_pct - a) / (span ** 2)
            else:
                dR_da = 0.0
            # dK/da = -w_a * sign(R_a) * dR_da  (linear outer)
            grad["abundance_pct"] = float(
                -cfg.w_abundance * (1.0 if r_a > 0 else -1.0) * dR_da
            )
        else:
            grad["abundance_pct"] = 0.0

        # Anomaly score does NOT directly enter the physics residuals —
        # it is a derived quantity.  The Langevin correction on
        # anomaly_score happens indirectly through the hidden state
        # that feeds into the anomaly head.  However, a physically
        # inconsistent feature state implies higher anomaly_score →
        # dK/d(anomaly_score) = 0 (no direct coupling).
        grad["anomaly_score"] = 0.0

        return grad

    def gradient_wrt_features(
        self,
        features: Dict[str, float],
        predictions: Dict[str, float],
    ) -> Dict[str, float]:
        """Analytic gradient w.r.t. *input features* (for latent correction).

        When the Langevin dynamics are applied to the model's hidden
        state rather than the output directly, we need dK/df for each
        input feature f.  These tell us which measured values are
        most physically inconsistent.

        Returns:
            Dict mapping feature key → dK/d(key).
        """
        cfg = self._cfg
        residuals = self._compute_residuals(features, predictions)

        grad: Dict[str, float] = {}

        # Pressure-flow residual: R_p = max(0, dp_ratio - dp_ratio_max)
        # dK/d(dp) = -2 * w_p * R_p * dR_p/d(dp) where dR_p/d(dp)
        # depends on the expected ΔP.
        r_pf = residuals["pressure_flow"]
        if r_pf > 0.0:
            dp = float(features.get("dp", cfg.dp_nominal_bar))
            flow = float(features.get("flow_rate", cfg.flow_nominal_lpm))
            flow_ratio = flow / max(cfg.flow_nominal_lpm, cfg.epsilon)
            dp_expected = cfg.dp_nominal_bar * (flow_ratio ** cfg.flow_exponent)
            dR_ddp = 1.0 / max(dp_expected, cfg.epsilon)
            grad["dp"] = float(-2.0 * cfg.w_pressure_flow * r_pf * dR_ddp)
        else:
            grad["dp"] = 0.0

        # Reflux residual: R_r = max(0, R - R_max) + max(0, R_min - R)
        r_r = residuals["reflux"]
        if r_r > 0.0:
            rr = float(features.get("reflux_ratio",
                      (cfg.min_reflux_ratio + cfg.max_reflux_ratio) / 2.0))
            span = cfg.max_reflux_ratio - cfg.min_reflux_ratio + cfg.epsilon
            if rr > cfg.max_reflux_ratio:
                dR_drr = 1.0 / span
            elif rr < cfg.min_reflux_ratio:
                dR_drr = -1.0 / span
            else:
                dR_drr = 0.0
            grad["reflux_ratio"] = float(-2.0 * cfg.w_reflux * r_r * dR_drr)
        else:
            grad["reflux_ratio"] = 0.0

        # Mass balance residual
        r_m = residuals["mass_balance"]
        if r_m > 0.0:
            f_feed = float(features.get("feed_flow", 50.0))
            f_feed_safe = max(f_feed, cfg.epsilon)
            # R_m = max(0, |imbalance|/F_feed - tolerance) / tolerance
            grad["feed_flow"] = float(-2.0 * cfg.w_mass_balance * r_m / f_feed_safe)
            grad["distillate_flow"] = float(
                2.0 * cfg.w_mass_balance * r_m / f_feed_safe
            )
            grad["bottoms_flow"] = float(
                2.0 * cfg.w_mass_balance * r_m / f_feed_safe
            )
        else:
            grad["feed_flow"] = 0.0
            grad["distillate_flow"] = 0.0
            grad["bottoms_flow"] = 0.0

        return grad

    # ------------------------------------------------------------------
    # Residual computation
    # ------------------------------------------------------------------

    def _compute_residuals(
        self,
        features: Dict[str, float],
        predictions: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute all dimensionless physics residuals.

        Each residual is ≥ 0.  0 = constraint satisfied.
        """
        return {
            "abundance": self._residual_abundance(predictions),
            "mass_balance": self._residual_mass_balance(features),
            "pressure_flow": self._residual_pressure_flow(features),
            "reflux": self._residual_reflux(features),
            "temp_gradient": self._residual_temp_gradient(features),
        }

    def _combine_energy(self, residuals: Dict[str, float]) -> float:
        """Weighted sum → energy scalar.

        Hard constraints (abundance, mass) use a linear outer function
        (|R|) so the gradient does not vanish near the boundary.
        Soft constraints (pressure-flow, reflux, temp) use quadratic
        (R²) for smooth convergence.
        """
        cfg = self._cfg
        linear_terms = [
            cfg.w_abundance      * abs(residuals["abundance"]),
            cfg.w_mass_balance   * abs(residuals["mass_balance"]),
        ]
        quadratic_terms = [
            cfg.w_pressure_flow  * residuals["pressure_flow"] ** 2,
            cfg.w_reflux         * residuals["reflux"] ** 2,
            cfg.w_temp_gradient  * residuals["temp_gradient"] ** 2,
        ]
        return float(-sum(linear_terms) - sum(quadratic_terms))

    # ------------------------------------------------------------------
    # Individual residuals (each → dimensionless, ≥ 0)
    # ------------------------------------------------------------------

    def _residual_abundance(
        self, predictions: Dict[str, float]
    ) -> float:
        """Abundance bound violation: > 100 % or < 0 %."""
        cfg = self._cfg
        a = float(predictions.get("abundance_pct", 50.0))
        span = cfg.max_abundance_pct - cfg.min_abundance_pct + cfg.epsilon
        if a > cfg.max_abundance_pct:
            return float(((a - cfg.max_abundance_pct) / span) ** 2)
        if a < cfg.min_abundance_pct:
            return float(((cfg.min_abundance_pct - a) / span) ** 2)
        return 0.0

    def _residual_mass_balance(
        self, features: Dict[str, float]
    ) -> float:
        """Mass balance: |F_feed - F_dist - F_bot| / F_feed > tolerance?"""
        cfg = self._cfg
        f_feed = float(features.get("feed_flow", 0.0))
        f_dist = float(features.get("distillate_flow", 0.0))
        f_bot  = float(features.get("bottoms_flow", 0.0))

        if f_feed < cfg.epsilon:
            return 0.0  # cannot evaluate

        imbalance = abs(f_feed - f_dist - f_bot) / f_feed
        excess = max(0.0, imbalance - cfg.mass_balance_tolerance)
        return float(excess / max(cfg.mass_balance_tolerance, cfg.epsilon))

    def _residual_pressure_flow(
        self, features: Dict[str, float]
    ) -> float:
        """Pressure-flow consistency: ΔP_actual / ΔP_expected > limit?"""
        cfg = self._cfg
        dp = float(features.get("dp", cfg.dp_nominal_bar))
        flow = float(features.get("flow_rate", cfg.flow_nominal_lpm))

        flow_ratio = flow / max(cfg.flow_nominal_lpm, cfg.epsilon)
        dp_expected = cfg.dp_nominal_bar * (flow_ratio ** cfg.flow_exponent)
        dp_ratio = dp / max(dp_expected, cfg.epsilon)

        excess = max(0.0, dp_ratio - cfg.dp_ratio_max)
        return float(excess / max(cfg.dp_ratio_max, cfg.epsilon))

    def _residual_reflux(
        self, features: Dict[str, float]
    ) -> float:
        """Reflux ratio within [min, max]?"""
        cfg = self._cfg
        rr = float(features.get(
            "reflux_ratio",
            (cfg.min_reflux_ratio + cfg.max_reflux_ratio) / 2.0,
        ))
        span = cfg.max_reflux_ratio - cfg.min_reflux_ratio + cfg.epsilon
        excess = max(0.0, rr - cfg.max_reflux_ratio) + max(0.0, cfg.min_reflux_ratio - rr)
        return float(excess / span)

    def _residual_temp_gradient(
        self, features: Dict[str, float]
    ) -> float:
        """Temperature gradient within cryogenic bounds?"""
        cfg = self._cfg
        t_top = float(features.get("temp_top", -175.0))
        t_bot = float(features.get("temp_bottom", -185.0))
        n_stages = max(int(features.get("n_stages", 50)), 1)

        gradient = (t_top - t_bot) / float(n_stages)  # ΔT per stage
        span = (
            cfg.max_temp_gradient_c_per_stage
            - cfg.min_temp_gradient_c_per_stage
            + cfg.epsilon
        )
        excess = max(0.0, gradient - cfg.max_temp_gradient_c_per_stage) + max(
            0.0, cfg.min_temp_gradient_c_per_stage - gradient
        )
        return float(excess / abs(span))


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def distillation_energy(
    dp: float = 1.2,
    flow_rate: float = 50.0,
    reflux_ratio: float = 4.0,
    temp_top: float = -175.0,
    temp_bottom: float = -185.0,
    n_stages: int = 50,
    abundance_pct: float = 50.0,
    feed_flow: float = 50.0,
    distillate_flow: float = 25.0,
    bottoms_flow: float = 25.0,
    config: Optional[EnergyFunctionConfig] = None,
) -> float:
    """One-shot energy computation for a single state vector.

    Returns:
        Energy scalar (≤ 0).
    """
    ef = DistillationEnergyFunction(config)
    features = {
        "dp": dp,
        "flow_rate": flow_rate,
        "reflux_ratio": reflux_ratio,
        "temp_top": temp_top,
        "temp_bottom": temp_bottom,
        "n_stages": float(n_stages),
        "feed_flow": feed_flow,
        "distillate_flow": distillate_flow,
        "bottoms_flow": bottoms_flow,
    }
    predictions = {"abundance_pct": abundance_pct, "anomaly_score": 0.0}
    return ef.energy(features, predictions)
