"""
Deterministic Annealing Filter for industrial time-series Kalman filtering.

The DAF wraps a standard Kalman filter with an annealing schedule that
automatically suppresses outlier measurements via Bayesian weights.
At high inverse temperature (beta >> 1), all measurements are weighted
equally (converging to standard Kalman). As beta cools, measurements with
large chi-squared residuals receive exponentially suppressed weights,
effectively removing them from the state update without a separate
pre-filtering stage.

Reference: R. Fruehwirth & A. Strandlie, CPC 120 (1999) 197-214.

This replaces the two-stage Kalman + wavelet cascade with a single unified
filter — the annealing mechanism handles high-frequency noise spikes
internally, eliminating the need for wavelet pre-denoising.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DAFConfig:
    """DAF annealing configuration.

    Attributes:
        beta_start: Initial inverse temperature (high = equal weights).
        beta_final: Final inverse temperature (low = outlier suppression).
        n_steps: Number of annealing steps in logarithmic schedule.
        prob_cut: Chi-squared cutoff probability for the penalty term.
        delta_weight: Convergence criterion on max weight change.
        max_iterations: Safety limit on total DAF iterations.
    """
    beta_start: float = 100.0
    beta_final: float = 0.1
    n_steps: int = 5
    prob_cut: float = 0.001
    delta_weight: float = 1e-3
    max_iterations: int = 4


@dataclass
class DAFState:
    """Per-measurement state during DAF annealing.

    Attributes:
        residual: Measurement residual (z - H*x_pred).
        chi2: Chi-squared of the residual.
        weight: Bayesian annealing weight in [0, 1].
        inlier: Final classification after annealing.
    """
    residual: np.ndarray
    chi2: float = 0.0
    weight: float = 1.0
    inlier: bool = True


class DAFKalmanFilter:
    """Deterministic Annealing Filter wrapping a standard Kalman update.

    Usage::

        daf = DAFKalmanFilter(DAFConfig(beta_start=100, beta_final=0.1, n_steps=5))
        kf = StandardKalman(Q=1e-4, R=0.04)

        for measurement in plc_stream:
            state, is_outlier = daf.update(kf, measurement, dt)
            if is_outlier:
                continue  # suppressed, not passed to downstream detection
    """

    def __init__(self, config: Optional[DAFConfig] = None):
        self.config = config or DAFConfig()
        self._betas: list[float] = []
        self._compute_annealing_schedule()

    def _compute_annealing_schedule(self) -> None:
        """Generate logarithmic annealing sequence.

        beta_i = beta_start * (beta_final / beta_start) ^ (i / (n_steps - 1))
        """
        cfg = self.config
        if cfg.n_steps == 1:
            self._betas = [cfg.beta_final]
            return
        for i in range(cfg.n_steps):
            frac = i / (cfg.n_steps - 1)
            beta = cfg.beta_start * (cfg.beta_final / cfg.beta_start) ** frac
            self._betas.append(beta)

    #
    # --- Core DAF update ---------------------------------------------------
    #

    def update(
        self,
        kalman_state: "KalmanState",
        measurement: float,
        measurement_variance: float,
        dt: float = 1.0,
    ) -> tuple["KalmanState", bool]:
        """Perform full DAF annealing update on a single measurement.

        Args:
            kalman_state: Current Kalman filter state (x, P, Q, R).
            measurement: New sensor reading.
            measurement_variance: Variance of the measurement (R).
            dt: Time delta since last update.

        Returns:
            (updated_state, is_outlier): Updated Kalman state and whether
            this measurement was classified as an outlier.
        """
        n_measurements = 1  # single-channel update
        daf_states = [DAFState(residual=np.array([0.0]))]

        prev_weights = np.ones(n_measurements)

        for iteration in range(self.config.max_iterations):
            beta = (
                self._betas[iteration]
                if iteration < len(self._betas)
                else self._betas[-1]
            )

            # --- Standard Kalman predict + update ---
            x_pred, p_pred = kalman_state.predict(dt)

            residual = measurement - x_pred
            innov_cov = p_pred + measurement_variance
            chi2 = (residual**2) / innov_cov if innov_cov > 1e-12 else 0.0

            daf_states[0].residual = np.array([residual])
            daf_states[0].chi2 = chi2

            # --- Bayesian annealing weight computation ---
            self._calc_weights(daf_states, beta, innov_cov)

            # --- Weighted Kalman update ---
            weight = daf_states[0].weight
            if weight > 0.0:
                k_gain = (p_pred * weight) / (
                    p_pred * weight + measurement_variance
                )
                x_new = x_pred + k_gain * residual
                p_new = (1.0 - k_gain) * p_pred
            else:
                # Outlier suppressed: skip update, predict only
                x_new = x_pred
                p_new = p_pred

            kalman_state.x = x_new
            kalman_state.P = p_new

            # --- Convergence check ---
            curr_weights = np.array([s.weight for s in daf_states])
            if np.max(np.abs(curr_weights - prev_weights)) < self.config.delta_weight:
                break
            prev_weights = curr_weights.copy()

        is_outlier = daf_states[0].weight < 0.1
        return kalman_state, is_outlier

    def _calc_weights(
        self,
        states: list[DAFState],
        beta: float,
        innov_cov: float,
    ) -> None:
        """Compute Bayesian annealing weights.

        w_j = phi_j / (sum(phi) + phi_cut)

        where:
          phi_j = (2*pi)^(-d/2) * |V|^(-1/2) * exp(-chi2_j / (2*beta))
          phi_cut = same formula evaluated at the chi2 cutoff
        """
        cfg = self.config
        d = 1  # scalar measurement

        norm = (2.0 * math.pi) ** (-d / 2.0) * (innov_cov ** (-0.5))
        chi2_cut = -2.0 * math.log(max(cfg.prob_cut, 1e-12))

        phi_total = 0.0
        phis = []

        for state in states:
            chi2 = state.chi2
            chi2_clamped = min(chi2 / (2.0 * beta), 50.0)
            phi = norm * math.exp(-chi2_clamped)
            phis.append(phi)
            phi_total += phi

        cut_val = chi2_cut / (2.0 * beta)
        phi_cut = norm * math.exp(-min(cut_val, 50.0))

        denom = phi_total + phi_cut
        if denom < 1e-12:
            for state in states:
                state.weight = 0.0
            return

        for i, state in enumerate(states):
            state.weight = phis[i] / denom

    @property
    def annealing_schedule(self) -> list[float]:
        """Return the computed beta schedule for inspection."""
        return self._betas.copy()


class KalmanState:
    """Minimal 1D Kalman filter state for DAF integration.

    In production, swap with the full KalmanFilter from kalman_filter.py.
    """

    def __init__(
        self,
        x0: float = 0.0,
        P0: float = 1.0,
        Q: float = 1e-5,
        R: float = 1e-3,
    ):
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R

    def predict(self, dt: float = 1.0) -> tuple[float, float]:
        """Predict step: x_pred = x, P_pred = P + Q*dt."""
        p_pred = self.P + self.Q * dt
        return self.x, p_pred


def compute_annealing_schedule(
    beta_start: float = 100.0,
    beta_final: float = 0.1,
    n_steps: int = 5,
) -> list[float]:
    """Standalone: generate logarithmic annealing schedule."""
    if n_steps == 1:
        return [beta_final]
    betas = []
    for i in range(n_steps):
        frac = i / (n_steps - 1)
        beta = beta_start * (beta_final / beta_start) ** frac
        betas.append(beta)
    return betas
