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

        # Freeze the prediction baseline — all annealing iterations must
        # compute residuals from the same (x_pred, p_pred) to avoid
        # state contamination across beta steps.
        x_pred_base, p_pred_base = kalman_state.predict(dt)

        for iteration in range(self.config.max_iterations):
            beta = (
                self._betas[iteration]
                if iteration < len(self._betas)
                else self._betas[-1]
            )

            # Residual computed from frozen prediction
            residual = measurement - x_pred_base
            innov_cov = p_pred_base + measurement_variance
            chi2 = (residual**2) / innov_cov if innov_cov > 1e-12 else 0.0

            daf_states[0].residual = np.array([residual])
            daf_states[0].chi2 = chi2

            # Recompute weights under current beta
            self._calc_weights(daf_states, beta, innov_cov)

            # --- Weighted Kalman update (applied to frozen base) ---
            weight = daf_states[0].weight
            if weight > 0.0:
                k_gain = (p_pred_base * weight) / (
                    p_pred_base * weight + measurement_variance
                )
                x_new = x_pred_base + k_gain * residual
                p_new = (1.0 - k_gain) * p_pred_base
            else:
                x_new = x_pred_base
                p_new = p_pred_base

            # --- Convergence check on weights (state not yet applied) ---
            curr_weights = np.array([s.weight for s in daf_states])
            if np.max(np.abs(curr_weights - prev_weights)) < self.config.delta_weight:
                break
            prev_weights = curr_weights.copy()

        # Apply final converged state
        kalman_state.x = x_new
        kalman_state.P = p_new

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


# =============================================================================
# Sliding-Window Batch DAF
#
# The original DAF algorithm (genfit/DAF.cc) operates on multiple measurements
# jointly: each beta iteration runs a full Kalman fit over ALL measurements,
# then recomputes Bayesian weights across the entire measurement set. This
# multi-measurement joint fitting is the core advantage of DAF — measurements
# compete for weight via the phi_cut penalty term.
#
# For streaming industrial time series, we adapt this as a sliding-window
# batch: buffer N measurements, run full batch DAF on the window, output
# outlier labels, then slide forward. Between batch DAF passes, real-time
# updates use the standard Kalman filter.
# =============================================================================

from dataclasses import dataclass


@dataclass
class WindowDAFResult:
    """Result of one sliding-window batch DAF pass.

    Attributes:
        measurements: Input measurements for this window.
        weights: Final DAF weight per measurement (after annealing).
        outliers: Boolean mask: True = outlier (weight < threshold).
        converged: Whether the annealing converged within max_iterations.
        n_iterations: Number of beta iterations actually used.
    """
    measurements: list[float]
    weights: list[float]
    outliers: list[bool]
    converged: bool
    n_iterations: int


class SlidingWindowDAF:
    """Sliding-window batch DAF for streaming industrial time series.

    Buffers N measurements, then runs the full DAF annealing loop (multiple
    beta iterations, each with a full Kalman pass over the entire window) to
    jointly classify outliers. The window slides by step_size for overlap.

    Usage::

        daf = SlidingWindowDAF(DAFConfig(beta_start=100, beta_final=0.1, n_steps=5),
                               window_size=32, step_size=16)
        daf.set_initial_state(14.0, P0=0.01, Q=1e-5, R=0.01)

        for measurement in plc_stream:
            result = daf.feed(measurement)
            if result is not None:
                # Batch DAF completed for this window
                for i, (m, w, is_outlier) in enumerate(zip(
                    result.measurements, result.weights, result.outliers
                )):
                    print(f"measurement={m:.3f} weight={w:.3f} outlier={is_outlier}")

    Attributes:
        config: DAF annealing configuration.
        window_size: Number of measurements to buffer before running batch DAF.
        step_size: Slide distance (default: window_size // 2 for 50% overlap).
        outlier_threshold: Weight below which a measurement is classified as outlier.
    """

    def __init__(
        self,
        config: Optional[DAFConfig] = None,
        window_size: int = 32,
        step_size: Optional[int] = None,
        outlier_threshold: float = 0.1,
    ):
        self.config = config or DAFConfig()
        self.window_size = window_size
        self.step_size = step_size or (window_size // 2)
        self.outlier_threshold = outlier_threshold

        # State for the standard Kalman used between batch passes
        self._x0: float = 0.0
        self._P0: float = 1.0
        self._Q: float = 1e-5
        self._R: float = 1e-3
        self._initialized: bool = False

        # Measurement buffer
        self._buffer: list[float] = []
        self._variances: list[float] = []

        # Per-measurement DAF for real-time fallback (single-measurement only)
        self._single_daf = DAFKalmanFilter(self.config)

        # Running Kalman state for between-window real-time updates
        self._kf: Optional[KalmanState] = None

        # Pre-compute annealing schedule for batch DAF iterations
        self._betas: list[float] = compute_annealing_schedule(
            self.config.beta_start,
            self.config.beta_final,
            self.config.n_steps,
        )

    def set_initial_state(
        self,
        x0: float,
        P0: float = 0.01,
        Q: float = 1e-5,
        R: float = 1e-3,
    ) -> None:
        """Set the initial Kalman state for real-time updates between windows."""
        self._x0 = x0
        self._P0 = P0
        self._Q = Q
        self._R = R
        self._kf = KalmanState(x0=x0, P0=P0, Q=Q, R=R)
        self._initialized = True

    def feed(self, measurement: float, variance: Optional[float] = None) -> Optional[WindowDAFResult]:
        """Feed a single measurement into the sliding window.

        If the window is not yet full, returns None (buffering). When the
        window is full, runs batch DAF and returns the result. The window
        then slides by step_size.

        Args:
            measurement: New sensor reading.
            variance: Measurement variance. Uses stored R if None.

        Returns:
            WindowDAFResult if a batch DAF pass completed, None if buffering.
        """
        if not self._initialized:
            raise RuntimeError("Call set_initial_state() before feed().")

        var = variance if variance is not None else self._R

        # Real-time Kalman update between batch passes
        if self._kf is not None and self._buffer:
            self._kf, _ = self._single_daf.update(
                self._kf, measurement, measurement_variance=var, dt=1.0
            )

        self._buffer.append(measurement)
        self._variances.append(var)

        if len(self._buffer) >= self.window_size:
            return self._run_batch_daf()
        return None

    def _run_batch_daf(self) -> WindowDAFResult:
        """Run full batch DAF on the current window.

        Implements the original DAF algorithm (genfit/DAF.cc processTrackWithRep):
        1. For each beta in the annealing schedule:
           a. Run a full Kalman pass over all window measurements
           b. Compute Bayesian weights for all measurements
        2. Check convergence (max weight change < delta_weight)
        3. Classify outliers

        The key difference from per-measurement DAF: all measurements are
        fitted jointly within each beta iteration, allowing the phi_cut
        penalty to distribute weight across the entire set.
        """
        n = len(self._buffer)
        measurements = list(self._buffer)
        variances = list(self._variances)

        # Initialize weights uniformly
        weights = [1.0] * n
        prev_weights = [1.0] * n

        # Kalman state at window start
        kf = self._kf
        if kf is None:
            kf = KalmanState(x0=self._x0, P0=self._P0, Q=self._Q, R=self._R)

        converged = False
        n_iterations = 0

        for iteration in range(self.config.max_iterations):
            beta = (
                self._betas[iteration]
                if iteration < len(self._betas)
                else self._betas[-1]
            )
            n_iterations = iteration + 1

            # --- Full Kalman pass over all measurements with current weights ---
            x = kf.x
            P = kf.P

            chi2s = []
            residuals_list = []
            innov_covs = []

            for i in range(n):
                # Predict
                x_pred = x
                p_pred = P + kf.Q

                # Residual
                residual = measurements[i] - x_pred
                innov_cov = p_pred + variances[i]
                chi2 = (residual**2) / innov_cov if innov_cov > 1e-12 else 0.0

                chi2s.append(chi2)
                residuals_list.append(np.array([residual]))
                innov_covs.append(innov_cov)

                # Weighted Kalman update
                w = weights[i]
                if w > 0.0:
                    k_gain = (p_pred * w) / (p_pred * w + variances[i])
                    x = x_pred + k_gain * residual
                    P = (1.0 - k_gain) * p_pred
                else:
                    x = x_pred
                    P = p_pred

            # --- Compute Bayesian weights for all measurements ---
            new_weights = self._batch_calc_weights(
                chi2s, innov_covs, beta, n
            )

            # --- Convergence check ---
            max_change = max(abs(nw - ow) for nw, ow in zip(new_weights, prev_weights))
            if max_change < self.config.delta_weight:
                converged = True
                weights = new_weights
                break

            prev_weights = weights
            weights = new_weights

        # Classify outliers
        outliers = [w < self.outlier_threshold for w in weights]

        # Slide the window
        self._buffer = self._buffer[self.step_size:]
        self._variances = self._variances[self.step_size:]

        return WindowDAFResult(
            measurements=measurements,
            weights=weights,
            outliers=outliers,
            converged=converged,
            n_iterations=n_iterations,
        )

    def _batch_calc_weights(
        self,
        chi2s: list[float],
        innov_covs: list[float],
        beta: float,
        n_measurements: int,
    ) -> list[float]:
        """Compute Bayesian weights for all measurements at current beta.

        Adapted from genfit/DAF.cc calcWeights() — the multi-measurement
        version where all measurements compete for weight via the shared
        phi_sum + phi_cut denominator.
        """
        cfg = self.config
        d = 1  # scalar measurements
        chi2_cut = -2.0 * math.log(max(cfg.prob_cut, 1e-12))

        phis = []
        phi_total = 0.0
        phi_cut_total = 0.0

        for i in range(n_measurements):
            norm = (2.0 * math.pi) ** (-d / 2.0) * (innov_covs[i] ** (-0.5))
            chi2_clamped = min(chi2s[i] / (2.0 * beta), 50.0)
            phi = norm * math.exp(-chi2_clamped)
            phis.append(phi)
            phi_total += phi

            # phi_cut contribution per measurement
            cut_val = chi2_cut / (2.0 * beta)
            phi_cut = norm * math.exp(-min(cut_val, 50.0))
            phi_cut_total += phi_cut

        denom = phi_total + phi_cut_total
        if denom < 1e-12:
            return [0.0] * n_measurements

        return [phi / denom for phi in phis]

    def flush(self) -> Optional[WindowDAFResult]:
        """Run batch DAF on any remaining buffered measurements."""
        if len(self._buffer) == 0:
            return None
        return self._run_batch_daf()

    @property
    def buffer_fill(self) -> float:
        """Ratio of buffer filled (0.0 to 1.0)."""
        return len(self._buffer) / self.window_size if self.window_size > 0 else 0.0
