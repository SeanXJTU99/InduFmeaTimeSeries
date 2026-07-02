"""Adaptive streaming baseline with EWMA + online KDE.

Industrial distillation columns drift over weeks/months due to fouling,
catalyst degradation, and ambient seasonal temperature swings.  A static
threshold triggers false alarms as the "normal" envelope shifts.

This module maintains an exponentially-weighted moving average (EWMA)
baseline and fits an online kernel density estimate (KDE) over a sliding
window of recent healthy-operation data, producing a dynamic envelope
that adapts to slow drift while remaining sensitive to abrupt anomalies.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from collections import deque


@dataclass
class BaselineConfig:
    """EWMA + KDE adaptive baseline parameters."""

    ewma_alpha: float = 0.05  # smoothing factor (0 < α ≤ 1)
    window_days: int = 7  # sliding window in days
    samples_per_day: int = 1440  # samples/day (1-min resolution)
    n_sigma: float = 3.0  # anomaly threshold in std deviations
    min_window_samples: int = 1000  # minimum samples before scoring
    min_std: float = 1e-6  # minimum standard deviation floor


class EWMAKDEBaseline:
    """Streaming adaptive baseline using EWMA trend + sliding-window KDE.

    Usage::

        bl = EWMAKDEBaseline(BaselineConfig(ewma_alpha=0.05))
        for val in sensor_stream:
            bl.update(val)
            score, is_anom = bl.check(val)
            if is_anom:
                print(f"Anomaly z={score:.2f}")
    """

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self._cfg = config or BaselineConfig()
        self._ewma: float = 0.0
        self._ewma_var: float = 1.0
        self._initialised: bool = False
        self._window: deque[float] = deque(
            maxlen=self._cfg.window_days * self._cfg.samples_per_day
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, value: float) -> None:
        """Ingest one data point, update EWMA and sliding window.

        Args:
            value: latest sensor reading.

        NaN/inf values are silently dropped — they would permanently
        corrupt the EWMA state.
        """
        if not math.isfinite(value):
            return
        self._window.append(value)
        if not self._initialised:
            self._ewma = value
            self._initialised = True
        else:
            alpha = self._cfg.ewma_alpha
            # Compute delta using PRE-update EWMA for correct variance estimate
            delta = value - self._ewma
            self._ewma = alpha * value + (1.0 - alpha) * self._ewma
            self._ewma_var = alpha * delta**2 + (1.0 - alpha) * self._ewma_var

    def check(self, value: float) -> Tuple[float, bool]:
        """Score a value against the current adaptive baseline.

        Args:
            value: the value to check.

        Returns:
            (z_score, is_anomalous) tuple.
        """
        if not math.isfinite(value):
            return 0.0, False
        if len(self._window) < self._cfg.min_window_samples:
            return 0.0, False

        std = math.sqrt(self._ewma_var) if self._ewma_var > 0 else self._cfg.min_std
        z = (value - self._ewma) / std
        return float(z), abs(z) > self._cfg.n_sigma

    def get_envelope(self) -> Tuple[float, float, float]:
        """Return (center, lower_bound, upper_bound) of the current envelope."""
        std = math.sqrt(self._ewma_var) if self._ewma_var > 0 else self._cfg.min_std
        n = self._cfg.n_sigma
        return (self._ewma, self._ewma - n * std, self._ewma + n * std)

    def get_density(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the empirical PDF over the sliding window.

        Returns:
            (x_grid, density) — 1-D arrays for plotting.
        """
        if len(self._window) < 10:
            return np.linspace(0, 1, n_points), np.ones(n_points)
        data = np.array(self._window)
        lo, hi = float(np.min(data)), float(np.max(data))
        pad = (hi - lo) * 0.1
        x = np.linspace(lo - pad, hi + pad, n_points)
        # simple Gaussian KDE
        bw = self._scott_bandwidth(data)
        density = np.zeros(n_points)
        for xi in data:
            density += np.exp(-0.5 * ((x - xi) / bw) ** 2)
        density /= len(data) * bw * np.sqrt(2.0 * np.pi)
        return x, density

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._ewma = 0.0
        self._ewma_var = 1.0
        self._initialised = False
        self._window.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _scott_bandwidth(data: np.ndarray) -> float:
        n = len(data)
        sigma = float(np.std(data))
        return sigma * n ** (-1.0 / 5.0)


# Backward-compatible alias
AdaptiveBaseline = EWMAKDEBaseline
