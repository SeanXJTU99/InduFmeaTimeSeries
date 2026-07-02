"""Dynamic Time Warping (DTW) alignment for asynchronous multi-source data.

In the cryogenic distillation context, PLC streams are 100 ms real-time
while isotope abundance reports arrive as async Excel files every 30–60
minutes.  DTW non-linearly stretches the historical PLC timeline to align
with the sparse abundance measurements, recovering the column's thermal
inertia lag constant τ.

Implementation uses a standard accumulated-cost-matrix DP with Sakoe-Chiba
band constraint for efficiency on long industrial sequences.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DTWConfig:
    """DTW alignment parameters."""

    window: Optional[int] = None  # Sakoe-Chiba band half-width (None = full matrix)
    metric: str = "euclidean"  # distance metric: 'euclidean' or 'manhattan'


class DTWAligner:
    """Align two 1-D time series with DTW and extract lag statistics.

    Usage::

        aligner = DTWAligner(DTWConfig(window=50))
        result = aligner.align(plc_pressure_1h, excel_abundance_interp)
        print(f"Estimated thermal lag τ ≈ {result.lag_mean:.1f} samples")
    """

    def __init__(self, config: DTWConfig | None = None) -> None:
        self._cfg = config or DTWConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(
        self, series_a: np.ndarray, series_b: np.ndarray
    ) -> "DTWResult":
        """Compute the optimal DTW alignment path.

        Args:
            series_a: reference series (e.g. PLC pressure, N samples).
            series_b: query series (e.g. interpolated abundance, M samples).

        Returns:
            :class:`DTWResult` with cost matrix, path, and lag statistics.
        """
        n, m = len(series_a), len(series_b)
        cost = self._accumulated_cost(series_a, series_b)
        path = self._backtrack(cost)
        lag = self._compute_lag(path)
        return DTWResult(
            cost_matrix=cost,
            path=path,
            distance=float(cost[-1, -1]),
            lag_mean=float(np.mean(np.abs(lag))) if len(lag) > 0 else 0.0,
            lag_std=float(np.std(lag)) if len(lag) > 0 else 0.0,
            lag_series=lag,
        )

    def estimate_lag_constant(self, series_a: np.ndarray, series_b: np.ndarray) -> float:
        """Return the mean absolute lag (in sample units) between two series.

        This is a one-shot convenience for computing the thermal-inertia
        time constant τ.
        """
        result = self.align(series_a, series_b)
        return result.lag_mean

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _distance(self, a: float, b: float) -> float:
        if self._cfg.metric == "manhattan":
            return abs(a - b)
        return float(a - b) ** 2  # squared Euclidean

    def _accumulated_cost(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n, m = len(a), len(b)
        cost = np.full((n, m), np.inf, dtype=np.float64)
        cost[0, 0] = self._distance(a[0], b[0])

        w = self._cfg.window or max(n, m)

        for i in range(1, n):
            j_start = max(1, i - w)
            j_end = min(m, i + w + 1)
            for j in range(j_start, j_end):
                prev = min(
                    cost[i - 1, j],  # insertion
                    cost[i, j - 1],  # deletion
                    cost[i - 1, j - 1],  # match
                )
                cost[i, j] = self._distance(a[i], b[j]) + prev
        return cost

    def _backtrack(self, cost: np.ndarray) -> np.ndarray:
        n, m = cost.shape
        i, j = n - 1, m - 1
        path = [(i, j)]
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                prev = np.argmin([cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]])
                if prev == 0:
                    i -= 1; j -= 1
                elif prev == 1:
                    i -= 1
                else:
                    j -= 1
            path.append((i, j))
        return np.array(path[::-1])

    @staticmethod
    def _compute_lag(path: np.ndarray) -> np.ndarray:
        return path[:, 0] - path[:, 1]  # i - j


@dataclass
class DTWResult:
    """Result of a DTW alignment."""

    cost_matrix: np.ndarray
    path: np.ndarray  # shape (K, 2) — optimal warp path indices
    distance: float  # total accumulated distance
    lag_mean: float  # mean lag (samples)
    lag_std: float  # std of lag (samples)
    lag_series: np.ndarray  # per-point lag values


def dtw_align(
    a: np.ndarray,
    b: np.ndarray,
    window: int | None = None,
) -> DTWResult:
    """Convenience function — align two 1-D series with DTW.

    Args:
        a: reference series.
        b: query series.
        window: Sakoe-Chiba band half-width.

    Returns:
        :class:`DTWResult`.
    """
    return DTWAligner(DTWConfig(window=window)).align(a, b)
