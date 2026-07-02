"""Scalogram (time-frequency representation) generation via CWT.

Produces multi-scale time-frequency feature maps that serve as rich inputs
to Transformer encoders in the later Kalman-Wavelet-Transformer cascade
(see ``src/models/kalman_wavelet_transformer.py``).

The scalogram captures how a signal's frequency content evolves over time,
which is critical for detecting incipient faults (e.g. flooding precursors
in a cryogenic distillation column) that manifest as subtle shifts in the
power distribution across scales.
"""

from __future__ import annotations

import numpy as np
import pywt
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ScalogramConfig:
    """Scalogram computation parameters."""

    wavelet: str = "cmor1.5-1.0"  # complex Morlet — good time-freq trade-off
    scales: Optional[np.ndarray] = None  # if None, auto-computed
    n_scales: int = 64  # number of scales (ignored if scales is provided)
    sampling_period: float = 1.0  # dt between consecutive samples


class ScalogramExtractor:
    """Compute CWT-based scalograms for 1-D industrial time series.

    Usage::

        extractor = ScalogramExtractor(ScalogramConfig(n_scales=64))
        scalogram, frequencies = extractor.compute(plc_pressure_signal)
        # scalogram shape: (n_scales, n_samples) — |coefficients|
    """

    def __init__(self, config: ScalogramConfig | None = None) -> None:
        self._cfg = config or ScalogramConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the scalogram (magnitude of CWT coefficients).

        Args:
            signal: 1-D array of real-valued samples.

        Returns:
            (scalogram, frequencies) tuple.
            *scalogram* — 2-D array, shape ``(n_scales, n_samples)``.
            *frequencies* — 1-D array of the pseudo-frequency (Hz) for
              each scale, assuming *sampling_period*.
        """
        scales = self._get_scales()
        coefs, freqs = pywt.cwt(
            signal, scales, self._cfg.wavelet, self._cfg.sampling_period
        )
        scalogram = np.abs(coefs)  # magnitude
        return scalogram, freqs

    def compute_log(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Same as :meth:`compute` but returns log-magnitude for visualisation."""
        scalogram, freqs = self.compute(signal)
        return np.log1p(scalogram), freqs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_scales(self) -> np.ndarray:
        if self._cfg.scales is not None:
            return self._cfg.scales
        # auto-range following PyWavelets conventions
        return np.arange(1, self._cfg.n_scales + 1, dtype=np.float64)


def compute_scalogram(
    signal: np.ndarray,
    n_scales: int = 64,
    sampling_period: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper — compute scalogram in one call.

    Args:
        signal: 1-D input.
        n_scales: number of wavelet scales.
        sampling_period: dt between samples.

    Returns:
        (scalogram, frequencies) tuple.
    """
    cfg = ScalogramConfig(n_scales=n_scales, sampling_period=sampling_period)
    return ScalogramExtractor(cfg).compute(signal)
