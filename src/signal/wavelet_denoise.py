"""Wavelet-based denoising for slow-varying, high-inertia industrial signals.

Targets cryogenic distillation temperature signals (typically -200 °C to
ambient) where true process changes are slow (minutes to hours) but sensor
chains introduce broad-spectrum noise.

Uses db4 (Daubechies-4) wavelets — a good default for piecewise-smooth
process signals.  Soft thresholding via the universal threshold
(Donoho-Johnstone) on detail coefficients.
"""

from __future__ import annotations

import numpy as np
import pywt
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class WaveletParams:
    """Configuration for wavelet denoising."""

    wavelet: str = "db4"  # wavelet family
    level: int = 3  # decomposition levels
    mode: str = "per"  # boundary extension mode
    threshold_mode: str = "soft"  # 'soft' or 'hard'
    threshold_method: str = "universal"  # 'universal' or 'sure' or 'bayes'
    sigma_method: str = "median"  # noise std estimation method


class WaveletDenoiser:
    """Configurable wavelet denoiser for batch processing.

    Usage::

        wd = WaveletDenoiser(WaveletParams(level=3, wavelet="db4"))
        clean = wd.denoise(raw_signal)

    For streaming / sliding-window use, call ``denoise`` on the most recent
    window of *window_size* samples.
    """

    def __init__(self, params: WaveletParams | None = None) -> None:
        self._p = params or WaveletParams()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def denoise(self, data: np.ndarray) -> np.ndarray:
        """Decompose, threshold detail coefficients, and reconstruct.

        Args:
            data: 1-D array of raw signal values.

        Returns:
            1-D array of denoised values (same length).
        """
        if len(data) < 2:
            return data.copy()

        # Decompose
        coeffs = pywt.wavedec(
            data, wavelet=self._p.wavelet, level=self._p.level, mode=self._p.mode
        )

        # Estimate noise standard deviation from finest detail coefficients
        sigma = self._estimate_sigma(coeffs[-1])

        # Threshold detail coefficients (keep approximation untouched)
        threshold = self._compute_threshold(sigma, len(data))
        new_coeffs = [coeffs[0]]  # approximation
        for detail in coeffs[1:]:
            new_coeffs.append(self._apply_threshold(detail, threshold))

        # Reconstruct
        reconstructed = pywt.waverec(
            new_coeffs, wavelet=self._p.wavelet, mode=self._p.mode
        )
        # waverec may be slightly longer due to convolution; trim
        return reconstructed[: len(data)]

    def decompose(self, data: np.ndarray) -> list[np.ndarray]:
        """Return raw decomposition coefficients (for feature extraction)."""
        return pywt.wavedec(
            data, wavelet=self._p.wavelet, level=self._p.level, mode=self._p.mode
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_sigma(self, detail: np.ndarray) -> float:
        """Estimate noise std from finest detail coefficients."""
        if self._p.sigma_method == "median":
            return float(
                (1.0 / 0.6745) * np.median(np.abs(detail - np.median(detail)))
            )
        # fallback: standard deviation
        return float(np.std(detail))

    def _compute_threshold(self, sigma: float, n_samples: int) -> float:
        """Compute threshold value."""
        if self._p.threshold_method == "universal":
            return float(sigma * np.sqrt(2.0 * np.log(n_samples)))
        if self._p.threshold_method == "sure":
            return float(sigma * np.sqrt(np.log(n_samples)))
        # Bayes shrink
        return float(sigma**2 / max(np.sqrt(max(sigma**2 - sigma**2, 0)), 1e-10))

    def _apply_threshold(self, coeffs: np.ndarray, threshold: float) -> np.ndarray:
        """Apply soft or hard thresholding."""
        return pywt.threshold(coeffs, value=threshold, mode=self._p.threshold_mode)


# ------------------------------------------------------------------
# Convenience function (backward-compatible with prototype in source doc)
# ------------------------------------------------------------------


def wavelet_denoise(
    data: np.ndarray,
    wavelet: str = "db4",
    level: int = 2,
    mode: str = "per",
) -> np.ndarray:
    """One-call wavelet denoising with universal soft thresholding.

    Args:
        data: 1-D raw signal.
        wavelet: wavelet name (``'db4'``, ``'sym5'``, etc.).
        level: decomposition depth.
        mode: boundary extension mode.

    Returns:
        Denoised signal.
    """
    params = WaveletParams(
        wavelet=wavelet, level=level, mode=mode, threshold_mode="soft"
    )
    denoiser = WaveletDenoiser(params)
    return denoiser.denoise(data)
