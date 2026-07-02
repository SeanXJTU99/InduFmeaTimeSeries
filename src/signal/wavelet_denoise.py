"""Wavelet-based denoising for slow-varying, high-inertia industrial signals.

Targets cryogenic distillation temperature signals (typically -200 °C to
ambient) where true process changes are slow (minutes to hours) but sensor
chains introduce broad-spectrum noise.

Uses db4 (Daubechies-4) wavelets — a good default for piecewise-smooth
process signals.  Soft thresholding via the universal threshold
(Donoho-Johnstone) on detail coefficients.
"""

from __future__ import annotations

import math
import numpy as np
import pywt
from dataclasses import dataclass


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

        Raises:
            ValueError: if signal length is insufficient for the configured
                decomposition level.
        """
        # Guard: insufficient data for requested decomposition level
        max_lvl = pywt.dwt_max_level(len(data), self._p.wavelet)
        safe_level = min(self._p.level, max_lvl)
        if safe_level != self._p.level:
            raise ValueError(
                f"Signal length {len(data)} too short for level={self._p.level} "
                f"with wavelet '{self._p.wavelet}'. Maximum is {max_lvl}."
            )
        if len(data) < 2:
            return data.copy()

        # Decompose
        coeffs = pywt.wavedec(
            data, wavelet=self._p.wavelet, level=self._p.level, mode=self._p.mode
        )

        # Estimate noise standard deviation from finest detail coefficients
        sigma = self._estimate_sigma(coeffs[-1])

        # Per-subband thresholding (each detail level gets its own threshold)
        new_coeffs: list[np.ndarray] = [coeffs[0]]  # approximation — untouched
        for detail in coeffs[1:]:
            threshold = self._compute_threshold(sigma, len(data), detail)
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

    def _compute_threshold(
        self, sigma: float, n_samples: int, detail: np.ndarray
    ) -> float:
        """Compute threshold value for a given detail subband.

        Args:
            sigma: estimated noise standard deviation.
            n_samples: total signal length (used by universal / SURE).
            detail: the detail coefficient array for this subband
                (used by Bayes shrink to estimate signal variance).

        Returns:
            Threshold value.
        """
        if self._p.threshold_method == "universal":
            return float(sigma * np.sqrt(2.0 * np.log(n_samples)))
        if self._p.threshold_method == "sure":
            return float(sigma * np.sqrt(np.log(n_samples)))
        # Bayes shrink: T = sigma² / sigma_x
        # where sigma_x = sqrt(max(var(observed) - sigma², 0))
        var_y = float(np.var(detail))
        sigma_x_sq = max(var_y - sigma**2, 0.0)
        if sigma_x_sq < 1e-15:
            # negligible signal energy — fall back to universal threshold
            return float(sigma * np.sqrt(2.0 * np.log(n_samples)))
        return float(sigma**2 / math.sqrt(sigma_x_sq))

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
