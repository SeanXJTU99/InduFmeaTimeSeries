"""Tests for wavelet denoiser."""

import numpy as np

from src.signal.wavelet_denoise import WaveletDenoiser, WaveletParams, wavelet_denoise


class TestWaveletDenoiser:
    def test_reduces_noise_std(self) -> None:
        rng = np.random.default_rng(42)
        t = np.linspace(0, 10, 256)
        clean = np.sin(t)
        noisy = clean + rng.normal(0, 0.2, 256)

        wd = WaveletDenoiser(WaveletParams(level=3, wavelet="db4"))
        denoised = wd.denoise(noisy)

        noise_std_before = float(np.std(noisy - clean))
        noise_std_after = float(np.std(denoised - clean))
        assert noise_std_after < noise_std_before

    def test_preserves_signal_length(self) -> None:
        data = np.random.default_rng(0).normal(0, 1, 128)
        result = wavelet_denoise(data, level=3)
        assert len(result) == len(data)

    def test_rejects_short_signal_for_deep_level(self) -> None:
        wd = WaveletDenoiser(WaveletParams(level=10, wavelet="db4"))
        with pytest.raises(ValueError, match="Signal length"):
            wd.denoise(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_bayes_threshold_produces_valid_output(self) -> None:
        wd = WaveletDenoiser(WaveletParams(
            level=2, wavelet="db4", threshold_method="bayes"
        ))
        data = np.sin(np.linspace(0, 10, 128)) + np.random.default_rng(1).normal(0, 0.1, 128)
        result = wd.denoise(data)
        assert len(result) == len(data)
        assert not np.any(np.isnan(result))
