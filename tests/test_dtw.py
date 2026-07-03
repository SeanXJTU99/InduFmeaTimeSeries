"""Tests for DTW aligner."""

import numpy as np
import pytest

from src.signal.dtw_aligner import DTWAligner, DTWConfig, DTWResult, dtw_align


class TestDTWAligner:
    def test_identical_series_zero_lag(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = a.copy()
        result = dtw_align(a, b)
        assert result.lag_mean < 1e-6

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            dtw_align(np.array([]), np.array([1.0, 2.0]))

    def test_slightly_shifted_series(self) -> None:
        a = np.sin(np.linspace(0, 4 * np.pi, 200))
        b = np.roll(a, 5)  # shift by 5 samples
        result = dtw_align(a, b)
        assert 3.0 <= result.lag_mean <= 7.0

    def test_output_shapes(self) -> None:
        a = np.random.default_rng(0).normal(0, 1, 50)
        b = np.random.default_rng(1).normal(0, 1, 60)
        result = dtw_align(a, b)
        assert result.cost_matrix.shape == (50, 60)
        assert result.path.shape[1] == 2
        assert result.distance >= 0
