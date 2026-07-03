"""Tests for two-stage Kalman filter."""

import numpy as np
import pytest

from src.signal.kalman_filter import (
    KalmanFilter, KalmanParams, TwoStageKalmanFilter, batch_filter,
)


class TestKalmanFilter:
    def test_converges_to_constant(self) -> None:
        kf = KalmanFilter(KalmanParams(Q=1e-4, R=0.01, initial_value=10.0))
        for _ in range(100):
            kf.update(14.0)
        assert abs(kf.state - 14.0) < 0.1

    def test_rejects_negative_params(self) -> None:
        with pytest.raises(ValueError):
            KalmanParams(process_noise=-0.1)
        with pytest.raises(ValueError):
            KalmanParams(measurement_noise=0.0)
        with pytest.raises(ValueError):
            KalmanParams(estimated_error=-1.0)

    def test_reset(self) -> None:
        kf = KalmanFilter(KalmanParams(initial_value=50.0))
        kf.update(100.0)
        kf.reset(value=0.0, covariance=0.5)
        assert kf.state == 0.0
        assert kf.covariance == 0.5


class TestTwoStageKalmanFilter:
    def test_smoothes_noise(self) -> None:
        t = np.linspace(0, 10, 500)
        clean = np.sin(t) * 2.0 + 14.0
        noisy = clean + np.random.default_rng(42).normal(0, 0.3, 500)
        kf = TwoStageKalmanFilter()
        filtered = np.array([kf.update(float(v)) for v in noisy])
        assert filtered.std() < noisy.std()


def test_batch_filter(self) -> None:
    signal = np.array([1.0, 1.1, 1.0, 0.9, 1.0])
    result = batch_filter(signal)
    assert len(result) == len(signal)
    np.testing.assert_array_less(np.abs(np.diff(result)), 0.5)
