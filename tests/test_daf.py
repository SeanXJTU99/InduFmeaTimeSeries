"""Tests for DAF (Deterministic Annealing Filter) Kalman."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.signal.daf_kalman import (
    DAFConfig,
    DAFKalmanFilter,
    KalmanState,
    compute_annealing_schedule,
)


class TestDAFKalmanFilter:

    def test_annealing_schedule_logarithmic(self):
        betas = compute_annealing_schedule(beta_start=100.0, beta_final=0.1, n_steps=5)
        assert len(betas) == 5
        assert betas[0] == pytest.approx(100.0, rel=1e-6)
        assert betas[-1] == pytest.approx(0.1, rel=1e-6)
        for i in range(len(betas) - 1):
            assert betas[i] > betas[i + 1]

    def test_single_step_schedule(self):
        betas = compute_annealing_schedule(beta_start=100.0, beta_final=0.1, n_steps=1)
        assert betas == [0.1]

    def test_clean_signal_not_flagged(self):
        daf = DAFKalmanFilter(DAFConfig(beta_start=100, beta_final=0.5, n_steps=3))
        kf = KalmanState(x0=14.0, P0=0.01, Q=1e-5, R=0.01)

        np.random.seed(42)
        outlier_count = 0
        for i in range(200):
            measurement = 14.0 + 0.1 * np.sin(i * 0.1)
            kf, is_outlier = daf.update(
                kf, measurement, measurement_variance=0.01, dt=1.0
            )
            if is_outlier:
                outlier_count += 1

        assert outlier_count <= 10  # < 5% false positive rate

    def test_large_spike_suppressed(self):
        daf = DAFKalmanFilter(DAFConfig(beta_start=100, beta_final=0.1, n_steps=5))
        kf = KalmanState(x0=14.0, P0=0.01, Q=1e-5, R=0.01)

        outlier_flags = []
        for i in range(100):
            if i == 50:
                measurement = 50.0  # ~3600-sigma spike
            else:
                measurement = 14.0 + 0.01 * np.random.randn()
            kf, is_outlier = daf.update(
                kf, measurement, measurement_variance=0.01, dt=1.0
            )
            outlier_flags.append(is_outlier)

        assert outlier_flags[50] is True
        assert sum(outlier_flags[45:50]) == 0
        assert sum(outlier_flags[51:56]) == 0

    def test_gradual_drift_not_flagged(self):
        daf = DAFKalmanFilter(DAFConfig(beta_start=100, beta_final=0.5, n_steps=3))
        kf = KalmanState(x0=14.0, P0=0.01, Q=1e-5, R=0.01)

        outlier_count = 0
        for i in range(200):
            drift = 0.05 * i
            measurement = 14.0 + drift + 0.01 * np.random.randn()
            kf, is_outlier = daf.update(
                kf, measurement, measurement_variance=0.01, dt=1.0
            )
            if is_outlier:
                outlier_count += 1

        assert outlier_count <= 5

    def test_annealing_schedule_exact(self):
        config = DAFConfig(beta_start=100.0, beta_final=0.1, n_steps=5)
        daf = DAFKalmanFilter(config)
        betas = daf.annealing_schedule

        expected = [100.0, 17.78279, 3.16228, 0.56234, 0.1]
        for b, e in zip(betas, expected):
            assert b == pytest.approx(e, rel=1e-4)


class TestKalmanState:

    def test_predict_increases_uncertainty(self):
        kf = KalmanState(x0=0.0, P0=1.0, Q=0.1, R=0.01)
        _, p_pred = kf.predict(dt=1.0)
        assert p_pred > 1.0

    def test_predict_dt_scaling(self):
        kf = KalmanState(x0=0.0, P0=1.0, Q=0.1, R=0.01)
        _, p1 = kf.predict(dt=1.0)
        _, p10 = kf.predict(dt=10.0)
        assert p10 > p1
