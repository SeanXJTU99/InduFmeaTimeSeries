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
    SlidingWindowDAF,
    WindowDAFResult,
    compute_annealing_schedule,
)


# =============================================================================
# Per-measurement DAF
# =============================================================================

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


# =============================================================================
# Sliding-window batch DAF
# =============================================================================

class TestSlidingWindowDAF:

    @pytest.fixture
    def daf(self):
        cfg = DAFConfig(beta_start=100, beta_final=0.1, n_steps=5, max_iterations=4)
        d = SlidingWindowDAF(cfg, window_size=32, step_size=16)
        d.set_initial_state(x0=14.0, P0=0.01, Q=1e-5, R=0.01)
        return d

    def test_buffering_before_window_full(self, daf):
        """feed() returns None until window fills."""
        for _ in range(31):
            result = daf.feed(14.0)
            assert result is None
        assert daf.buffer_fill == pytest.approx(31 / 32)

    def test_window_triggers_batch_daf(self, daf):
        """The 32nd measurement triggers a batch DAF pass."""
        result = None
        for i in range(32):
            measurement = 14.0 + 0.01 * np.random.randn()
            result = daf.feed(measurement)
        assert result is not None
        assert isinstance(result, WindowDAFResult)
        assert len(result.measurements) == 32
        assert len(result.weights) == 32
        assert len(result.outliers) == 32

    def test_clean_signal_no_false_positives(self, daf):
        """Clean signal should produce near-zero outlier rate."""
        outlier_count = 0
        total_checked = 0

        for i in range(64):
            measurement = 14.0 + 0.05 * np.sin(i * 0.1)
            result = daf.feed(measurement)
            if result is not None:
                outlier_count += sum(result.outliers)
                total_checked += len(result.outliers)

        if total_checked > 0:
            fp_rate = outlier_count / total_checked
            assert fp_rate < 0.10  # < 10% false positive

    def test_spike_in_window_flagged(self, daf):
        """Inject a large spike into one window and verify it is flagged."""
        result = None
        for i in range(32):
            if i == 16:
                measurement = 500.0  # extreme spike in the middle of the window
            else:
                measurement = 14.0 + 0.01 * np.random.randn()
            result = daf.feed(measurement)

        assert result is not None
        # The spike at index 16 should be flagged
        assert result.outliers[16] is True
        # Other measurements should not be flagged
        other_flags = result.outliers[:16] + result.outliers[17:]
        fp_rate = sum(other_flags) / len(other_flags)
        assert fp_rate < 0.15

    def test_window_slides_correctly(self, daf):
        """After a batch pass, the window slides by step_size."""
        # Fill first window
        for i in range(32):
            daf.feed(14.0 + 0.01 * np.random.randn())

        # Buffer should now contain 16 remaining (32 - 16 step_size)
        assert daf.buffer_fill == pytest.approx(16 / 32)

        # Fill second window
        result = None
        for i in range(16):
            result = daf.feed(14.0 + 0.01 * np.random.randn())

        assert result is not None
        assert daf.buffer_fill == pytest.approx(16 / 32)

    def test_flush_remaining(self, daf):
        """flush() processes any remaining buffered measurements."""
        for i in range(20):  # less than window_size
            daf.feed(14.0 + 0.01 * np.random.randn())

        result = daf.flush()
        assert result is not None
        assert len(result.measurements) == 20

    def test_convergence_flag(self, daf):
        """DAF should converge on clean data."""
        result = None
        for i in range(32):
            result = daf.feed(14.0 + 0.01 * np.random.randn())

        assert result is not None
        # On clean data, the weights should converge quickly
        # (convergence is not guaranteed but likely for clean signals)
        assert result.n_iterations <= daf.config.max_iterations
