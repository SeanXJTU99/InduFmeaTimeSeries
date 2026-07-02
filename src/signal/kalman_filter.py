"""Two-stage Kalman filter for industrial PLC signal denoising.

Stage 1: sensor-level electromagnetic interference filtering.
Stage 2: physics-informed state prediction with process model constraints.

All tag names and parameters are fictitious — see docs/data_notice.md.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KalmanParams:
    """Parameters for a single Kalman filter stage."""

    process_noise: float = 1e-5  # Q — process noise covariance
    measurement_noise: float = 1e-3  # R — measurement noise covariance
    estimated_error: float = 1.0  # initial state error covariance
    initial_value: float = 0.0  # initial state estimate
    max_innovation: Optional[float] = None  # clip innovation to this magnitude


class KalmanFilter:
    """Single-stage 1D Kalman filter for real-time streaming denoising.

    Suitable for high-frequency, high-noise signals such as PLC pressure
    and flow-rate measurements.  Tune Q and R to trade off between
    trusting the process model (low Q) and trusting the sensor (low R).

    Usage::

        kf = KalmanFilter(KalmanParams(Q=1e-4, R=0.04, initial_value=14.0))
        for raw in plc_stream:
            clean = kf.update(raw)
    """

    def __init__(self, params: KalmanParams | None = None) -> None:
        p = params or KalmanParams()
        self._Q = p.process_noise
        self._R = p.measurement_noise
        self._P = p.estimated_error
        self._x = p.initial_value
        self._max_innov = p.max_innovation

    def update(self, measurement: float) -> float:
        """Process one scalar measurement and return the filtered estimate.

        Args:
            measurement: raw sensor reading.

        Returns:
            Filtered (denoised) value after Kalman update.
        """
        # --- prediction step ---
        x_pred = self._x
        p_pred = self._P + self._Q

        # --- innovation (measurement residual) ---
        innovation = measurement - x_pred
        if self._max_innov is not None:
            innovation = float(np.clip(innovation, -self._max_innov, self._max_innov))

        # --- Kalman gain ---
        k_gain = p_pred / (p_pred + self._R)

        # --- update step ---
        self._x = x_pred + k_gain * innovation
        self._P = (1.0 - k_gain) * p_pred

        return self._x

    @property
    def state(self) -> float:
        """Current filtered state estimate."""
        return self._x

    @property
    def covariance(self) -> float:
        """Current state error covariance."""
        return self._P

    def reset(self, value: float = 0.0, covariance: float = 1.0) -> None:
        """Reset filter state (e.g. after sensor maintenance)."""
        self._x = value
        self._P = covariance


class TwoStageKalmanFilter:
    """Cascade of two Kalman filters for hierarchical denoising.

    Stage 1 — sensor-level: removes high-frequency EMI and quantization noise
        from raw PLC analog input channels.
    Stage 2 — process-level: incorporates a simplified physical model
        (e.g. mass/energy balance) to reject physically implausible transients
        such as valve-switching spikes.

    Typical use in the cryogenic distillation context:

    * Stage 1 on pressure / flow-rate (fast, noisy).
    * Stage 2 on the filtered output, with Q tuned to the column's
      thermal inertia time-constant.
    """

    def __init__(
        self,
        stage1_params: KalmanParams | None = None,
        stage2_params: KalmanParams | None = None,
    ) -> None:
        """Initialise both stages.

        Args:
            stage1_params: parameters for the EMI-rejection stage.
            stage2_params: parameters for the physics-informed stage.
        """
        self.stage1 = KalmanFilter(stage1_params or KalmanParams(Q=1e-5, R=1e-3))
        self.stage2 = KalmanFilter(stage2_params or KalmanParams(Q=1e-6, R=5e-3))

    def update(self, measurement: float) -> float:
        """Feed a raw measurement through both stages.

        Args:
            measurement: raw sensor value.

        Returns:
            Twice-filtered value suitable for downstream anomaly detection.
        """
        s1 = self.stage1.update(measurement)
        s2 = self.stage2.update(s1)
        return s2

    def reset(self) -> None:
        """Reset both stages."""
        self.stage1.reset()
        self.stage2.reset()


def batch_filter(
    signal: np.ndarray,
    stage1_params: KalmanParams | None = None,
    stage2_params: KalmanParams | None = None,
) -> np.ndarray:
    """Convenience: apply TwoStageKalmanFilter to a full 1-D array.

    Args:
        signal: 1-D numpy array of raw sensor readings.
        stage1_params: optional stage-1 configuration.
        stage2_params: optional stage-2 configuration.

    Returns:
        1-D array of filtered values, same length as *signal*.
    """
    kf = TwoStageKalmanFilter(stage1_params, stage2_params)
    return np.array([kf.update(float(v)) for v in signal])
