"""Virtual soft sensor for isotope abundance prediction.

Since isotope measurement instruments output abundance data as asynchronous
Excel reports (30–60 min lag), this module provides a real-time *virtual
observer* that estimates current abundance from live PLC streams (temperature,
pressure, flow rates) using a lightweight ML model (XGBoost or LSTM).

The virtual sensor is trained on DTW-aligned historical data and runs on
the edge IPC, bridging the time gap between physical fault onset and the
next Excel report.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class SoftSensorConfig:
    """Configuration for the virtual soft sensor."""

    model_type: str = "xgboost"  # 'xgboost' or 'lstm'
    input_features: Sequence[str] = field(default_factory=lambda: [
        "temperature", "pressure", "flow_rate", "valve_opening", "reflux_ratio"
    ])
    lookback_window: int = 60  # past samples to feed the model
    predict_horizon: int = 1  # steps ahead to predict


class VirtualSoftSensor:
    """Lightweight virtual observer for real-time isotope abundance estimation.

    This is a *placeholder interface* that can be backed by either a
    pre-trained XGBoost model or a small LSTM, depending on edge IPC
    resources.  The interface is designed so the downstream Agent and
    anomaly-detection pipeline are model-agnostic.

    Usage::

        sensor = VirtualSoftSensor(config)
        sensor.load("checkpoints/soft_sensor_tower3.json")
        abundance_est = sensor.predict(live_features)
    """

    def __init__(self, config: SoftSensorConfig | None = None) -> None:
        self._cfg = config or SoftSensorConfig()
        self._model: object | None = None
        self._buffer: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str) -> None:
        """Load a pre-trained model from disk.

        Args:
            path: file path to the serialised model (JSON for XGBoost,
                .pth for LSTM).
        """
        # In production this deserialises an xgboost.Booster or torch.nn.Module.
        # Placeholder implementation — see commit 5/5 for the Transformer
        # cascade that supersedes this module.
        self._model = path  # placeholder

    def predict(self, features: np.ndarray) -> float:
        """Estimate current isotope abundance from live PLC features.

        Args:
            features: 1-D array of [T, P, F, valve%, reflux_ratio, ...].

        Returns:
            Estimated abundance (0–100 %).
        """
        if self._model is None:
            # Fallback: kinematic surrogate using temperature-pressure
            # linear approximation (valid for small perturbations around
            # the nominal operating point).
            return self._kinematic_fallback(features)
        # In production: model.predict(features.reshape(1, -1))
        return self._kinematic_fallback(features)

    def update_buffer(self, feature_vector: np.ndarray) -> None:
        """Push a new feature vector into the lookback buffer.

        Args:
            feature_vector: 1-D array matching ``input_features``.
        """
        self._buffer.append(feature_vector)
        if len(self._buffer) > self._cfg.lookback_window:
            self._buffer.pop(0)

    def predict_from_buffer(self) -> float:
        """Predict abundance using the current lookback buffer."""
        if len(self._buffer) < self._cfg.lookback_window:
            return 50.0  # default nominal
        stacked = np.stack(self._buffer[-self._cfg.lookback_window :], axis=0)
        # mean over the window as naive baseline
        return self.predict(stacked.mean(axis=0))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _kinematic_fallback(features: np.ndarray) -> float:
        """Simple linear surrogate based on temperature-pressure ratio.

        Fictitious formula — for demonstration only.
        """
        if len(features) < 2:
            return 50.0  # safe default when features are incomplete
        T_norm = float(features[0]) / 200.0  # normalise temperature
        P_norm = float(features[1]) / 15.0  # normalise pressure
        base = 95.0 + 3.0 * (1.0 - T_norm) - 2.0 * (P_norm - 1.0)
        return float(np.clip(base, 0.0, 100.0))


class SoftSensorPredictor:
    """Higher-level wrapper that combines the virtual sensor with DTW lag.

    In production, this is called by the anomaly-detection pipeline to
    provide a continuous abundance estimate synchronised with the PLC
    stream.
    """

    def __init__(
        self,
        sensor: VirtualSoftSensor | None = None,
        config: SoftSensorConfig | None = None,
    ) -> None:
        self.sensor = sensor or VirtualSoftSensor(config)

    def estimate(self, live_features: np.ndarray) -> dict[str, float]:
        """Return abundance estimate with confidence bounds.

        Args:
            live_features: current feature vector.

        Returns:
            Dict with keys ``abundance``, ``lower_bound``, ``upper_bound``.
        """
        est = self.sensor.predict(live_features)
        margin = 1.5  # ±1.5% typical uncertainty for the kinematic surrogate
        return {
            "abundance": est,
            "lower_bound": max(0.0, est - margin),
            "upper_bound": min(100.0, est + margin),
        }
