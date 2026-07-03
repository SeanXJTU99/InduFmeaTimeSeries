"""Kalman feedback loop — corrects Transformer hidden states with real measurements.

The Transformer acts as a virtual soft sensor, predicting isotope
abundance from PLC streams.  When the async Excel report finally
arrives (30-60 min later), the residual between predicted and actual
abundance is fed through a Kalman filter that corrects the Transformer's
hidden states, preventing long-term trajectory drift.

This is the "double closed-loop" architecture described in the 2026
world-model design:  Kalman (physical filter) ←→ Transformer (learned
predictor).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class KalmanFeedbackConfig:
    """Kalman feedback loop parameters."""

    hidden_dim: int = 512
    process_noise: float = 1e-4  # Q — trust in Transformer prediction
    measurement_noise: float = 1e-2  # R — trust in real Excel measurement
    initial_covariance: float = 1.0


class KalmanFeedback(nn.Module):
    """Kalman-based correction layer for Transformer hidden states.

    Usage::

        kfb = KalmanFeedback()
        transformer_output = model(plc_input)  # (B, T, hidden_dim)
        # ... 45 minutes later, real Excel data arrives ...
        corrected = kfb(transformer_output, real_measurement)
    """

    def __init__(self, config: KalmanFeedbackConfig | None = None) -> None:
        super().__init__()
        cfg = config or KalmanFeedbackConfig()
        self._Q = cfg.process_noise
        self._R = cfg.measurement_noise
        self._P: Optional[torch.Tensor] = None  # state covariance
        self._state: Optional[torch.Tensor] = None
        # Learnable observation matrix: maps hidden state → abundance
        self.obs_proj = nn.Linear(cfg.hidden_dim, 1)

    def forward(
        self, hidden: torch.Tensor, measurement: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Optionally correct hidden states with a real measurement.

        Args:
            hidden: Transformer output, (batch, hidden_dim) or (batch, seq, hidden_dim).
            measurement: real abundance value (scalar or batch tensor).
                If None, returns hidden unchanged (prediction mode).

        Returns:
            Corrected hidden states (same shape as *hidden*).
        """
        if measurement is None:
            return hidden

        # Reduce to batch-mean if sequential
        if hidden.dim() == 3:
            h = hidden.mean(dim=1)  # (B, hidden_dim)
        else:
            h = hidden

        # Kalman prediction
        if self._state is None:
            self._state = h.detach().clone()
            self._P = torch.full(
                (h.shape[-1], h.shape[-1]),
                self._Q + self._R,
                device=h.device, dtype=h.dtype,
            )

        # Innovation: real abundance - predicted abundance
        pred_abundance = self.obs_proj(self._state)
        innovation = measurement - pred_abundance

        # Kalman gain (simplified scalar form)
        # K = P_pred * H^T / (H * P_pred * H^T + R)
        p_diag = torch.diag(self._P)
        k_gain = p_diag / (p_diag + self._R)

        # Update state
        self._state = self._state + k_gain.unsqueeze(-1) * innovation.unsqueeze(-1) * self.obs_proj.weight
        self._P = self._P - torch.outer(k_gain, k_gain) * (p_diag + self._R).mean()

        # Apply correction to hidden
        correction = self._state - h
        return hidden + correction.unsqueeze(0) if hidden.dim() == 3 else hidden + correction

    def reset(self) -> None:
        """Reset Kalman state (e.g. after column maintenance)."""
        self._state = None
        self._P = None


def apply_kalman_correction(
    hidden: torch.Tensor,
    measurement: torch.Tensor,
) -> torch.Tensor:
    """Convenience: apply Kalman feedback correction in one call."""
    return KalmanFeedback().forward(hidden, measurement)
