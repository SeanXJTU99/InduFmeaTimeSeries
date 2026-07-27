"""Kalman-Wavelet-Transformer (KWT) cascade — the 2026 flagship anomaly detector.

Architecture:
    1. Kalman filter (streaming, edge) → denoised PLC signals
    2. Wavelet decomposition → multi-scale time-frequency features
    3. Multi-scale embedding → structured Transformer input
    4. Transformer Encoder → cross-channel attention over time
    5. Kalman feedback loop → residual correction from real Excel data
    6. Anomaly scoring head → (regression + classification)

This supersedes the Phase-1 standalone Kalman + wavelet + XGBoost
pipeline with a unified end-to-end differentiable model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.multi_scale_embedding import MultiScaleEmbedding, EmbeddingConfig
from src.models.kalman_feedback import KalmanFeedback, KalmanFeedbackConfig


@dataclass
class KWTConfig:
    """KWT model hyperparameters."""

    # Input
    n_signals: int = 6
    d_model: int = 512
    max_seq_len: int = 1024

    # Transformer encoder
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    activation: str = "gelu"
    use_flash_attention: bool = True  # PyTorch 2.0+ SDPA Flash Attention backend

    # Kalman feedback
    kalman_q: float = 1e-4
    kalman_r: float = 1e-2

    # Langevin physics correction (PINFDiT inference plugin)
    enable_langevin: bool = False
    langevin_n_steps: int = 5
    langevin_step_size: float = 0.05
    langevin_alpha: float = 0.1

    # Output heads
    anomaly_threshold: float = 0.5


class KWTransformer(nn.Module):
    """Kalman-Wavelet-Transformer for industrial time-series anomaly detection.

    Usage::

        model = KWTransformer(KWTConfig())
        # Training
        anomaly_scores, abundance_pred = model(plc_batch)
        loss = bce(anomaly_scores, labels) + mse(abundance_pred, abundance_true)
        # Inference with Kalman feedback
        scores, pred = model(plc_batch, excel_measurement=real_abundance)
    """

    def __init__(self, config: KWTConfig | None = None) -> None:
        super().__init__()
        cfg = config or KWTConfig()

        # Multi-scale embedding (wavelet → linear projection)
        emb_cfg = EmbeddingConfig(
            n_signals=cfg.n_signals, d_model=cfg.d_model,
            max_seq_len=cfg.max_seq_len, dropout=cfg.dropout,
        )
        self.embedding = MultiScaleEmbedding(emb_cfg)

        # Transformer encoder with Flash Attention (PyTorch 2.0+ sdpa)
        self._use_flash_attn = cfg.use_flash_attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation=cfg.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # Kalman feedback loop
        kf_cfg = KalmanFeedbackConfig(
            hidden_dim=cfg.d_model,
            process_noise=cfg.kalman_q,
            measurement_noise=cfg.kalman_r,
        )
        self.kalman_feedback = KalmanFeedback(kf_cfg)

        # Output heads
        self.anomaly_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.abundance_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
            nn.Sigmoid(),  # output in [0, 1] → scaled to [0, 100]%
        )

    def forward(
        self,
        plc_signals: torch.Tensor,
        excel_measurement: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            plc_signals: (batch, seq_len, n_signals) — Kalman-filtered PLC data.
            excel_measurement: optional (batch, 1) — real abundance from async Excel.

        Returns:
            (anomaly_scores, abundance_pred) tuple.
            - anomaly_scores: (batch, 1) — probability of anomaly.
            - abundance_pred: (batch, 1) — predicted abundance in [0, 1].
        """
        # 1. Multi-scale embedding
        emb = self.embedding(plc_signals)  # (B, T, d_model)

        # 2. Transformer encoder (Flash Attention via PyTorch 2.0+ SDPA)
        if self._use_flash_attn and torch.cuda.is_available():
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True, enable_math=False, enable_mem_efficient=False,
                ):
                    encoded = self.encoder(emb)  # (B, T, d_model)
            except RuntimeError:
                # Flash Attention not supported (head dim, seq len, GPU arch) —
                # fall back to PyTorch's default SDPA backend (math/mem_efficient).
                encoded = self.encoder(emb)
        else:
            encoded = self.encoder(emb)  # (B, T, d_model)

        # 3. Pool over time dimension
        pooled = encoded.mean(dim=1)  # (B, d_model)

        # 4. Kalman feedback correction (if real measurement available)
        if excel_measurement is not None:
            pooled = self.kalman_feedback(pooled, excel_measurement)

        # 5. Output heads
        anomaly_scores = self.anomaly_head(pooled)  # (B, 1)
        abundance_pred = self.abundance_head(pooled)  # (B, 1)

        return anomaly_scores, abundance_pred

    def apply_langevin(
        self,
        anomaly_scores: torch.Tensor,
        abundance_pred: torch.Tensor,
        features: dict[str, float] | None = None,
        kwt_baseline: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Langevin physics correction to scalar outputs (inference only).

        This is the PINFDiT "Generalist-to-Specialist" plugin: the frozen
        KWT produces a statistically-plausible prediction, then Langevin
        dynamics steers it toward physical consistency without retraining.

        Args:
            anomaly_scores: (B, 1) tensor from KWT forward pass.
            abundance_pred: (B, 1) tensor from KWT forward pass.
            features: optional PLC measurements per batch item.
            kwt_baseline: original KWT predictions (used as score target).

        Returns:
            (refined_anomaly_scores, refined_abundance_pred) — same shapes.
        """
        if not self.config.enable_langevin:
            return anomaly_scores, abundance_pred

        try:
            from src.models.langevin_correction import (
                LangevinConfig,
                LangevinCorrector,
            )
            from src.detection.energy_function import DistillationEnergyFunction

            lc_cfg = LangevinConfig(
                n_steps=self.config.langevin_n_steps,
                step_size=self.config.langevin_step_size,
                alpha=self.config.langevin_alpha,
            )
            lc = LangevinCorrector(
                energy_func=DistillationEnergyFunction(),
                config=lc_cfg,
            )
        except ImportError:
            return anomaly_scores, abundance_pred

        batch_size = anomaly_scores.shape[0]
        refined_scores = []
        refined_abund = []

        for i in range(batch_size):
            a_score = float(anomaly_scores[i].item())
            a_abund = float(abundance_pred[i].item()) * 100.0  # [0,1] → %

            preds = {"abundance_pct": a_abund, "anomaly_score": a_score}
            baseline = kwt_baseline.copy() if kwt_baseline else dict(preds)
            feats = features or {}

            refined = lc.correct(preds, feats, baseline)
            refined_scores.append(refined["anomaly_score"])
            refined_abund.append(refined["abundance_pct"] / 100.0)

        return (
            torch.tensor(refined_scores, device=anomaly_scores.device,
                         dtype=anomaly_scores.dtype).view(-1, 1),
            torch.tensor(refined_abund, device=abundance_pred.device,
                         dtype=abundance_pred.dtype).view(-1, 1),
        )

    def reset_kalman(self) -> None:
        """Reset the Kalman feedback state."""
        self.kalman_feedback.reset()


# Backward-compatible alias
KWTModel = KWTransformer
