"""Models subpackage: KWT cascade, multi-scale embedding, Kalman feedback, Triton fused kernel."""

from src.models.kalman_wavelet_transformer import KWTModel, KWTransformer
from src.models.multi_scale_embedding import MultiScaleEmbedding, build_embedding
from src.models.kalman_feedback import KalmanFeedback, apply_kalman_correction
from src.models.kwt_triton_kernel import kwt_fused_embed

__all__ = [
    "KWTModel",
    "KWTransformer",
    "MultiScaleEmbedding",
    "build_embedding",
    "KalmanFeedback",
    "apply_kalman_correction",
    "kwt_fused_embed",
]
