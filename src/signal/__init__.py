"""Signal processing subpackage: Kalman filtering, wavelet denoising, DTW alignment, soft sensing."""

from src.signal.kalman_filter import TwoStageKalmanFilter
from src.signal.wavelet_denoise import wavelet_denoise, WaveletDenoiser
from src.signal.scalogram import compute_scalogram, ScalogramExtractor
from src.signal.dtw_aligner import dtw_align, DTWAligner
from src.signal.soft_sensor import VirtualSoftSensor, SoftSensorPredictor

__all__ = [
    "TwoStageKalmanFilter",
    "wavelet_denoise",
    "WaveletDenoiser",
    "compute_scalogram",
    "ScalogramExtractor",
    "dtw_align",
    "DTWAligner",
    "VirtualSoftSensor",
    "SoftSensorPredictor",
]
