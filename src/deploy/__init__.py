"""Deployment subpackage: AWQ quantization, vLLM/TensorRT-LLM serving, Jetson edge deploy, DMA config."""

from src.deploy.quantize_awq import AWQQuantizer, quantize_model
from src.deploy.dma_config import DMAConfig, configure_dma

__all__ = [
    "AWQQuantizer",
    "quantize_model",
    "DMAConfig",
    "configure_dma",
]
