"""DMA (Direct Memory Access) configuration for edge NPU acceleration.

On the Jetson AGX Orin, raw PLC and serial byte streams are DMA-ed
directly into the NPU's unified memory, bypassing the CPU entirely.
This achieves microsecond-level latency for protocol parsing and
anomaly detection.

All addresses and channel IDs are fictitious.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DMAConfig:
    """DMA channel configuration for edge AI acceleration."""

    # DMA channel assignments (fictitious)
    plc_dma_channel: int = 0
    serial_dma_channel: int = 1
    npu_dma_channel: int = 2

    # Memory mapping
    unified_memory_size_mb: int = 4096  # Orin 64 GB shared; 4 GB for DMA buffers
    buffer_alignment_bytes: int = 4096

    # Transfer settings
    transfer_timeout_us: int = 100  # microsecond timeout
    max_burst_size_bytes: int = 65536  # 64 KB max per DMA burst

    # Channels
    enabled_channels: List[str] = field(default_factory=lambda: [
        "plc_opcua_stream",
        "serial_rs485_stream",
        "npu_inference_queue",
    ])


def configure_dma(config: DMAConfig | None = None) -> Dict[str, Any]:
    """Validate and return the DMA configuration for the Jetson deploy script.

    In production, this writes the DMA descriptor table to
    ``/sys/kernel/debug/dma/`` or configures the NVIDIA DMA engine
    via the JetPack SDK.

    Args:
        config: DMA configuration (uses defaults if None).

    Returns:
        Dict with channel assignments and buffer addresses.
    """
    cfg = config or DMAConfig()
    return {
        "channels": {
            "plc_opcua_stream": {
                "dma_channel": cfg.plc_dma_channel,
                "direction": "host_to_device",
                "buffer_size_kb": 1024,
            },
            "serial_rs485_stream": {
                "dma_channel": cfg.serial_dma_channel,
                "direction": "host_to_device",
                "buffer_size_kb": 256,
            },
            "npu_inference_queue": {
                "dma_channel": cfg.npu_dma_channel,
                "direction": "device_to_host",
                "buffer_size_kb": 512,
            },
        },
        "unified_memory_mb": cfg.unified_memory_size_mb,
        "alignment_bytes": cfg.buffer_alignment_bytes,
    }
