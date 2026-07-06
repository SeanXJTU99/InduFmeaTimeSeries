#!/usr/bin/env python3
"""
Raw covariance array packing for DMA-optimized edge inference.

When deploying Kalman filters on Jetson AGX Orin via DMA, the default
NumPy->pickle->bytes->CUDA memcpy path adds unnecessary serialization
overhead. This module packs symmetric covariance matrices into flat C arrays
(same layout as GPU Kalman track parameter buffers in high-energy physics)
for direct DMA transfer to the NPU.

A 5x5 symmetric matrix becomes a 15-element flat array (lower triangular).
An 8x8 becomes 36 elements. The NPU-side code can load these with a single
cudaMemcpy or zero-copy DMA read.
"""

from __future__ import annotations

import ctypes
import time
from typing import Optional

import numpy as np


def pack_covariance_5x5(cov: np.ndarray) -> np.ndarray:
    """Pack 5x5 symmetric covariance into 15-element flat array.

    Layout (lower-triangular, row-major):
      mC[ 0] = C[0,0]
      mC[ 1] = C[1,0], mC[ 2] = C[1,1]
      mC[ 3] = C[2,0], mC[ 4] = C[2,1], mC[ 5] = C[2,2]
      mC[ 6] = C[3,0], ...                 mC[ 9] = C[3,3]
      mC[10] = C[4,0], ...                 mC[14] = C[4,4]
    """
    if cov.shape != (5, 5):
        raise ValueError(f"Expected 5x5 covariance, got {cov.shape}")
    packed = np.empty(15, dtype=cov.dtype)
    idx = 0
    for i in range(5):
        for j in range(i + 1):
            packed[idx] = cov[i, j]
            idx += 1
    return packed


def unpack_covariance_5x5(packed: np.ndarray) -> np.ndarray:
    """Reverse pack: 15-element array -> 5x5 symmetric matrix."""
    if len(packed) != 15:
        raise ValueError(f"Expected 15 elements, got {len(packed)}")
    cov = np.zeros((5, 5), dtype=packed.dtype)
    idx = 0
    for i in range(5):
        for j in range(i + 1):
            cov[i, j] = packed[idx]
            cov[j, i] = packed[idx]
            idx += 1
    return cov


def pack_covariance_8x8(cov: np.ndarray) -> np.ndarray:
    """Pack 8x8 symmetric covariance into 36-element flat array."""
    if cov.shape != (8, 8):
        raise ValueError(f"Expected 8x8 covariance, got {cov.shape}")
    packed = np.empty(36, dtype=cov.dtype)
    idx = 0
    for i in range(8):
        for j in range(i + 1):
            packed[idx] = cov[i, j]
            idx += 1
    return packed


def unpack_covariance_8x8(packed: np.ndarray) -> np.ndarray:
    """Reverse pack: 36-element array -> 8x8 symmetric matrix."""
    if len(packed) != 36:
        raise ValueError(f"Expected 36 elements, got {len(packed)}")
    cov = np.zeros((8, 8), dtype=packed.dtype)
    idx = 0
    for i in range(8):
        for j in range(i + 1):
            cov[i, j] = packed[idx]
            cov[j, i] = packed[idx]
            idx += 1
    return cov


def allocate_dma_buffer(size: int, alignment: int = 64) -> np.ndarray:
    """Allocate a DMA-aligned buffer for Jetson NPU transfer.

    Jetson DMA engine requires 64-byte alignment for optimal throughput.

    Args:
        size: Number of float32 elements.
        alignment: Byte alignment (default 64).

    Returns:
        numpy array backed by aligned memory.
    """
    raw_size = size * np.dtype(np.float32).itemsize
    buf = ctypes.create_string_buffer(raw_size + alignment)
    raw_addr = ctypes.addressof(buf)
    aligned_addr = (raw_addr + alignment - 1) & ~(alignment - 1)
    offset = aligned_addr - raw_addr
    return np.frombuffer(buf, dtype=np.float32, count=size, offset=offset)


def prepare_kalman_dma_packet(
    state: np.ndarray,       # 5-element state vector
    covariance: np.ndarray,  # 5x5 covariance matrix
) -> np.ndarray:
    """Prepare a complete Kalman state packet for DMA transfer.

    Layout (22 floats, 88 bytes, single DMA burst):
      [0:4]   = state vector (5 floats)
      [5:19]  = packed covariance (15 floats, lower-triangular)
      [20]    = timestamp (seconds since epoch)
      [21]    = flags (bit-packed)

    Returns flat C-contiguous float32 array.
    """
    if state.shape != (5,):
        raise ValueError(f"Expected 5-element state, got {state.shape}")
    if covariance.shape != (5, 5):
        raise ValueError(f"Expected 5x5 covariance, got {covariance.shape}")

    packet = np.empty(22, dtype=np.float32)
    packet[0:5] = state.astype(np.float32)
    packet[5:20] = pack_covariance_5x5(covariance.astype(np.float32))
    packet[20] = np.float32(time.time())
    packet[21] = np.float32(0.0)
    return packet
