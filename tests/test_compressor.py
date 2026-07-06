"""Tests for dictionary quantization compressor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.signal.compressor import (
    CompressionResult,
    DictionaryCompressor,
    StreamingCompressor,
)


class TestDictionaryCompressor:

    def test_empty_input(self):
        comp = DictionaryCompressor(nbits=8)
        result = comp.compress(np.array([], dtype=np.float32))
        assert len(result.order) == 0
        assert result.original_nbytes == 0

    def test_few_unique_values_no_merge(self):
        """When unique values <= 2^nbits, no merging needed."""
        comp = DictionaryCompressor(nbits=8)
        data = np.array([1.0, 2.0, 1.0, 2.0, 3.0], dtype=np.float32)
        result = comp.compress(data)
        assert len(result.dict_) <= 8
        assert result.rms_error < 1e-6

    def test_storage_reduction(self):
        """Verify that compression reduces storage."""
        comp = DictionaryCompressor(nbits=8)
        np.random.seed(42)
        data = np.random.normal(14.0, 0.5, 2000).astype(np.float32)
        result = comp.compress(data)
        assert result.compressed_nbytes < result.original_nbytes
        assert result.ratio < 1.0

    def test_ratio_improves_with_lower_bits(self):
        """Lower nbits → higher compression ratio."""
        np.random.seed(42)
        data = np.random.normal(14.0, 0.5, 1000).astype(np.float32)

        r5 = DictionaryCompressor(nbits=5).compress(data)
        r8 = DictionaryCompressor(nbits=8).compress(data)

        assert r5.ratio < r8.ratio  # 5-bit more aggressive

    def test_reconstruction_error_bounded(self):
        """RMS error should be small for well-behaved data."""
        comp = DictionaryCompressor(nbits=8)
        data = np.linspace(0, 100, 500).astype(np.float32)
        result = comp.compress(data)
        # For linear data with 256 clusters, RMS error < 1% of range
        assert result.rms_error < 1.0

    def test_reconstruct_roundtrip(self):
        comp = DictionaryCompressor(nbits=8)
        data = np.array([14.1, 14.2, 14.15, 14.18, 14.12], dtype=np.float32)
        result = comp.compress(data)
        recon = comp.reconstruct(result)
        assert len(recon) == len(data)
        assert np.max(np.abs(recon - data)) < 0.1

    def test_invalid_nbits(self):
        with pytest.raises(ValueError):
            DictionaryCompressor(nbits=1)
        with pytest.raises(ValueError):
            DictionaryCompressor(nbits=17)

    def test_all_identical_values(self):
        """Degenerate case: single unique value."""
        comp = DictionaryCompressor(nbits=8)
        data = np.full(100, 3.14, dtype=np.float32)
        result = comp.compress(data)
        assert len(result.dict_) == 1
        assert result.rms_error < 1e-6

    def test_nbits_boundary(self):
        """Verify index dtype changes at nbits=8."""
        c8 = DictionaryCompressor(nbits=8)
        c9 = DictionaryCompressor(nbits=9)
        data = np.random.randn(500).astype(np.float32)

        r8 = c8.compress(data)
        r9 = c9.compress(data)

        assert r8.order.dtype == np.uint8
        assert r9.order.dtype == np.uint16


class TestStreamingCompressor:

    def test_buffering(self):
        sc = StreamingCompressor(nbits=8, block_size=100)
        for _ in range(99):
            result = sc.feed(14.0)
            assert result is None
        assert sc.buffer_fill == pytest.approx(0.99)

    def test_block_triggers_compression(self):
        sc = StreamingCompressor(nbits=8, block_size=50)
        result = None
        for i in range(50):
            result = sc.feed(14.0 + 0.1 * np.random.randn())
        assert result is not None
        assert isinstance(result, CompressionResult)
        assert len(result.order) == 50

    def test_flush_remaining(self):
        sc = StreamingCompressor(nbits=8, block_size=100)
        for i in range(30):
            sc.feed(14.0 + 0.1 * np.random.randn())
        result = sc.flush()
        assert result is not None
        assert len(result.order) == 30

    def test_sliding_window_behavior(self):
        """After block completes, buffer resets for next window."""
        sc = StreamingCompressor(nbits=8, block_size=50)
        for i in range(50):
            sc.feed(14.0)
        result = sc.feed(14.0)  # 51st: starts new block
        assert result is not None  # first block output
        assert sc.buffer_fill == pytest.approx(1 / 50)  # one value in new buffer
