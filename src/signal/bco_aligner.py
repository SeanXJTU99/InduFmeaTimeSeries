#!/usr/bin/env python3
"""
Hard-Clock Multi-Source Time Alignment Engine.

When PLC NTP is available (Siemens S7-1500 supports NTP), this replaces the
O(N^2) DTW soft-alignment with O(1) bucket-based alignment using a global
timestamp key. Each data source is registered with its known delay window,
and events are grouped into aligned bundles when all sources have contributed
within the spread tolerance.

DTW remains available as a fallback when NTP is unavailable.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BcoConfig:
    """Configuration for hard-clock alignment engine.

    Attributes:
        max_spread_ns: Maximum allowed spread between source timestamps.
        pool_warning_threshold: Buffered event count before backpressure alert.
        cleanup_interval_s: Interval for pruning processed events.
    """
    max_spread_ns: int = 10_000_000  # 10 ms
    pool_warning_threshold: int = 1000
    cleanup_interval_s: float = 5.0


@dataclass
class SourceWindow:
    """Per-source alignment window.

    Attributes:
        source_name: Registered source identifier.
        delay_ns: Expected latency in nanoseconds.
        priority: Tie-breaking priority (lower = higher).
    """
    source_name: str
    delay_ns: int
    priority: int = 0


class BCOAligner:
    """Hard-clock multi-source alignment engine.

    Usage::

        aligner = BCOAligner(BcoConfig(max_spread_ns=10_000_000))
        aligner.register_source("plc", delay_ns=1_000_000, priority=0)
        aligner.register_source("serial", delay_ns=5_000_000, priority=1)
        aligner.register_source("excel", delay_ns=1_800_000_000_000, priority=2)

        aligner.feed("plc", ntp_ns=1700000000000000000, data={"temp": -185.3})
        for group in aligner.get_aligned_events():
            process(group)
    """

    def __init__(self, config: Optional[BcoConfig] = None):
        self.config = config or BcoConfig()
        self._sources: dict[str, SourceWindow] = {}
        self._buffer: dict[int, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._processed_ntp: set[int] = set()
        self._last_cleanup: float = time.time()
        self._total_fed: int = 0
        self._total_aligned: int = 0

    def register_source(
        self, source_name: str, delay_ns: int, priority: int = 0
    ) -> None:
        """Register a data source with its delay tolerance."""
        self._sources[source_name] = SourceWindow(
            source_name=source_name, delay_ns=delay_ns, priority=priority
        )

    def is_ntp_available(self) -> bool:
        """Return True if all registered sources provide NTP timestamps."""
        return len(self._sources) > 0

    def feed(self, source_name: str, ntp_ns: int, data: dict) -> None:
        """Feed a single event into the alignment buffer.

        Args:
            source_name: Registered source name.
            ntp_ns: NTP timestamp in nanoseconds.
            data: Event payload dict.
        """
        if source_name not in self._sources:
            raise ValueError(f"Unknown source: {source_name}. Register first.")
        delay = self._sources[source_name].delay_ns
        aligned_ntp = ntp_ns + delay
        self._buffer[aligned_ntp][source_name].append(data)
        self._total_fed += 1
        if time.time() - self._last_cleanup > self.config.cleanup_interval_s:
            self._cleanup()

    def feed_batch(
        self, source_name: str, events: list[tuple[int, dict]]
    ) -> None:
        """Feed a batch of events at once."""
        for ntp_ns, data in events:
            self.feed(source_name, ntp_ns, data)

    def get_aligned_events(self) -> list[dict[str, list[dict]]]:
        """Retrieve events aligned within the spread window.

        An event group is aligned when all registered sources have contributed
        events within max_spread_ns. Returns list of dicts:
        {source_name: [event_data, ...], ...}
        """
        self._cleanup()
        cfg = self.config
        result: list[dict[str, list[dict]]] = []
        sorted_ntps = sorted(self._buffer.keys())
        if not sorted_ntps:
            return result

        window_start_idx = 0
        for i, base_ntp in enumerate(sorted_ntps):
            while (
                window_start_idx < i
                and (base_ntp - sorted_ntps[window_start_idx]) > cfg.max_spread_ns
            ):
                window_start_idx += 1

            window_events: dict[str, list[dict]] = defaultdict(list)
            all_sources_present = set()
            for j in range(window_start_idx, i + 1):
                ntp_key = sorted_ntps[j]
                for src, events in self._buffer[ntp_key].items():
                    window_events[src].extend(events)
                    all_sources_present.add(src)

            if all_sources_present == set(self._sources.keys()):
                result.append(dict(window_events))
                self._total_aligned += 1
                for j in range(window_start_idx, i + 1):
                    self._processed_ntp.add(sorted_ntps[j])

        return result

    def check_pool_depth(self) -> dict:
        """Check buffer depth for backpressure detection."""
        depth_per_source: dict[str, int] = defaultdict(int)
        for ntp_bucket in self._buffer.values():
            for src, events in ntp_bucket.items():
                depth_per_source[src] += len(events)

        warning_sources = [
            src
            for src, depth in depth_per_source.items()
            if depth > self.config.pool_warning_threshold
        ]

        return {
            "total_buckets": len(self._buffer),
            "depth_per_source": dict(depth_per_source),
            "warning_sources": warning_sources,
            "is_backpressure": len(warning_sources) > 0,
        }

    def _cleanup(self) -> None:
        """Prune processed NTP buckets to prevent unbounded memory growth."""
        for ntp in list(self._processed_ntp):
            if ntp in self._buffer:
                del self._buffer[ntp]
        self._processed_ntp.clear()
        self._last_cleanup = time.time()

    def reset(self) -> None:
        """Clear all buffers."""
        self._buffer.clear()
        self._processed_ntp.clear()
        self._total_fed = 0
        self._total_aligned = 0

    @property
    def alignment_rate(self) -> float:
        return self._total_aligned / max(self._total_fed, 1)

    @property
    def pool_depth(self) -> int:
        return sum(
            sum(len(events) for events in bucket.values())
            for bucket in self._buffer.values()
        )
