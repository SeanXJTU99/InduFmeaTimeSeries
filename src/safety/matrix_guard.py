"""
Fast 3D boolean matrix safety gateway for deterministic hard-rule checks.

Instead of chaining JSON Schema + Pydantic + Constrained Decoding for every
safety rule (enrichment > 100%, valve position < 0%, thermal limit exceeded),
this module uses a pre-configured 3D boolean matrix:

    state[device_type][sensor_id][severity] -> O(1) allow/block

Hard rules are resolved in nanoseconds with a single array lookup. The LLM
is only invoked for soft reasoning paths (uncertainty cases). Known-safe
sensor readings bypass the LLM entirely.

This reduces the four-layer defense (JSON Schema -> Pydantic -> BM25+BGE ->
Guardrails) to two layers: matrix lookup (hard) + LLM reasoning (soft).
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


class Severity(IntEnum):
    """Severity levels for industrial FMEA safety gating."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3
    PHYSICAL_IMPOSSIBLE = 4


@dataclass
class MatrixGuardConfig:
    """Configuration for the 3D boolean matrix gateway.

    Attributes:
        n_device_types: Number of device type categories.
        n_sensors: Max sensor ID count.
        n_severities: Number of severity levels (default 5).
        default_action: Default for unconfigured cells (True = allow).
    """
    n_device_types: int = 50
    n_sensors: int = 2000
    n_severities: int = 5
    default_action: bool = True


class MatrixGuard:
    """3D boolean matrix for O(1) hard-rule safety gating.

    Usage::

        guard = MatrixGuard(MatrixGuardConfig())
        guard.block(device_type=3, sensor_id=101, severity=Severity.PHYSICAL_IMPOSSIBLE)
        guard.allow(device_type=3, sensor_id=999, severity=Severity.INFO)

        if guard.check(device_type=3, sensor_id=101, severity=4):
            ...  # pass to LLM for soft reasoning
        else:
            ...  # blocked: dump to fallback / human override
    """

    def __init__(self, config: Optional[MatrixGuardConfig] = None):
        self.config = config or MatrixGuardConfig()
        cfg = self.config
        self._state = np.full(
            (cfg.n_device_types, cfg.n_sensors, cfg.n_severities),
            cfg.default_action,
            dtype=bool,
        )
        self._device_names: dict[int, str] = {}
        self._sensor_tags: dict[int, str] = {}
        self._block_count: int = 0
        self._allow_count: int = 0

    #
    # --- Per-cell configuration ---------------------------------------------
    #

    def block(
        self,
        device_type: int,
        sensor_id: int,
        severity: Severity,
    ) -> None:
        """Set a cell to BLOCK."""
        self._validate_indices(device_type, sensor_id, severity)
        self._state[device_type, sensor_id, int(severity)] = False

    def allow(
        self,
        device_type: int,
        sensor_id: int,
        severity: Severity,
    ) -> None:
        """Set a cell to ALLOW."""
        self._validate_indices(device_type, sensor_id, severity)
        self._state[device_type, sensor_id, int(severity)] = True

    def block_all_severities(self, device_type: int, sensor_id: int) -> None:
        """Block a sensor across all severity levels."""
        for sev in Severity:
            self.block(device_type, sensor_id, sev)

    def allow_all_severities(self, device_type: int, sensor_id: int) -> None:
        """Allow a sensor across all severity levels."""
        for sev in Severity:
            self.allow(device_type, sensor_id, sev)

    #
    # --- Bulk operations ----------------------------------------------------
    #

    def block_severity_below(self, threshold: Severity) -> None:
        """Globally block all messages below a severity threshold."""
        cfg = self.config
        for dt in range(cfg.n_device_types):
            for sid in range(cfg.n_sensors):
                for sev in range(min(int(threshold), cfg.n_severities)):
                    self._state[dt, sid, sev] = False

    def set_device_type(self, device_type: int, action: bool) -> None:
        """Allow or block all sensors of a device type."""
        cfg = self.config
        for sid in range(cfg.n_sensors):
            for sev in range(cfg.n_severities):
                self._state[device_type, sid, sev] = action

    def all_on(self) -> None:
        """Reset entire matrix to ALLOW (emergency override)."""
        self._state.fill(True)

    def all_off(self) -> None:
        """Set entire matrix to BLOCK (safety shutdown)."""
        self._state.fill(False)

    #
    # --- Query API ----------------------------------------------------------
    #

    def check(self, device_type: int, sensor_id: int, severity: int) -> bool:
        """Check if a message should be allowed.

        Returns:
            True = ALLOW (pass to downstream), False = BLOCK.
        """
        try:
            result = bool(self._state[device_type, sensor_id, severity])
        except IndexError:
            result = self.config.default_action

        if result:
            self._allow_count += 1
        else:
            self._block_count += 1
        return result

    def check_and_route(
        self, device_type: int, sensor_id: int, severity: int
    ) -> str:
        """Check matrix and return routing decision.

        Returns:
            "llm" — soft reasoning (uncertainty, send to LLM).
            "block" — hard rule violation (dump to fallback).
            "pass" — known-safe (bypass LLM, use template response).
        """
        allowed = self.check(device_type, sensor_id, severity)
        if not allowed:
            return "block"
        if severity <= Severity.INFO:
            return "pass"
        return "llm"

    def get_cell_state(
        self, device_type: int, sensor_id: int, severity: Severity
    ) -> bool:
        """Get raw cell state without incrementing counters."""
        try:
            return bool(self._state[device_type, sensor_id, int(severity)])
        except IndexError:
            return self.config.default_action

    #
    # --- Bulk loading ------------------------------------------------------
    #

    def load_physical_bounds(self, bounds: dict) -> int:
        """Load physical impossibility rules from a config dict.

        Expected format::

            {
                "enrichment": {"max": 100.0, "device_type": 0, "sensor_id": 50},
                "valve_position": {"min": 0.0, "device_type": 1, "sensor_id": 200},
            }

        Returns count of rules loaded.
        """
        count = 0
        for _rule_name, rule in bounds.items():
            if "device_type" in rule and "sensor_id" in rule:
                self.block(
                    rule["device_type"],
                    rule["sensor_id"],
                    Severity.PHYSICAL_IMPOSSIBLE,
                )
                count += 1
            elif "devices" in rule:
                for dt_str, dev_cfg in rule["devices"].items():
                    dt = int(dt_str)
                    for sid in dev_cfg.get("sensor_ids", []):
                        self.block(dt, sid, Severity.PHYSICAL_IMPOSSIBLE)
                        count += 1
        return count

    def load_allowlist(self, allowlist: list[dict]) -> int:
        """Load known-safe sensor allowlist.

        Returns count of rules loaded.
        """
        count = 0
        for entry in allowlist:
            dt = entry["device_type"]
            sid = entry["sensor_id"]
            for sev_val in entry.get("severities", [0]):
                self.allow(dt, sid, Severity(sev_val))
                count += 1
        return count

    #
    # --- Statistics ---------------------------------------------------------
    #

    @property
    def block_count(self) -> int:
        return self._block_count

    @property
    def allow_count(self) -> int:
        return self._allow_count

    @property
    def block_ratio(self) -> float:
        total = self._block_count + self._allow_count
        return self._block_count / total if total > 0 else 0.0

    #
    # --- Name registry ------------------------------------------------------
    #

    def register_device_name(self, device_type: int, name: str) -> None:
        self._device_names[device_type] = name

    def register_sensor_tag(self, sensor_id: int, tag: str) -> None:
        self._sensor_tags[sensor_id] = tag

    def get_device_name(self, device_type: int) -> str:
        return self._device_names.get(device_type, f"Device-{device_type}")

    def get_sensor_tag(self, sensor_id: int) -> str:
        return self._sensor_tags.get(sensor_id, f"Sensor-{sensor_id}")

    #
    # --- Internal -----------------------------------------------------------
    #

    def _validate_indices(
        self, device_type: int, sensor_id: int, severity: Severity
    ) -> None:
        cfg = self.config
        if device_type >= cfg.n_device_types:
            raise IndexError(
                f"device_type {device_type} >= max {cfg.n_device_types}"
            )
        if sensor_id >= cfg.n_sensors:
            raise IndexError(
                f"sensor_id {sensor_id} >= max {cfg.n_sensors}"
            )
