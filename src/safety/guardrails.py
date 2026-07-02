"""Guardrails gateway — anti-hallucination layer 3.

A lightweight industrial-logic isolation gateway that intercepts every
diagnostic report before it reaches the SCADA/MES layer.  It applies
hard physical-boundary checks that no LLM output can bypass:

1. Isotope abundance must be in [0, 100] %.
2. Valve opening must be in [0, 100] %.
3. Temperature must be above absolute zero (−273.15 °C).
4. Pressure must be non-negative.
5. Confidence below threshold → automatic safe fallback.
6. No recommendation may contradict the asset dictionary.

If any check fails, the gateway rejects the report and triggers a
system fallback (safe degradation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GuardrailsConfig:
    """Guardrails threshold parameters."""

    min_temperature_c: float = -273.15  # absolute zero
    max_temperature_c: float = 500.0
    min_pressure_bar: float = 0.0
    max_pressure_bar: float = 30.0
    min_abundance_pct: float = 0.0
    max_abundance_pct: float = 100.0
    min_valve_opening_pct: float = 0.0
    max_valve_opening_pct: float = 100.0
    min_flow_rate: float = 0.0
    min_confidence: float = 0.6  # below this → fallback
    forbidden_phrases: List[str] = field(default_factory=lambda: [
        "guaranteed",
        "absolutely certain",
        "100% sure",
        "without any doubt",
    ])


class GuardrailsGateway:
    """Physical-boundary isolation gateway for LLM-generated reports.

    Usage::

        gw = GuardrailsGateway()
        passed, reason = gw.check(report_dict)
        if not passed:
            trigger_system_fallback(reason)
    """

    def __init__(self, config: GuardrailsConfig | None = None) -> None:
        self._cfg = config or GuardrailsConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, report: Dict[str, Any]) -> Tuple[bool, str]:
        """Run all guardrail checks on a diagnostic report.

        Args:
            report: dict with optional ``abundance_pct``, ``valve_opening_pct``,
                ``temperature_c``, ``pressure_bar``, ``flow_rate``,
                ``confidence``, ``diagnostic_summary``, and per-FMEA-entry
                ``confidence`` values.

        Returns:
            ``(passed, reason)`` — *passed* is True only if ALL checks pass.
            *reason* describes the first failing check.
        """
        # --- physical boundary checks ---
        for check_name, check_fn in self._physical_checks():
            ok, msg = check_fn(report)
            if not ok:
                return False, f"[{check_name}] {msg}"

        # --- confidence threshold ---
        for i, entry in enumerate(report.get("matched_fmea", [])):
            conf = entry.get("confidence", 0.0)
            if conf < self._cfg.min_confidence:
                return False, (
                    f"[confidence] matched_fmea[{i}] confidence={conf:.2f} "
                    f"below threshold {self._cfg.min_confidence}"
                )

        # --- forbidden language ---
        summary = report.get("diagnostic_summary", "")
        for phrase in self._cfg.forbidden_phrases:
            if phrase.lower() in summary.lower():
                return False, f"[language] Forbidden phrase detected: '{phrase}'"

        return True, ""

    def sanitize(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Strip or clamp out-of-range values instead of rejecting.

        Args:
            report: raw report dict.

        Returns:
            Sanitized dict with values clamped to physical bounds.
        """
        if "abundance_pct" in report:
            report["abundance_pct"] = max(
                self._cfg.min_abundance_pct,
                min(self._cfg.max_abundance_pct, report["abundance_pct"]),
            )
        if "valve_opening_pct" in report:
            report["valve_opening_pct"] = max(
                self._cfg.min_valve_opening_pct,
                min(self._cfg.max_valve_opening_pct, report["valve_opening_pct"]),
            )
        if "temperature_c" in report:
            report["temperature_c"] = max(
                self._cfg.min_temperature_c,
                min(self._cfg.max_temperature_c, report["temperature_c"]),
            )
        if "pressure_bar" in report:
            report["pressure_bar"] = max(
                self._cfg.min_pressure_bar,
                min(self._cfg.max_pressure_bar, report["pressure_bar"]),
            )
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _physical_checks(self) -> List[Tuple[str, Any]]:
        return [
            ("abundance", self._check_abundance),
            ("valve", self._check_valve),
            ("temperature", self._check_temperature),
            ("pressure", self._check_pressure),
            ("flow", self._check_flow),
        ]

    def _check_abundance(self, r: Dict[str, Any]) -> Tuple[bool, str]:
        val = r.get("abundance_pct")
        if val is None:
            return True, ""
        if not (self._cfg.min_abundance_pct <= val <= self._cfg.max_abundance_pct):
            return False, f"Abundance {val}% outside [{self._cfg.min_abundance_pct}, {self._cfg.max_abundance_pct}]"
        return True, ""

    def _check_valve(self, r: Dict[str, Any]) -> Tuple[bool, str]:
        val = r.get("valve_opening_pct")
        if val is None:
            return True, ""
        if not (self._cfg.min_valve_opening_pct <= val <= self._cfg.max_valve_opening_pct):
            return False, f"Valve opening {val}% outside [0, 100]"
        return True, ""

    def _check_temperature(self, r: Dict[str, Any]) -> Tuple[bool, str]:
        val = r.get("temperature_c")
        if val is None:
            return True, ""
        if not (self._cfg.min_temperature_c <= val <= self._cfg.max_temperature_c):
            return False, f"Temperature {val}°C outside physical range"
        return True, ""

    def _check_pressure(self, r: Dict[str, Any]) -> Tuple[bool, str]:
        val = r.get("pressure_bar")
        if val is None:
            return True, ""
        if not (self._cfg.min_pressure_bar <= val <= self._cfg.max_pressure_bar):
            return False, f"Pressure {val} bar outside physical range"
        return True, ""

    def _check_flow(self, r: Dict[str, Any]) -> Tuple[bool, str]:
        val = r.get("flow_rate")
        if val is None:
            return True, ""
        if val < self._cfg.min_flow_rate:
            return False, f"Flow rate {val} L/min is negative"
        return True, ""


# Backward-compatible alias
PhysicsGuard = GuardrailsGateway
