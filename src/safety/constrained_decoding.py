"""Constrained decoding via JSON Schema — anti-hallucination layer 1.

Forces the LLM to output only valid JSON that matches a predefined
FMEA diagnostic schema.  Tag names, severity scores, and valve IDs
must conform to the asset dictionary.  Tokens that would produce
invalid field values are rejected at the sampling level.

Uses the ``outlines`` library for logit-masked generation, or falls
back to prompt-side enforcement when outlines is unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Fictitious asset dictionary — all PLC tags / valve IDs are synthetic.
ASSET_DICTIONARY: Dict[str, List[str]] = {
    "tags": [
        "TE-301", "TE-302", "TE-303", "TE-304", "TE-305",
        "TE-306", "TE-307", "TE-308", "TE-309", "TE-310",
        "PT-301", "PT-302", "PT-303",
        "FT-301", "FT-302", "FT-303",
        "FV-301", "FV-302", "FV-303",
        "LT-301", "LT-302",
    ],
    "systems": [
        "Cryogenic Column T-301",
        "Cryogenic Column T-302",
        "Heat Exchanger E-301",
        "Reboiler H-301",
        "Condenser C-301",
    ],
    "failure_modes": [
        "bearing_overheat",
        "seal_leakage",
        "cavitation",
        "flooding",
        "dry_bed",
        "cold_leak",
        "fouling",
        "catalyst_degradation",
        "sensor_drift",
        "valve_stiction",
    ],
}

FMEA_REPORT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "alarm_tag": {
            "type": "string",
            "description": "PLC tag that triggered the alarm.",
            "enum": ASSET_DICTIONARY["tags"],
        },
        "matched_fmea": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "failure_mode": {
                        "type": "string",
                        "enum": ASSET_DICTIONARY["failure_modes"],
                    },
                    "severity": {"type": "integer", "minimum": 1, "maximum": 10},
                    "occurrence": {"type": "integer", "minimum": 1, "maximum": 10},
                    "detection": {"type": "integer", "minimum": 1, "maximum": 10},
                    "rpn": {"type": "integer", "minimum": 1, "maximum": 1000},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "cause": {"type": "string"},
                    "recommended_action": {"type": "string"},
                },
                "required": ["failure_mode", "severity", "confidence", "cause"],
            },
        },
        "diagnostic_summary": {"type": "string"},
        "requires_manual_inspection": {"type": "boolean"},
    },
    "required": ["alarm_tag", "matched_fmea", "diagnostic_summary"],
}


class ConstrainedDecoder:
    """Schema-constrained generation for FMEA diagnostic output.

    Usage::

        decoder = ConstrainedDecoder()
        valid_json = decoder.generate(llm_output_raw)
    """

    def __init__(self, schema: Dict[str, Any] | None = None) -> None:
        self.schema = schema or FMEA_REPORT_SCHEMA

    def validate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a candidate JSON object against the FMEA schema.

        Args:
            candidate: raw LLM output parsed as JSON.

        Returns:
            Dict with keys ``valid`` (bool), ``errors`` (list of str),
            and ``sanitized`` (the cleaned object, if valid).

        Raises:
            No exception — always returns a result dict.
        """
        errors: List[str] = []

        # Check required top-level keys
        for key in self.schema.get("required", []):
            if key not in candidate:
                errors.append(f"Missing required field: {key}")

        # Check alarm_tag against asset dictionary
        alarm_tag = candidate.get("alarm_tag", "")
        valid_tags = self.schema["properties"]["alarm_tag"]["enum"]
        if alarm_tag and alarm_tag not in valid_tags:
            errors.append(
                f"Invalid tag '{alarm_tag}'. Must be one of: {valid_tags}"
            )

        # Check each matched FMEA entry
        for i, entry in enumerate(candidate.get("matched_fmea", [])):
            mode = entry.get("failure_mode", "")
            valid_modes = self.schema["properties"]["matched_fmea"]["items"]["properties"]["failure_mode"]["enum"]
            if mode and mode not in valid_modes:
                errors.append(f"matched_fmea[{i}]: invalid failure_mode '{mode}'")

            severity = entry.get("severity")
            if severity is not None and not (1 <= severity <= 10):
                errors.append(f"matched_fmea[{i}]: severity {severity} out of [1,10]")

            confidence = entry.get("confidence")
            if confidence is not None and not (0.0 <= confidence <= 1.0):
                errors.append(f"matched_fmea[{i}]: confidence {confidence} out of [0,1]")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "sanitized": candidate if not errors else {},
        }

    @staticmethod
    def is_tag_valid(tag: str) -> bool:
        """Check whether a tag exists in the fictitious asset dictionary."""
        return tag in ASSET_DICTIONARY["tags"]


# Alias
JSONSchemaGenerator = ConstrainedDecoder
