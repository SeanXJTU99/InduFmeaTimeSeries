"""Pydantic runtime validator — anti-hallucination layer 2.

After constrained decoding, a Pydantic model performs strong runtime
type-checking on the generated FMEA report.  This catches edge cases
that schema validation misses (e.g. negative RPN values, malformed
dates, inconsistent severity/RPN relationships).

All models use fictitious tag names.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class FMEAEntry(BaseModel):
    """A single matched FMEA entry within a diagnostic report."""

    failure_mode: str = Field(..., description="Failure mode identifier")
    severity: int = Field(..., ge=1, le=10, description="Severity score (1-10)")
    occurrence: int = Field(default=1, ge=1, le=10, description="Occurrence score (1-10)")
    detection: int = Field(default=1, ge=1, le=10, description="Detection score (1-10)")
    rpn: int = Field(default=1, ge=1, le=1000, description="Risk Priority Number")
    confidence: float = Field(..., ge=0.0, le=1.0, description="LLM confidence (0-1)")
    cause: str = Field(..., min_length=1, description="Root cause description")
    recommended_action: str = Field(default="", description="Recommended action")

    @field_validator("failure_mode")
    @classmethod
    def mode_must_be_known(cls, v: str) -> str:
        known = {
            "bearing_overheat", "seal_leakage", "cavitation", "flooding",
            "dry_bed", "cold_leak", "fouling", "catalyst_degradation",
            "sensor_drift", "valve_stiction",
        }
        if v not in known:
            raise ValueError(f"Unknown failure mode: {v}. Must be one of {sorted(known)}")
        return v

    @model_validator(mode="after")
    def rpn_must_match_product(self) -> "FMEAEntry":
        expected = self.severity * self.occurrence * self.detection
        if self.rpn != expected:
            raise ValueError(
                f"RPN {self.rpn} does not equal S×O×D = {self.severity}×{self.occurrence}×{self.detection} = {expected}"
            )
        return self


class FMEAReport(BaseModel):
    """Top-level FMEA diagnostic report."""

    alarm_tag: str = Field(..., min_length=1, description="PLC tag that triggered alarm")
    matched_fmea: List[FMEAEntry] = Field(..., min_length=0, max_length=5)
    diagnostic_summary: str = Field(..., min_length=1)
    requires_manual_inspection: bool = Field(default=False)
    citation_sources: List[str] = Field(default_factory=list)

    @field_validator("alarm_tag")
    @classmethod
    def tag_must_be_known(cls, v: str) -> str:
        valid = {
            "TE-301", "TE-302", "TE-303", "TE-304", "TE-305",
            "TE-306", "TE-307", "TE-308", "TE-309", "TE-310",
            "PT-301", "PT-302", "PT-303",
            "FT-301", "FT-302", "FT-303",
            "FV-301", "FV-302", "FV-303",
            "LT-301", "LT-302",
        }
        if v not in valid:
            raise ValueError(f"Unknown tag: {v}. Must be a valid fictitious PLC tag.")
        return v


class FMEAReportValidator:
    """Validate FMEA diagnostic reports at runtime.

    Usage::

        validator = FMEAReportValidator()
        report = validator.validate(raw_dict)
    """

    def validate(self, data: dict) -> FMEAReport:
        """Parse and validate a raw dict into a FMEAReport.

        Args:
            data: raw dictionary from LLM output.

        Returns:
            Validated :class:`FMEAReport` instance.

        Raises:
            pydantic.ValidationError: if validation fails.
        """
        return FMEAReport.model_validate(data)

    def safe_validate(self, data: dict) -> tuple[Optional[FMEAReport], List[str]]:
        """Validate without raising — returns (report, errors).

        Args:
            data: raw dictionary from LLM output.

        Returns:
            (report_or_none, list_of_error_strings).
        """
        try:
            return self.validate(data), []
        except Exception as e:
            return None, [str(e)]


def validate_report(data: dict) -> FMEAReport:
    """Convenience: validate a raw report dict."""
    return FMEAReportValidator().validate(data)
