"""Tests for guardrails gateway."""

from src.safety.guardrails import GuardrailsGateway, GuardrailsConfig


class TestGuardrailsGateway:
    def test_passes_valid_report(self) -> None:
        gw = GuardrailsGateway()
        ok, msg = gw.check({
            "abundance_pct": 98.5,
            "matched_fmea": [{"confidence": 0.85}],
            "diagnostic_summary": "Normal diagnosis.",
        })
        assert ok, msg

    def test_rejects_over_100_abundance(self) -> None:
        gw = GuardrailsGateway()
        ok, msg = gw.check({"abundance_pct": 105.0})
        assert not ok
        assert "abundance" in msg.lower()

    def test_rejects_low_confidence(self) -> None:
        gw = GuardrailsGateway()
        ok, msg = gw.check({
            "matched_fmea": [{"confidence": 0.45}],
        })
        assert not ok

    def test_rejects_forbidden_language(self) -> None:
        gw = GuardrailsGateway()
        ok, msg = gw.check({
            "diagnostic_summary": "I am absolutely certain this is the cause.",
        })
        assert not ok

    def test_sanitize_clamps_values(self) -> None:
        gw = GuardrailsGateway()
        report = {"abundance_pct": 150.0}
        sanitized = gw.sanitize(report)
        assert sanitized["abundance_pct"] == 100.0

    def test_rejects_negative_temperature(self) -> None:
        gw = GuardrailsGateway()
        ok, msg = gw.check({"temperature_c": -300.0})
        assert not ok
