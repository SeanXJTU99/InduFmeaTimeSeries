"""Safety subpackage: constrained decoding, Pydantic validation, guardrails, citation tracking, structured output (xgrammar)."""

from src.safety.constrained_decoding import ConstrainedDecoder, JSONSchemaGenerator
from src.safety.pydantic_validator import FMEAReportValidator, validate_report
from src.safety.guardrails import GuardrailsGateway, PhysicsGuard
from src.safety.citation_tracker import CitationTracker, track_citations
from src.safety.structured_output import (
    XGrammarCompiler,
    PythonFSMCompiler,
    compile_fmea_grammar,
    CompiledFMEAGrammar,
)

__all__ = [
    "ConstrainedDecoder",
    "JSONSchemaGenerator",
    "FMEAReportValidator",
    "validate_report",
    "GuardrailsGateway",
    "PhysicsGuard",
    "CitationTracker",
    "track_citations",
    "XGrammarCompiler",
    "PythonFSMCompiler",
    "compile_fmea_grammar",
    "CompiledFMEAGrammar",
]
