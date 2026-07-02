"""Citation tracker — anti-hallucination layer 4.

Every LLM-generated diagnostic statement must cite its source FMEA
entry (row number or chunk ID).  This module tracks which sources are
referenced and flags uncited claims (potential hallucinations).

In production, this is enforced via the Agent's system prompt:
"Every claim must include a citation.  Claims without citations are
automatically rejected."
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple


# Pattern: [Source: <id>] or [FMEA #<num>] or (Source: <id>)
CITATION_PATTERNS = [
    re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE),
    re.compile(r"\[FMEA\s*#(\d+)\]", re.IGNORECASE),
    re.compile(r"\(Source:\s*([^)]+)\)", re.IGNORECASE),
    re.compile(r"\[Ref:\s*([^\]]+)\]", re.IGNORECASE),
]


class CitationTracker:
    """Track citation coverage in LLM-generated FMEA reports.

    Usage::

        tracker = CitationTracker(source_ids={"fmea_001", "fmea_042"})
        report = tracker.audit(llm_output_text)
        if report.uncited_claims:
            reject(report)
    """

    def __init__(self, source_ids: Set[str] | None = None) -> None:
        """Initialise with the set of valid source identifiers.

        Args:
            source_ids: set of valid chunk/source IDs from the RAG
                documents that were retrieved.
        """
        self._source_ids = source_ids or set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit(self, text: str) -> CitationReport:
        """Audit a generated text for citation coverage.

        Args:
            text: the LLM-generated diagnostic report.

        Returns:
            :class:`CitationReport` with cited sources, uncited claims,
            and invalid citations.
        """
        cited = self._extract_citations(text)
        invalid = cited - self._source_ids
        # Heuristic: sentences that make factual claims (contain numbers,
        # tag names, or failure mode keywords) are "claims".
        claims = self._detect_claims(text)
        uncited_count = max(0, claims - len(cited))
        return CitationReport(
            total_claims=claims,
            cited_sources=cited,
            invalid_citations=invalid,
            uncited_claim_count=uncited_count,
            is_fully_cited=(uncited_count == 0 and len(invalid) == 0),
        )

    def score(self, text: str) -> float:
        """Return a citation quality score in [0, 1].

        1.0 = every claim is backed by a valid source citation.
        """
        report = self.audit(text)
        if report.total_claims == 0:
            return 1.0
        valid = len(report.cited_sources - report.invalid_citations)
        return min(1.0, valid / max(report.total_claims, 1))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_citations(text: str) -> Set[str]:
        found: Set[str] = set()
        for pattern in CITATION_PATTERNS:
            for match in pattern.finditer(text):
                found.add(match.group(1).strip())
        return found

    @staticmethod
    def _detect_claims(text: str) -> int:
        """Count sentences that appear to be factual claims.

        Heuristic: a sentence containing a numeric value, a tag-like
        identifier (e.g. TE-301), or a failure mode keyword.
        """
        sentences = re.split(r"[.!?\n]+", text)
        count = 0
        tag_pattern = re.compile(r"[A-Z]{2,}-\d{3}")
        number_pattern = re.compile(r"\d+\.?\d*")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            has_tag = bool(tag_pattern.search(sent))
            has_number = bool(number_pattern.search(sent))
            if has_tag or has_number:
                count += 1
        return count


class CitationReport:
    """Result of a citation audit."""

    def __init__(
        self,
        total_claims: int,
        cited_sources: Set[str],
        invalid_citations: Set[str],
        uncited_claim_count: int,
        is_fully_cited: bool,
    ) -> None:
        self.total_claims = total_claims
        self.cited_sources = cited_sources
        self.invalid_citations = invalid_citations
        self.uncited_claim_count = uncited_claim_count
        self.is_fully_cited = is_fully_cited

    def __repr__(self) -> str:
        return (
            f"CitationReport(claims={self.total_claims}, cited={len(self.cited_sources)}, "
            f"invalid={len(self.invalid_citations)}, uncited={self.uncited_claim_count}, "
            f"fully_cited={self.is_fully_cited})"
        )


def track_citations(text: str, source_ids: Set[str] | None = None) -> CitationReport:
    """Convenience: audit citations in one call."""
    return CitationTracker(source_ids).audit(text)
