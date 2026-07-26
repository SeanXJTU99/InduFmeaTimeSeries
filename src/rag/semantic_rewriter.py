"""Semantic rewriter: row-level declarative rewriting of FMEA tables.

FMEA Excel tables lose column-header context when naively chunked.
This module rewrites each FMEA row into a self-contained declarative
sentence that preserves the full causal chain::

    "In [System], tag [Tag] exhibits [Failure Mode].  The consequence
    is [Effect] (Severity S).  The root cause is [Cause].  Current
    controls: [Controls].  Detection score D."

This rewritten form dramatically improves embedding recall.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


# Fictitious column name mappings — adapt to your actual FMEA schema.
FMEA_TEMPLATE = (
    "In system/equipment [{system}], monitoring tag [{tag}], "
    "the potential failure mode is: [{failure_mode}]. "
    "The failure effect (consequence) is: [{effect}], "
    "with severity score S={severity}. "
    "The potential root cause is: [{cause}]. "
    "Current controls or time-series anomaly signature: [{controls}]. "
    "Detection score D={detection}. "
    "Overall RPN = {rpn}."
)


class SemanticRewriter:
    """Rewrite FMEA table rows into declarative text chunks.

    Usage::

        rewriter = SemanticRewriter()
        chunks = rewriter.rewrite(markdown_table_text)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rewrite(self, markdown_text: str) -> List[Dict[str, Any]]:
        """Parse a Markdown table and return declarative chunks.

        Args:
            markdown_text: Markdown content, potentially containing one or
                more pipe-delimited tables.

        Returns:
            List of dicts, each with ``page_content`` (the rewritten
            declarative sentence) and ``metadata`` (tag, system, severity,
            occurrence, detection, rpn).
        """
        chunks: List[Dict[str, Any]] = []
        lines = markdown_text.split("\n")
        headers: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            # Detect header row by FMEA keywords
            if not headers and self._is_header(cells):
                headers = cells
                continue
            # Skip separator row (e.g. |---|---|)
            if headers and all("---" in c for c in cells if c):
                continue
            if headers and len(cells) >= len(headers):
                row = dict(zip(headers, cells))
                chunk = self._rewrite_row(row)
                chunks.append(chunk)
        return chunks

    def rewrite_from_dicts(
        self, rows: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Rewrite pre-parsed FMEA rows (e.g. from pandas).

        Args:
            rows: list of dicts, each representing one FMEA row.

        Returns:
            List of declarative chunks.
        """
        return [self._rewrite_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _is_header(cells: List[str]) -> bool:
        """Heuristic: does this row look like an FMEA table header?"""
        keywords = {"failure", "mode", "severity", "cause", "effect", "controls", "detection"}
        joined = " ".join(cells).lower()
        return any(kw in joined for kw in keywords)

    @staticmethod
    def _get(row: Dict[str, str], *keys: str, default: str = "N/A") -> str:
        """Fuzzy key lookup — tries multiple possible column names."""
        for k in keys:
            if k in row:
                return row[k]
        # case-insensitive fallback
        for k in keys:
            for rk in row:
                if rk.lower() == k.lower():
                    return row[rk]
        return default

    @staticmethod
    def _safe_int(row: Dict[str, str], *keys: str) -> int:
        val = SemanticRewriter._get(row, *keys, default="0")
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 0

    def _rewrite_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        system = self._get(row, "system", default="Unspecified")
        tag = self._get(row, "tag", default="N/A")
        fm = self._get(row, "failure_mode", default="N/A")
        effect = self._get(row, "effect", default="N/A")
        cause = self._get(row, "cause", default="N/A")
        controls = self._get(row, "controls", default="N/A")
        severity = self._safe_int(row, "severity")
        occurrence = self._safe_int(row, "occurrence")
        detection = self._safe_int(row, "detection")
        rpn = severity * occurrence * detection

        page_content = FMEA_TEMPLATE.format(
            system=system,
            tag=tag,
            failure_mode=fm,
            effect=effect,
            severity=severity,
            cause=cause,
            controls=controls,
            detection=detection,
            rpn=rpn,
        )
        return {
            "page_content": page_content,
            "metadata": {
                "tag": tag,
                "system": system,
                "severity": severity,
                "occurrence": occurrence,
                "detection": detection,
                "rpn": rpn,
            },
        }


def rewrite_fmea_rows(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convenience: rewrite pre-parsed FMEA rows."""
    return SemanticRewriter().rewrite_from_dicts(rows)
