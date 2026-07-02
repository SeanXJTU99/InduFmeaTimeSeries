"""Metadata filter for tag-based scalar filtering.

In industrial FMEA, filtering results by PLC tag (e.g. "TE-101") or
system (e.g. "Cryogenic Column T-301") is the single most effective
anti-hallucination measure.  This module provides fast, in-memory
metadata matching before or after vector search.

All tag names are fictitious.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class MetadataFilter:
    """Exact-match metadata filter for RAG results.

    Usage::

        mf = MetadataFilter()
        matches = mf.filter(chunks, {"tag": "TE-101", "system": "T-301"})
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        chunks: List[Dict[str, Any]],
        conditions: Dict[str, str],
        mode: str = "all",
    ) -> List[Dict[str, Any]]:
        """Filter chunks by metadata key-value pairs.

        Args:
            chunks: list of chunk dicts with ``metadata``.
            conditions: ``{key: value}`` pairs to match.
            mode: ``'all'`` (AND) or ``'any'`` (OR).

        Returns:
            Filtered list of chunks.
        """
        if not conditions:
            return chunks
        result: List[Dict[str, Any]] = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            if mode == "all":
                if all(str(meta.get(k, "")) == str(v) for k, v in conditions.items()):
                    result.append(chunk)
            else:  # 'any'
                if any(str(meta.get(k, "")) == str(v) for k, v in conditions.items()):
                    result.append(chunk)
        return result

    def get_tags(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Return unique tag values across all chunks."""
        tags: set[str] = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            tag = meta.get("tag", "")
            if tag and tag != "N/A":
                tags.add(str(tag))
        return sorted(tags)

    def get_systems(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Return unique system names across all chunks."""
        systems: set[str] = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            sys = meta.get("system", "")
            if sys and sys != "N/A":
                systems.add(str(sys))
        return sorted(systems)


def filter_by_tag(
    chunks: List[Dict[str, Any]], tag: str
) -> List[Dict[str, Any]]:
    """Convenience: filter chunks by exact tag match.

    Args:
        chunks: list of chunk dicts.
        tag: exact PLC tag to match (e.g. ``'TE-101'``).

    Returns:
        Matching chunks.
    """
    return MetadataFilter().filter(chunks, {"tag": tag})
