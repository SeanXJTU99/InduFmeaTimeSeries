"""FMEA-aware chunking strategy.

Standard character-window or recursive-split chunking destroys FMEA
table context (column headers get separated from data rows).  This
module provides chunkers that respect FMEA structure:

1. **Row-based chunker** — one chunk per FMEA entry (the declarative
   sentence produced by :class:`SemanticRewriter`).
2. **Heading-based chunker** — splits narrative documents (SOPs,
   maintenance manuals) on Markdown headings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChunkingConfig:
    """Chunking parameters."""

    # Row-based chunker
    max_chars_per_chunk: int = 1500  # soft cap — chunks exceeding this are split
    chunk_overlap_chars: int = 100  # overlap for split chunks

    # Heading-based chunker
    heading_levels: tuple[int, ...] = (1, 2, 3)  # split on #, ##, ###


class FMEAChunker:
    """Chunk FMEA documents with structure-aware strategies.

    Usage::

        chunker = FMEAChunker()
        chunks = chunker.chunk_declarative(rewritten_rows)
        chunks += chunker.chunk_narrative(markdown_text)
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self._cfg = config or ChunkingConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_declarative(
        self, rewritten_rows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Chunk pre-rewritten declarative FMEA rows.

        Each row is already a self-contained sentence; we only split
        those that exceed ``max_chars_per_chunk``.

        Args:
            rewritten_rows: output of :meth:`SemanticRewriter.rewrite`.

        Returns:
            List of chunk dicts with ``page_content`` and ``metadata``.
        """
        chunks: List[Dict[str, Any]] = []
        for row in rewritten_rows:
            text = row["page_content"]
            meta = row.get("metadata", {})
            if len(text) <= self._cfg.max_chars_per_chunk:
                chunks.append({"page_content": text, "metadata": meta})
            else:
                # split long text with overlap
                for i in range(0, len(text), self._cfg.max_chars_per_chunk - self._cfg.chunk_overlap_chars):
                    sub = text[i : i + self._cfg.max_chars_per_chunk]
                    if sub:
                        chunks.append({"page_content": sub, "metadata": meta})
        return chunks

    def chunk_narrative(
        self, markdown_text: str, source: str = ""
    ) -> List[Dict[str, Any]]:
        """Split narrative documents on Markdown headings.

        Args:
            markdown_text: raw markdown of a manual, SOP, or report.
            source: optional source identifier for metadata.

        Returns:
            List of chunk dicts.
        """
        chunks: List[Dict[str, Any]] = []
        lines = markdown_text.split("\n")
        current_lines: List[str] = []
        current_heading = ""
        for line in lines:
            # detect heading
            level = self._heading_level(line)
            if level is not None and level in self._cfg.heading_levels:
                # flush previous chunk
                if current_lines:
                    chunks.append(self._make_chunk(current_lines, current_heading, source))
                current_heading = line.strip().lstrip("#").strip()
                current_lines = [line]
            else:
                current_lines.append(line)
        # flush last chunk
        if current_lines:
            chunks.append(self._make_chunk(current_lines, current_heading, source))
        return chunks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _heading_level(line: str) -> Optional[int]:
        stripped = line.strip()
        if stripped.startswith("#"):
            return len(stripped) - len(stripped.lstrip("#"))
        return None

    @staticmethod
    def _make_chunk(lines: List[str], heading: str, source: str) -> Dict[str, Any]:
        text = "\n".join(lines).strip()
        return {
            "page_content": text,
            "metadata": {
                "heading": heading,
                "source": source,
                "chunk_type": "narrative",
            },
        }


def chunk_documents(
    declarative_rows: List[Dict[str, Any]] | None = None,
    narrative_texts: List[tuple[str, str]] | None = None,
) -> List[Dict[str, Any]]:
    """Convenience: chunk both declarative rows and narrative documents.

    Args:
        declarative_rows: output from SemanticRewriter (optional).
        narrative_texts: list of (markdown_text, source_name) pairs (optional).

    Returns:
        Combined list of chunk dicts.
    """
    chunker = FMEAChunker()
    chunks: List[Dict[str, Any]] = []
    if declarative_rows:
        chunks.extend(chunker.chunk_declarative(declarative_rows))
    if narrative_texts:
        for md, src in narrative_texts:
            chunks.extend(chunker.chunk_narrative(md, source=src))
    return chunks
