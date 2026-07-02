"""Document loader: MarkItDown-based heterogeneous document ingestion.

Handles Excel (.xlsx), Word (.docx), and PDF documents that make up
the industrial FMEA knowledge base.  Converts everything to Markdown
for downstream semantic rewriting and chunking.

All file paths and content are fictitious.
"""

from __future__ import annotations

from typing import Optional


class DocumentLoader:
    """Load and convert heterogeneous industrial documents via MarkItDown.

    Usage::

        loader = DocumentLoader()
        md_text = loader.convert("fmea_matrix_2025.xlsx")
    """

    def __init__(self) -> None:
        self._md = self._init_markitdown()

    @staticmethod
    def _init_markitdown() -> object:
        """Initialise MarkItDown converter (lazy import for edge compatibility)."""
        try:
            from markitdown import MarkItDown
            return MarkItDown()
        except ImportError:
            return None  # graceful degradation on edge IPC without the library

    def convert(self, file_path: str) -> str:
        """Convert a document to Markdown text.

        Args:
            file_path: path to an .xlsx, .docx, or .pdf file.

        Returns:
            Markdown text content.

        Raises:
            RuntimeError: if markitdown is not installed.
        """
        if self._md is None:
            raise RuntimeError(
                "markitdown is not installed. Install with: pip install markitdown"
            )
        result = self._md.convert(file_path)
        return result.text_content

    def convert_batch(self, file_paths: list[str]) -> dict[str, str]:
        """Convert multiple documents.

        Args:
            file_paths: list of file paths.

        Returns:
            Dict mapping ``file_path → markdown_text``.
        """
        return {fp: self.convert(fp) for fp in file_paths}


def load_document(file_path: str) -> str:
    """Convenience: convert one document to Markdown.

    Args:
        file_path: path to the document.

    Returns:
        Markdown text.
    """
    return DocumentLoader().convert(file_path)
