"""Cross-encoder reranker for improving hybrid search precision.

After BM25+BGE hybrid retrieval returns candidate chunks, a lightweight
cross-encoder reranks them by jointly encoding (query, chunk) pairs.
This catches semantic mismatches that separate-encoding (bi-encoder)
models miss.

For edge deployment, the reranker defaults to a small distilled model
(e.g. ms-marco-MiniLM-L-6-v2) that runs on CPU with acceptable latency.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RerankerConfig:
    """Reranker parameters."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_input: int = 20  # candidates to rerank (reranker is more expensive)
    top_k_output: int = 5  # results after reranking
    batch_size: int = 16


class CrossEncoderReranker:
    """Cross-encoder based reranker for FMEA search results.

    Usage::

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, candidate_results)
    """

    def __init__(self, config: RerankerConfig | None = None) -> None:
        self._cfg = config or RerankerConfig()
        self._model: object | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._cfg.model_name)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is required for the cross-encoder. "
                "Install with: pip install sentence-transformers"
            )

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank candidate chunks using the cross-encoder.

        Args:
            query: user/agent query string.
            candidates: list of result dicts from hybrid search.

        Returns:
            Reranked list (same dicts, with ``rerank_score`` added).
        """
        if not candidates:
            return []

        self.load()
        # Limit candidates to rerank
        pool = candidates[: self._cfg.top_k_input]
        pairs = [(query, c["page_content"]) for c in pool]
        scores = self._model.predict(
            pairs, batch_size=self._cfg.batch_size, show_progress_bar=False
        )
        scores = np.asarray(scores, dtype=np.float32)

        # Attach rerank scores and sort
        for i, c in enumerate(pool):
            c["rerank_score"] = float(scores[i]) if i < len(scores) else 0.0

        pool.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        return pool[: self._cfg.top_k_output]


# Backward-compatible alias
Reranker = CrossEncoderReranker
