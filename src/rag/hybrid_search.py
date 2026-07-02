"""Hybrid search: BM25 keyword retrieval + BGE dense vector retrieval.

Industrial terminology includes precise nouns ("汽蚀", "flooding",
"FV-301") that pure dense retrieval can miss.  BM25 catches exact
tag/part-number matches while BGE captures semantic intent.  Results
are merged via reciprocal rank fusion (RRF).

All tag names and search examples are fictitious.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class HybridSearchConfig:
    """Hybrid search parameters."""

    dense_weight: float = 0.6  # weight of dense (BGE) scores in RRF fusion
    sparse_weight: float = 0.4  # weight of BM25 scores
    top_k_dense: int = 50  # dense candidate pool size
    top_k_sparse: int = 50  # BM25 candidate pool size
    final_top_k: int = 10  # number of results after fusion
    rrf_k: int = 60  # RRF smoothing constant


class BM25Index:
    """Minimal BM25 index for keyword retrieval.

    Uses the rank-bm25 package under the hood.  Tokens are split on
    whitespace; for production use, consider jieba (Chinese) or a
    domain-specific tokeniser.
    """

    def __init__(self) -> None:
        self._index: object | None = None
        self._corpus: List[str] = []
        self._tokenized: List[List[str]] = []

    def index(self, documents: List[Dict[str, Any]]) -> None:
        """Build BM25 index from chunked documents.

        Args:
            documents: list of dicts with ``page_content`` and ``metadata``.
        """
        from rank_bm25 import BM25Okapi
        self._corpus = [d["page_content"] for d in documents]
        self._tokenized = [text.lower().split() for text in self._corpus]
        self._index = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Return top-k (doc_index, score) pairs."""
        if self._index is None:
            return []
        tokenized_query = query.lower().split()
        scores = self._index.get_scores(tokenized_query)
        # Normalise scores to [0, 1]
        max_score = float(np.max(scores)) if len(scores) > 0 else 1.0
        if max_score > 0:
            scores = scores / max_score
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


class HybridSearcher:
    """BM25 + BGE dual-path hybrid retriever with RRF fusion.

    Usage::

        searcher = HybridSearcher(encoder)
        searcher.index(chunks)
        results = searcher.search("TE-101 bearing temperature high")
    """

    def __init__(
        self,
        embedder: "Embedder | None" = None,
        config: HybridSearchConfig | None = None,
    ) -> None:
        from src.rag.embedder import Embedder
        self._embedder = embedder or Embedder()
        self._cfg = config or HybridSearchConfig()
        self._bm25 = BM25Index()
        self._documents: List[Dict[str, Any]] = []
        self._dense_vectors: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents for both dense and sparse retrieval.

        Args:
            documents: list of chunk dicts (``page_content`` + ``metadata``).
        """
        self._documents = documents
        # BM25 index
        self._bm25.index(documents)
        # Dense index
        texts = [d["page_content"] for d in documents]
        self._embedder.load()
        self._dense_vectors = self._embedder.encode(texts, normalize=True)

    def search(
        self,
        query: str,
        metadata_filter: Dict[str, str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Execute hybrid search with optional metadata filter.

        Args:
            query: natural-language search query.
            metadata_filter: optional ``{key: value}`` filter applied
                post-retrieval (e.g. ``{"tag": "TE-101"}``).

        Returns:
            List of result dicts with ``page_content``, ``metadata``,
            ``dense_score``, ``bm25_score``, ``fused_score``.
        """
        # --- dense path ---
        query_vec = self._embedder.encode_query(query)
        dense_hits = self._dense_search(query_vec, self._cfg.top_k_dense)

        # --- sparse path ---
        sparse_hits = self._bm25.search(query, top_k=self._cfg.top_k_sparse)

        # --- RRF fusion ---
        fused = self._rrf_fuse(dense_hits, sparse_hits, metadata_filter)
        return fused[: self._cfg.final_top_k]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dense_search(
        self, query_vec: np.ndarray, top_k: int
    ) -> List[Tuple[int, float]]:
        if self._dense_vectors is None:
            return []
        scores = np.dot(self._dense_vectors, query_vec)  # cosine via inner product
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    def _rrf_fuse(
        self,
        dense_hits: List[Tuple[int, float]],
        sparse_hits: List[Tuple[int, float]],
        metadata_filter: Dict[str, str] | None,
    ) -> List[Dict[str, Any]]:
        """Reciprocal rank fusion."""
        k = self._cfg.rrf_k
        rrf: Dict[int, float] = {}
        dense_score: Dict[int, float] = {}
        bm25_score: Dict[int, float] = {}

        # Dense contributions
        for rank, (doc_id, score) in enumerate(dense_hits):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + self._cfg.dense_weight / (k + rank + 1)
            dense_score[doc_id] = score

        # Sparse contributions
        for rank, (doc_id, score) in enumerate(sparse_hits):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + self._cfg.sparse_weight / (k + rank + 1)
            bm25_score[doc_id] = score

        # Sort by fused score, apply metadata filter
        results: List[Dict[str, Any]] = []
        for doc_id, fused in sorted(rrf.items(), key=lambda x: x[1], reverse=True):
            doc = self._documents[doc_id]
            if metadata_filter and not self._match_filter(doc.get("metadata", {}), metadata_filter):
                continue
            results.append({
                "page_content": doc["page_content"],
                "metadata": doc.get("metadata", {}),
                "dense_score": dense_score.get(doc_id, 0.0),
                "bm25_score": bm25_score.get(doc_id, 0.0),
                "fused_score": fused,
            })
        return results

    @staticmethod
    def _match_filter(meta: Dict[str, Any], filt: Dict[str, str]) -> bool:
        for k, v in filt.items():
            if str(meta.get(k, "")) != str(v):
                return False
        return True


def hybrid_search(
    query: str,
    documents: List[Dict[str, Any]],
    embedder: "Embedder | None" = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Convenience: one-shot hybrid search.

    Args:
        query: search query string.
        documents: pre-chunked document list.
        embedder: optional pre-loaded Embedder.
        top_k: number of results.

    Returns:
        Ranked list of result dicts.
    """
    cfg = HybridSearchConfig(final_top_k=top_k)
    searcher = HybridSearcher(embedder, cfg)
    searcher.index(documents)
    return searcher.search(query)
