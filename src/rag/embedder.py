"""Embedding encoder: BGE-Large wrapper for vectorising FMEA chunks.

Uses BAAI/bge-large-zh-v1.5 (or bge-large-en) for semantic embedding
of Chinese/English industrial text.  The encoder normalises outputs to
unit length for cosine-similarity search via inner product.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


class Embedder:
    """Lightweight wrapper around sentence-transformers / BGE.

    Usage::

        encoder = Embedder(model_name="BAAI/bge-large-en-v1.5")
        vectors = encoder.encode(["chunk 1 text", "chunk 2 text"])
        # vectors.shape == (2, 1024)
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        self._model_name = model_name
        self._model: object | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Lazy-load the sentence-transformers model into memory."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is required. Install with: "
                "pip install sentence-transformers"
            )

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed a list of text chunks.

        Args:
            texts: list of chunk text strings.
            batch_size: encoding batch size.
            show_progress: show tqdm progress bar.
            normalize: L2-normalise output vectors.

        Returns:
            2-D numpy array of shape ``(len(texts), dim)``.
        """
        self.load()
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single search query (BGE adds 'Represent this sentence...' prefix)."""
        # BGE models benefit from a query prefix for asymmetric search
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self.encode([prefixed])[0]

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        self.load()
        return self._model.get_sentence_embedding_dimension()


# Backward-compatible alias
BGEEncoder = Embedder
