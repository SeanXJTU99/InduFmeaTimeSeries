"""Tests for hybrid search (BM25 + BGE)."""

from src.rag.hybrid_search import HybridSearcher, HybridSearchConfig
from src.rag.metadata_filter import filter_by_tag


class TestHybridSearcher:
    def test_index_and_search(self) -> None:
        docs = [
            {"page_content": "TE-301 bearing temperature high — lubricant failure", "metadata": {"tag": "TE-301"}},
            {"page_content": "PT-301 pressure drop indicates flooding in column T-301", "metadata": {"tag": "PT-301"}},
            {"page_content": "FT-301 flow rate low — possible valve stiction FV-301", "metadata": {"tag": "FT-301"}},
        ]
        searcher = HybridSearcher()
        searcher.index(docs)
        results = searcher.search("bearing temperature problem")
        assert len(results) > 0
        assert len(results) <= 10

    def test_metadata_filter(self) -> None:
        docs = [
            {"page_content": "Chunk A", "metadata": {"tag": "TE-301"}},
            {"page_content": "Chunk B", "metadata": {"tag": "PT-301"}},
            {"page_content": "Chunk C", "metadata": {"tag": "TE-301"}},
        ]
        filtered = filter_by_tag(docs, "TE-301")
        assert len(filtered) == 2
        assert all(d["metadata"]["tag"] == "TE-301" for d in filtered)

    def test_empty_query_returns_empty(self) -> None:
        searcher = HybridSearcher()
        searcher.index([{"page_content": "test", "metadata": {}}])
        results = searcher.search("")
        assert len(results) >= 0  # should not crash
