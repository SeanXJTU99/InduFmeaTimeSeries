"""RAG subpackage: document loading, semantic rewriting, chunking, embedding, hybrid search, reranking, metadata filtering."""

from src.rag.document_loader import DocumentLoader, load_document
from src.rag.semantic_rewriter import SemanticRewriter, rewrite_fmea_rows
from src.rag.chunker import FMEAChunker, chunk_documents
from src.rag.embedder import Embedder, BGEEncoder
from src.rag.hybrid_search import HybridSearcher, hybrid_search
from src.rag.reranker import Reranker, CrossEncoderReranker
from src.rag.metadata_filter import MetadataFilter, filter_by_tag

__all__ = [
    "DocumentLoader",
    "load_document",
    "SemanticRewriter",
    "rewrite_fmea_rows",
    "FMEAChunker",
    "chunk_documents",
    "Embedder",
    "BGEEncoder",
    "HybridSearcher",
    "hybrid_search",
    "Reranker",
    "CrossEncoderReranker",
    "MetadataFilter",
    "filter_by_tag",
]
