"""Generic RAG primitives: embeddings, vector store, retriever.

This package is independent of `memory/`; it is meant to be reusable
(e.g. by the knowledge-extraction pipeline).
"""

from terno_agent.rag.embeddings import (
    EmbeddingClient,
    OpenAIEmbeddingClient,
    create_embedding_client,
)
from terno_agent.rag.retriever import Retriever
from terno_agent.rag.vector_store import FileVectorStore, Hit

__all__ = [
    "EmbeddingClient",
    "FileVectorStore",
    "Hit",
    "OpenAIEmbeddingClient",
    "Retriever",
    "create_embedding_client",
]
