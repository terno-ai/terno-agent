"""Generic RAG primitives: embeddings, vector store, retriever.

This package is independent of `memory/`; it is meant to be reusable
(e.g. by the knowledge-extraction pipeline).
"""

from terno_agent.rag.embeddings import (
    EmbeddingClient,
    OpenAIEmbeddingClient,
    create_embedding_client,
)
from terno_agent.rag.milvus_store import MilvusVectorStore
from terno_agent.rag.retriever import Retriever
from terno_agent.rag.vector_store import (
    FileVectorStore,
    Hit,
    VectorStore,
    create_vector_store,
)

__all__ = [
    "EmbeddingClient",
    "FileVectorStore",
    "Hit",
    "MilvusVectorStore",
    "OpenAIEmbeddingClient",
    "Retriever",
    "VectorStore",
    "create_embedding_client",
    "create_vector_store",
]
