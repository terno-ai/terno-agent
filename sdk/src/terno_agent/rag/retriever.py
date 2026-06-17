"""Generic top-k retriever over an embedding client + vector store(s)."""

from __future__ import annotations

from dataclasses import dataclass

from terno_agent.rag.embeddings import EmbeddingClient
from terno_agent.rag.vector_store import Hit, VectorStore


@dataclass(slots=True)
class Retriever:
    embedder: EmbeddingClient
    stores: list[VectorStore]

    def top_k(self, query: str, k: int = 5) -> list[Hit]:
        if not query.strip() or k <= 0:
            return []
        vectors = self.embedder.embed([query])
        if not vectors:
            return []
        qv = vectors[0]
        merged: list[Hit] = []
        for store in self.stores:
            merged.extend(store.query(qv, k=k))
        merged.sort(key=lambda h: h.score, reverse=True)
        return merged[:k]


__all__ = ["Retriever"]
