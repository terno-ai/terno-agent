"""Milvus-backed vector store.

A thin adapter over ``pymilvus``'s high-level ``MilvusClient`` that exposes
the same surface as :class:`~terno_agent.rag.vector_store.FileVectorStore`
(see the :class:`~terno_agent.rag.vector_store.VectorStore` protocol). The
``uri`` may point at a Milvus server (``http://host:19530``) or a local file
for Milvus Lite (``./milvus.db``).

The collection is created on first use with a fixed embedding dimension and
a COSINE metric, so search distances are similarity scores (higher is more
relevant) — matching ``FileVectorStore``'s cosine ranking.
"""

from __future__ import annotations

import json
from threading import Lock
from typing import Any

from terno_agent.core.exceptions import ConfigError, TernoError
from terno_agent.rag.vector_store import Hit

# Milvus VARCHAR fields need an explicit max length; memory keys are short
# slugs and bodies are summaries, so generous caps are plenty.
_KEY_MAX_LEN = 512
_TEXT_MAX_LEN = 65535
# Upper bound for full-collection scans (keys / len). Memory volumes are
# tiny; this stays well under Milvus's query window.
_SCAN_LIMIT = 16384


class MilvusError(TernoError):
    """Raised when a Milvus operation fails."""


class MilvusVectorStore:
    """Vector store backed by a Milvus collection.

    Thread-safe: a single lock serializes writes and the read-modify-write
    used by :meth:`delete`. Reads (``query``) are naturally concurrent on the
    Milvus side, but we keep them under the lock for a consistent view.
    """

    def __init__(
        self,
        *,
        uri: str,
        collection: str,
        dimensions: int,
        token: str | None = None,
    ) -> None:
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:
            raise ConfigError(
                "pymilvus package not installed. Install with: "
                "pip install 'terno-agent[milvus]'"
            ) from exc

        self.collection = collection
        self.dimensions = dimensions
        self._lock = Lock()
        try:
            self._client = MilvusClient(uri=uri, token=token or "")
        except Exception as exc:  # connection / auth errors
            raise MilvusError(f"could not connect to Milvus at {uri!r}: {exc}") from exc
        self._ensure_collection(DataType)

    def _ensure_collection(self, data_type: Any) -> None:
        if self._client.has_collection(self.collection):
            return
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            "key", data_type.VARCHAR, is_primary=True, max_length=_KEY_MAX_LEN
        )
        schema.add_field("vector", data_type.FLOAT_VECTOR, dim=self.dimensions)
        schema.add_field("text", data_type.VARCHAR, max_length=_TEXT_MAX_LEN)
        # Metadata is stored as a JSON string so any shape round-trips
        # regardless of the Milvus server's JSON support.
        schema.add_field("metadata", data_type.VARCHAR, max_length=_TEXT_MAX_LEN)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
        )
        try:
            self._client.create_collection(
                self.collection, schema=schema, index_params=index_params
            )
        except Exception as exc:
            raise MilvusError(f"could not create collection {self.collection!r}: {exc}") from exc

    # ----- mutations --------------------------------------------------- #

    def upsert(
        self,
        key: str,
        text: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "key": key,
            "vector": list(vector),
            "text": text,
            "metadata": json.dumps(metadata or {}),
        }
        with self._lock:
            try:
                self._client.upsert(self.collection, data=[row])
            except Exception as exc:
                raise MilvusError(f"upsert failed for {key!r}: {exc}") from exc

    def delete(self, key: str) -> bool:
        with self._lock:
            if not self._exists(key):
                return False
            try:
                self._client.delete(self.collection, ids=[key])
            except Exception as exc:
                raise MilvusError(f"delete failed for {key!r}: {exc}") from exc
            return True

    # ----- queries ----------------------------------------------------- #

    def query(self, vector: list[float], k: int = 5) -> list[Hit]:
        if k <= 0 or not vector:
            return []
        with self._lock:
            try:
                results = self._client.search(
                    self.collection,
                    data=[list(vector)],
                    limit=k,
                    output_fields=["key", "text", "metadata"],
                )
            except Exception as exc:
                raise MilvusError(f"search failed: {exc}") from exc
        hits: list[Hit] = []
        for match in results[0] if results else []:
            entity = match.get("entity", {})
            hits.append(
                Hit(
                    key=entity.get("key", match.get("id", "")),
                    text=entity.get("text", ""),
                    score=float(match.get("distance", 0.0)),
                    metadata=_load_metadata(entity.get("metadata")),
                )
            )
        return hits

    def keys(self) -> list[str]:
        with self._lock:
            rows = self._scan(["key"])
        return [r["key"] for r in rows]

    def __len__(self) -> int:
        with self._lock:
            return len(self._scan(["key"]))

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return self._exists(key)

    # ----- helpers ----------------------------------------------------- #

    def _exists(self, key: str) -> bool:
        rows = self._client.get(self.collection, ids=[key], output_fields=["key"])
        return bool(rows)

    def _scan(self, output_fields: list[str]) -> list[dict[str, Any]]:
        return list(
            self._client.query(
                self.collection,
                filter="",
                output_fields=output_fields,
                limit=_SCAN_LIMIT,
            )
        )


def _load_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw:
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass
    return {}


__all__ = ["MilvusError", "MilvusVectorStore"]
