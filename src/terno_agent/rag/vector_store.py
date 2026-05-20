"""File-backed vector store.

A single ``.vectors.jsonl`` file per scope holds one JSON object per
entry: ``{"key": ..., "text": ..., "vector": [...], "metadata": {...}}``.
The whole file is loaded into memory on construction; writes are atomic
via tempfile + rename. Designed for the memory module's volumes (tens to
low hundreds of entries) — no sqlite-vec or FAISS required.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass(slots=True)
class VectorRecord:
    key: str
    text: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Hit:
    key: str
    text: str
    score: float
    metadata: dict[str, Any]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0.0:
        return 0.0
    return dot / denom


class FileVectorStore:
    """Persistent in-memory vector list backed by JSONL.

    Operations are thread-safe. Every mutating call flushes to disk so
    concurrent readers (e.g. retrieval at the next turn) see fresh data.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._records: dict[str, VectorRecord] = {}
        self._lock = Lock()
        self._load()

    # ----- I/O --------------------------------------------------------- #

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    rec = VectorRecord(
                        key=obj["key"],
                        text=obj.get("text", ""),
                        vector=list(obj.get("vector") or []),
                        metadata=dict(obj.get("metadata") or {}),
                    )
                    self._records[rec.key] = rec
        except (OSError, json.JSONDecodeError, KeyError):
            # Corrupt file — start fresh; don't crash callers.
            self._records = {}

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp file in same dir, then rename.
        fd, tmp_path = tempfile.mkstemp(
            prefix=".vectors-", suffix=".jsonl.tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for rec in self._records.values():
                    json.dump(
                        {
                            "key": rec.key,
                            "text": rec.text,
                            "vector": rec.vector,
                            "metadata": rec.metadata,
                        },
                        f,
                    )
                    f.write("\n")
            os.replace(tmp_path, self.path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    # ----- mutations --------------------------------------------------- #

    def upsert(
        self,
        key: str,
        text: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._records[key] = VectorRecord(
                key=key,
                text=text,
                vector=list(vector),
                metadata=dict(metadata or {}),
            )
            self.flush()

    def delete(self, key: str) -> bool:
        with self._lock:
            existed = self._records.pop(key, None) is not None
            if existed:
                self.flush()
            return existed

    # ----- queries ----------------------------------------------------- #

    def query(self, vector: list[float], k: int = 5) -> list[Hit]:
        with self._lock:
            scored: list[Hit] = []
            for rec in self._records.values():
                score = _cosine(vector, rec.vector)
                scored.append(
                    Hit(key=rec.key, text=rec.text, score=score, metadata=dict(rec.metadata))
                )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[: max(0, k)]

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, key: str) -> bool:
        return key in self._records

    def keys(self) -> list[str]:
        return list(self._records.keys())


__all__ = ["FileVectorStore", "Hit", "VectorRecord"]
