"""Shared fixtures for memory tests."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from terno_agent.memory.paths import GLOBAL_ENV_VAR
from terno_agent.rag.embeddings import EmbeddingClient


class StubEmbedder(EmbeddingClient):
    """Deterministic 16-dim embedder.

    Produces a vector by hashing the text — same text always returns the
    same vector. Useful for testing the store/retriever wiring without
    hitting the OpenAI API.
    """

    model = "stub"
    dimensions = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            # Map 16 bytes -> 16 floats in [-1, 1].
            out.append([(b - 128) / 128.0 for b in digest[: self.dimensions]])
        return out


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def isolated_memory_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect the global memory dir into ``tmp_path`` so tests never touch ``~``."""
    global_dir = tmp_path / "global_memory"
    monkeypatch.setenv(GLOBAL_ENV_VAR, str(global_dir))
    yield tmp_path
