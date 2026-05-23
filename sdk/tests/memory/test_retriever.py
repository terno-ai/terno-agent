"""MemoryRetriever: formats top-k for system prompt injection."""

from __future__ import annotations

from pathlib import Path

from terno_agent.memory.retriever import MemoryRetriever
from terno_agent.memory.store import MemoryStore
from terno_agent.memory.types import MemoryEntry, MemoryType


def test_fetch_relevant_empty_when_store_is_empty(
    isolated_memory_dirs: Path, stub_embedder
) -> None:
    workdir = isolated_memory_dirs / "p"
    workdir.mkdir()
    store = MemoryStore(workdir, stub_embedder)
    retriever = MemoryRetriever(store=store, k=3)
    assert retriever.fetch_relevant("anything") == ""


def test_fetch_relevant_renders_hits(
    isolated_memory_dirs: Path, stub_embedder
) -> None:
    workdir = isolated_memory_dirs / "p"
    workdir.mkdir()
    store = MemoryStore(workdir, stub_embedder)
    store.save(
        MemoryEntry(
            name="user-role",
            description="data engineer",
            type=MemoryType.USER,
            body="User is a data engineer.",
        )
    )
    retriever = MemoryRetriever(store=store, k=5)
    text = retriever.fetch_relevant("data engineer")
    assert "Relevant memories" in text
    assert "user-role" in text
    assert "data engineer" in text


def test_fetch_relevant_empty_for_blank_query(
    isolated_memory_dirs: Path, stub_embedder
) -> None:
    workdir = isolated_memory_dirs / "p"
    workdir.mkdir()
    store = MemoryStore(workdir, stub_embedder)
    store.save(
        MemoryEntry(name="x", description="d", type=MemoryType.USER, body="b")
    )
    retriever = MemoryRetriever(store=store, k=3)
    assert retriever.fetch_relevant("") == ""
    assert retriever.fetch_relevant("   ") == ""
