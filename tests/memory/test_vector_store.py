"""FileVectorStore: upsert, query, persistence."""

from __future__ import annotations

from pathlib import Path

from terno_agent.rag.vector_store import FileVectorStore


def test_upsert_and_query_basic(tmp_path: Path) -> None:
    store = FileVectorStore(tmp_path / "v.jsonl")
    store.upsert("a", "alpha", [1.0, 0.0, 0.0], {"kind": "x"})
    store.upsert("b", "beta", [0.0, 1.0, 0.0], {"kind": "y"})
    store.upsert("c", "gamma", [0.0, 0.0, 1.0], {"kind": "z"})

    hits = store.query([1.0, 0.0, 0.0], k=2)
    assert [h.key for h in hits] == ["a", "b"]  # b and c tie at 0; insertion order
    assert hits[0].score > hits[1].score
    assert hits[0].metadata["kind"] == "x"


def test_query_ranks_by_cosine(tmp_path: Path) -> None:
    store = FileVectorStore(tmp_path / "v.jsonl")
    store.upsert("close", "t", [1.0, 1.0, 0.0], {})
    store.upsert("far", "t", [0.0, 0.0, 1.0], {})
    hits = store.query([1.0, 1.0, 0.1], k=2)
    assert hits[0].key == "close"
    assert hits[1].key == "far"


def test_persistence_across_reload(tmp_path: Path) -> None:
    path = tmp_path / "v.jsonl"
    store = FileVectorStore(path)
    store.upsert("a", "alpha", [1.0, 0.0])
    store.upsert("b", "beta", [0.0, 1.0])

    reloaded = FileVectorStore(path)
    assert "a" in reloaded
    assert "b" in reloaded
    assert len(reloaded) == 2
    hits = reloaded.query([1.0, 0.0], k=1)
    assert hits[0].key == "a"


def test_delete(tmp_path: Path) -> None:
    store = FileVectorStore(tmp_path / "v.jsonl")
    store.upsert("a", "alpha", [1.0, 0.0])
    store.upsert("b", "beta", [0.0, 1.0])
    assert store.delete("a") is True
    assert store.delete("nope") is False
    assert "a" not in store
    assert len(store) == 1


def test_corrupt_file_starts_fresh(tmp_path: Path) -> None:
    path = tmp_path / "v.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    store = FileVectorStore(path)
    assert len(store) == 0
    # Still usable after corruption.
    store.upsert("a", "alpha", [1.0])
    assert "a" in store
