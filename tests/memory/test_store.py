"""MemoryStore: save/read/list/delete + frontmatter format."""

from __future__ import annotations

from pathlib import Path

from terno_agent.memory.paths import memory_dir
from terno_agent.memory.store import INDEX_FILENAME, MemoryStore
from terno_agent.memory.types import MemoryEntry, MemoryType


def _make_store(workdir: Path, embedder) -> MemoryStore:
    return MemoryStore(workdir=workdir, embedder=embedder)


def test_save_roundtrip_user_type(isolated_memory_dirs: Path, stub_embedder) -> None:
    workdir = isolated_memory_dirs / "proj"
    workdir.mkdir()
    store = _make_store(workdir, stub_embedder)

    entry = MemoryEntry(
        name="user-role",
        description="data engineer focused on the Snowflake pipeline",
        type=MemoryType.USER,
        body="User is a senior data engineer; prefers Python 3.12.",
    )
    path = store.save(entry)

    assert path == memory_dir(workdir) / "user-role.md"
    text = path.read_text(encoding="utf-8")
    assert "name: user-role" in text
    assert "type: user" in text
    assert "Snowflake" in text

    fetched = store.read("user-role")
    assert fetched is not None
    assert fetched.type is MemoryType.USER
    assert fetched.description == entry.description
    assert "Python 3.12" in fetched.body


def test_save_roundtrip_project_type(
    isolated_memory_dirs: Path, stub_embedder
) -> None:
    workdir = isolated_memory_dirs / "proj"
    workdir.mkdir()
    store = _make_store(workdir, stub_embedder)

    entry = MemoryEntry(
        name="project-auth-rewrite",
        description="auth rewrite driven by legal compliance",
        type=MemoryType.PROJECT,
        body=(
            "Legal flagged session token storage.\n"
            "**Why:** GDPR.\n"
            "**How to apply:** prefer compliance."
        ),
    )
    path = store.save(entry)

    # Same single dir as user-type memories.
    assert path == memory_dir(workdir) / "project-auth-rewrite.md"
    assert path.exists()


def test_save_insight_type(isolated_memory_dirs: Path, stub_embedder) -> None:
    workdir = isolated_memory_dirs / "proj"
    workdir.mkdir()
    store = _make_store(workdir, stub_embedder)

    entry = MemoryEntry(
        name="prod-database-host",
        description="Prod DB hostname",
        type=MemoryType.INSIGHT,
        body="db.terno-prod.us-east-1.rds.amazonaws.com",
    )
    path = store.save(entry)
    assert path == memory_dir(workdir) / "prod-database-host.md"
    fetched = store.read("prod-database-host")
    assert fetched is not None
    assert fetched.type is MemoryType.INSIGHT


def test_index_file_is_rewritten(isolated_memory_dirs: Path, stub_embedder) -> None:
    workdir = isolated_memory_dirs / "proj"
    workdir.mkdir()
    store = _make_store(workdir, stub_embedder)

    store.save(
        MemoryEntry(
            name="user-role",
            description="data engineer",
            type=MemoryType.USER,
            body="b",
        )
    )
    store.save(
        MemoryEntry(
            name="feedback-testing",
            description="hit real DB in tests",
            type=MemoryType.FEEDBACK,
            body="b",
        )
    )

    index = (memory_dir(workdir) / INDEX_FILENAME).read_text(encoding="utf-8")
    assert "user-role" in index
    assert "feedback-testing" in index


def test_delete_removes_file_and_vector(
    isolated_memory_dirs: Path, stub_embedder
) -> None:
    workdir = isolated_memory_dirs / "proj"
    workdir.mkdir()
    store = _make_store(workdir, stub_embedder)

    store.save(
        MemoryEntry(
            name="ephemeral",
            description="d",
            type=MemoryType.USER,
            body="b",
        )
    )
    assert store.read("ephemeral") is not None
    assert store.delete("ephemeral") is True
    assert store.read("ephemeral") is None
    assert store.delete("ephemeral") is False  # already gone


def test_list_all_filters_by_type(isolated_memory_dirs: Path, stub_embedder) -> None:
    workdir = isolated_memory_dirs / "proj"
    workdir.mkdir()
    store = _make_store(workdir, stub_embedder)
    store.save(
        MemoryEntry(name="user-a", description="d", type=MemoryType.USER, body="b")
    )
    store.save(
        MemoryEntry(
            name="feedback-a", description="d", type=MemoryType.FEEDBACK, body="b"
        )
    )
    store.save(
        MemoryEntry(name="project-a", description="d", type=MemoryType.PROJECT, body="b")
    )
    store.save(
        MemoryEntry(name="insight-a", description="d", type=MemoryType.INSIGHT, body="b")
    )

    assert {e.name for e in store.list_all()} == {
        "user-a",
        "feedback-a",
        "project-a",
        "insight-a",
    }
    assert [e.name for e in store.list_all(MemoryType.USER)] == ["user-a"]
    assert [e.name for e in store.list_all(MemoryType.PROJECT)] == ["project-a"]
    assert [e.name for e in store.list_all(MemoryType.INSIGHT)] == ["insight-a"]


def test_save_slugifies_invalid_names(
    isolated_memory_dirs: Path, stub_embedder
) -> None:
    workdir = isolated_memory_dirs / "proj"
    workdir.mkdir()
    store = _make_store(workdir, stub_embedder)
    path = store.save(
        MemoryEntry(
            name="User Role!! v2",
            description="d",
            type=MemoryType.USER,
            body="b",
        )
    )
    assert path.name == "user-role-v2.md"


def test_save_updates_existing_entry(
    isolated_memory_dirs: Path, stub_embedder
) -> None:
    workdir = isolated_memory_dirs / "proj"
    workdir.mkdir()
    store = _make_store(workdir, stub_embedder)
    store.save(
        MemoryEntry(name="x", description="v1", type=MemoryType.USER, body="body1")
    )
    store.save(
        MemoryEntry(name="x", description="v2", type=MemoryType.USER, body="body2")
    )
    entry = store.read("x")
    assert entry is not None
    assert entry.description == "v2"
    assert entry.body.strip() == "body2"


def test_search_uses_embedder(isolated_memory_dirs: Path, stub_embedder) -> None:
    workdir = isolated_memory_dirs / "proj"
    workdir.mkdir()
    store = _make_store(workdir, stub_embedder)
    store.save(
        MemoryEntry(
            name="user-role",
            description="data engineer focused on Snowflake",
            type=MemoryType.USER,
            body="User works on the Snowflake pipeline.",
        )
    )
    store.save(
        MemoryEntry(
            name="reference-grafana",
            description="oncall dashboard URL",
            type=MemoryType.REFERENCE,
            body="grafana.internal/d/api-latency",
        )
    )

    # Stub embedder is deterministic — querying with the exact embedding text
    # returns the matching entry as top hit.
    hits = store.search(
        "user-role\ndata engineer focused on Snowflake\n\nUser works on the Snowflake pipeline."
    )
    assert hits
    assert hits[0].key == "user-role"
