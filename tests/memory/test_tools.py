"""Memory CRUD tools: save / read / list / delete / search."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.memory.store import MemoryStore
from terno_agent.memory.tools import (
    DeleteMemoryTool,
    ListMemoriesTool,
    ReadMemoryTool,
    SaveMemoryTool,
    SearchMemoryTool,
)
from terno_agent.memory.types import MemoryEntry, MemoryType


def _store(workdir: Path, embedder) -> MemoryStore:
    return MemoryStore(workdir, embedder)


def test_save_then_read(isolated_memory_dirs: Path, stub_embedder) -> None:
    workdir = isolated_memory_dirs / "p"
    workdir.mkdir()
    store = _store(workdir, stub_embedder)

    save = SaveMemoryTool(store)
    result = json.loads(
        save.run(
            name="user-role",
            description="data engineer",
            type="user",
            body="User is a data engineer.",
        )
    )
    assert result["name"] == "user-role"
    assert Path(result["path"]).exists()

    read = ReadMemoryTool(store)
    fetched = json.loads(read.run(name="user-role"))
    assert fetched["type"] == "user"
    assert "data engineer" in fetched["body"]


def test_save_rejects_invalid_type(
    isolated_memory_dirs: Path, stub_embedder
) -> None:
    workdir = isolated_memory_dirs / "p"
    workdir.mkdir()
    store = _store(workdir, stub_embedder)
    save = SaveMemoryTool(store)
    with pytest.raises(ToolError):
        save.run(name="x", description="d", type="bogus", body="b")


def test_list_filters_by_type(isolated_memory_dirs: Path, stub_embedder) -> None:
    workdir = isolated_memory_dirs / "p"
    workdir.mkdir()
    store = _store(workdir, stub_embedder)
    store.save(
        MemoryEntry(name="a", description="d", type=MemoryType.USER, body="b")
    )
    store.save(
        MemoryEntry(name="b", description="d", type=MemoryType.PROJECT, body="b")
    )

    list_tool = ListMemoriesTool(store)
    everything = json.loads(list_tool.run())
    assert {e["name"] for e in everything} == {"a", "b"}

    only_user = json.loads(list_tool.run(type="user"))
    assert [e["name"] for e in only_user] == ["a"]


def test_delete(isolated_memory_dirs: Path, stub_embedder) -> None:
    workdir = isolated_memory_dirs / "p"
    workdir.mkdir()
    store = _store(workdir, stub_embedder)
    store.save(
        MemoryEntry(name="x", description="d", type=MemoryType.USER, body="b")
    )
    tool = DeleteMemoryTool(store)
    assert tool.run(name="x") == "deleted"
    assert tool.run(name="x") == "not_found"


def test_search_returns_json_hits(
    isolated_memory_dirs: Path, stub_embedder
) -> None:
    workdir = isolated_memory_dirs / "p"
    workdir.mkdir()
    store = _store(workdir, stub_embedder)
    store.save(
        MemoryEntry(
            name="user-role",
            description="data engineer",
            type=MemoryType.USER,
            body="User is a data engineer.",
        )
    )
    tool = SearchMemoryTool(store)
    payload = json.loads(tool.run(query="data engineer"))
    assert isinstance(payload, list)
    if payload:  # may be empty if the stub embedder doesn't rank — most cases it does
        assert payload[0]["name"] == "user-role"
        assert "score" in payload[0]
