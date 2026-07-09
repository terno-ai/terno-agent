"""Workspace memory tools: location, org-admin gate, traversal, read/list."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.wiki.tools import (
    MemoryEditTool,
    MemoryListTool,
    MemoryReadTool,
    MemorySearchTool,
    MemoryWriteTool,
)


@pytest.fixture
def user_root(tmp_path: Path) -> Path:
    return tmp_path / "user_workspace" / "memory"


@pytest.fixture
def org_root(tmp_path: Path) -> Path:
    return tmp_path / "org_workspace" / "memory"


def _write(tool: MemoryWriteTool, **over):
    args = dict(
        memory_id="active_user",
        title="Active user",
        type="metric",
        scope="datasource:1",
        datasource_name="sales_db",
        body="An active user has status = 1.",
    )
    args.update(over)
    return json.loads(tool.run(**args))


def test_private_write_lands_directly_in_memory_folder(user_root: Path):
    out = _write(MemoryWriteTool(user_root))
    written = Path(out["path"])
    assert out["shared"] is False
    # A flat file directly under the memory folder — no subdir, no `.terno`.
    assert written == user_root / "active_user.md"
    assert written.exists()
    assert ".terno" not in written.parts
    # Flat: the file is a single path segment inside the memory folder.
    assert written.relative_to(user_root).parts == ("active_user.md",)


def test_nested_memory_id_is_flattened_never_creates_subdir(user_root: Path):
    # A '/'-bearing id must NOT create a subdirectory — it is folded to a flat
    # file name so the memory folder always stays flat.
    out = _write(MemoryWriteTool(user_root), memory_id="metrics/active_user")
    written = Path(out["path"])
    assert out["memory_id"] == "metrics-active_user"
    assert written == user_root / "metrics-active_user.md"
    assert written.relative_to(user_root).parts == ("metrics-active_user.md",)
    assert not (user_root / "metrics").exists()


def test_write_then_read_roundtrip_and_index(user_root: Path):
    _write(MemoryWriteTool(user_root))
    doc = MemoryReadTool(user_root).run(memory_id="active_user")
    assert "type: metric" in doc
    assert "status = 1" in doc
    # The single bundle index sits at the memory-folder root.
    assert (user_root / "index.md").exists()


def test_shared_write_denied_for_non_admin(user_root: Path, org_root: Path):
    tool = MemoryWriteTool(user_root, org_root=org_root, is_org_admin=False)
    with pytest.raises(ToolError, match="org admin"):
        _write(tool, shared=True)
    assert not org_root.exists()  # nothing written to the org folder


def test_shared_write_allowed_for_admin_lands_in_org_folder(
    user_root: Path, org_root: Path
):
    tool = MemoryWriteTool(user_root, org_root=org_root, is_org_admin=True)
    out = _write(tool, shared=True)
    written = Path(out["path"])
    assert out["shared"] is True
    assert written == org_root / "active_user.md"
    assert not (user_root / "active_user.md").exists()  # not in the user folder


def test_shared_write_without_org_folder_errors(user_root: Path):
    tool = MemoryWriteTool(user_root, org_root=None, is_org_admin=True)
    with pytest.raises(ToolError, match="No organisation memory"):
        _write(tool, shared=True)


def test_edit_shared_denied_for_non_admin(user_root: Path, org_root: Path):
    _write(MemoryWriteTool(user_root, org_root=org_root, is_org_admin=True), shared=True)
    non_admin = MemoryEditTool(user_root, org_root=org_root, is_org_admin=False)
    with pytest.raises(ToolError, match="org admin"):
        non_admin.run(
            memory_id="active_user", shared=True, append="tampered"
        )


def test_read_spans_user_then_org(user_root: Path, org_root: Path):
    _write(MemoryWriteTool(user_root, org_root=org_root, is_org_admin=True), shared=True)
    # A reader with both roots finds the org-shared fact even with nothing
    # private.
    doc = MemoryReadTool(user_root, org_root=org_root).run(
        memory_id="active_user"
    )
    assert "Active user" in doc


def test_list_merges_user_and_org(user_root: Path, org_root: Path):
    _write(MemoryWriteTool(user_root), memory_id="output_prefs", type="user",
           scope="global", title="Output prefs", datasource_name="")
    _write(
        MemoryWriteTool(user_root, org_root=org_root, is_org_admin=True),
        shared=True,
    )
    rows = json.loads(MemoryListTool(user_root, org_root=org_root).run())
    shared_flags = {r["memory_id"]: r["shared"] for r in rows}
    assert shared_flags["output_prefs"] is False
    assert shared_flags["active_user"] is True


def test_search_reports_scope_and_shared(user_root: Path, org_root: Path):
    _write(
        MemoryWriteTool(user_root, org_root=org_root, is_org_admin=True),
        shared=True,
    )
    hits = json.loads(
        MemorySearchTool(user_root, org_root=org_root).run(query="active user")
    )
    assert hits and hits[0]["shared"] is True
    assert hits[0]["memory_id"] == "active_user"


def test_write_requires_core_fields(user_root: Path):
    with pytest.raises(ToolError):
        MemoryWriteTool(user_root).run(memory_id="c", title="t")


@pytest.mark.parametrize("bad", ["../escape", "/etc/passwd", "a/../../b"])
def test_traversal_ids_are_rejected(user_root: Path, bad: str):
    with pytest.raises(ToolError, match="unsafe"):
        _write(MemoryWriteTool(user_root), memory_id=bad)
    with pytest.raises(ToolError, match="unsafe"):
        MemoryReadTool(user_root).run(memory_id=bad)


def test_edit_append_preserves_body(user_root: Path):
    _write(MemoryWriteTool(user_root))
    MemoryEditTool(user_root).run(
        memory_id="active_user",
        append="## Notes\nConfirmed by query.",
    )
    doc = MemoryReadTool(user_root).run(memory_id="active_user")
    assert "status = 1" in doc  # original body kept
    assert "Confirmed by query." in doc  # appended
