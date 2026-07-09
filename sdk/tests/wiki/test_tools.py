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
    return tmp_path / "users" / "acme" / "ada" / "memory"


@pytest.fixture
def org_root(tmp_path: Path) -> Path:
    return tmp_path / "orgs" / "acme" / "memory"


def _write(tool: MemoryWriteTool, **over):
    args = dict(
        datasource="sales_db",
        memory_id="metrics/active_user",
        title="Active user",
        type="metric",
        scope="datasource:1",
        datasource_name="sales_db",
        body="An active user has status = 1.",
    )
    args.update(over)
    return json.loads(tool.run(**args))


def test_private_write_lands_in_memory_folder_not_terno(user_root: Path):
    out = _write(MemoryWriteTool(user_root))
    written = Path(out["path"])
    assert out["shared"] is False
    # Directly under the workspace memory folder, never under `.terno`.
    assert written == user_root / "sales_db" / "metrics" / "active_user.md"
    assert written.exists()
    assert ".terno" not in written.parts


def test_write_then_read_roundtrip_and_index(user_root: Path):
    _write(MemoryWriteTool(user_root))
    doc = MemoryReadTool(user_root).run(
        datasource="sales_db", memory_id="metrics/active_user"
    )
    assert "type: metric" in doc
    assert "status = 1" in doc
    # The bundle index was regenerated so the fact is discoverable.
    assert (user_root / "sales_db" / "index.md").exists()


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
    assert written == org_root / "sales_db" / "metrics" / "active_user.md"
    assert not (user_root / "sales_db").exists()  # not in the user folder


def test_shared_write_without_org_folder_errors(user_root: Path):
    tool = MemoryWriteTool(user_root, org_root=None, is_org_admin=True)
    with pytest.raises(ToolError, match="No organisation memory"):
        _write(tool, shared=True)


def test_edit_shared_denied_for_non_admin(user_root: Path, org_root: Path):
    # Seed a shared fact as admin, then a non-admin tool must not edit it.
    _write(MemoryWriteTool(user_root, org_root=org_root, is_org_admin=True), shared=True)
    non_admin = MemoryEditTool(user_root, org_root=org_root, is_org_admin=False)
    with pytest.raises(ToolError, match="org admin"):
        non_admin.run(
            datasource="sales_db",
            memory_id="metrics/active_user",
            shared=True,
            append="tampered",
        )


def test_read_spans_user_then_org(user_root: Path, org_root: Path):
    _write(MemoryWriteTool(user_root, org_root=org_root, is_org_admin=True), shared=True)
    # A reader with both roots finds the org-shared fact even with nothing
    # private.
    doc = MemoryReadTool(user_root, org_root=org_root).run(
        datasource="sales_db", memory_id="metrics/active_user"
    )
    assert "Active user" in doc


def test_list_merges_user_and_org_bundles(user_root: Path, org_root: Path):
    _write(MemoryWriteTool(user_root), memory_id="prefs/output", type="user",
           scope="global", title="Output prefs", datasource_name="")
    _write(
        MemoryWriteTool(user_root, org_root=org_root, is_org_admin=True),
        shared=True,
    )
    names = json.loads(MemoryListTool(user_root, org_root=org_root).run())
    assert names == ["sales_db"]  # same bundle name in both roots, de-duped
    rows = json.loads(
        MemoryListTool(user_root, org_root=org_root).run(datasource="sales_db")
    )
    shared_flags = {r["memory_id"]: r["shared"] for r in rows}
    assert shared_flags["prefs/output"] is False
    assert shared_flags["metrics/active_user"] is True


def test_search_reports_scope_and_shared(user_root: Path, org_root: Path):
    _write(
        MemoryWriteTool(user_root, org_root=org_root, is_org_admin=True),
        shared=True,
    )
    hits = json.loads(
        MemorySearchTool(user_root, org_root=org_root).run(query="active user")
    )
    assert hits and hits[0]["shared"] is True
    assert hits[0]["memory_id"] == "metrics/active_user"


def test_write_requires_core_fields(user_root: Path):
    with pytest.raises(ToolError):
        MemoryWriteTool(user_root).run(
            datasource="d", memory_id="c", title="t"
        )


@pytest.mark.parametrize("bad", ["../escape", "/etc/passwd", "a/../../b"])
def test_traversal_ids_are_rejected(user_root: Path, bad: str):
    with pytest.raises(ToolError, match="unsafe"):
        _write(MemoryWriteTool(user_root), memory_id=bad)
    with pytest.raises(ToolError, match="unsafe"):
        MemoryReadTool(user_root).run(datasource="sales_db", memory_id=bad)


def test_edit_append_preserves_body(user_root: Path):
    _write(MemoryWriteTool(user_root))
    MemoryEditTool(user_root).run(
        datasource="sales_db",
        memory_id="metrics/active_user",
        append="## Notes\nConfirmed by query.",
    )
    doc = MemoryReadTool(user_root).run(
        datasource="sales_db", memory_id="metrics/active_user"
    )
    assert "status = 1" in doc  # original body kept
    assert "Confirmed by query." in doc  # appended
