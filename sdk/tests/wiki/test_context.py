"""MemoryContextProvider: pre-turn recall block over user + org folders."""

from __future__ import annotations

from pathlib import Path

from terno_agent.wiki.context import MemoryContextProvider
from terno_agent.wiki.tools import MemoryWriteTool


def _seed(root: Path, memory_id: str, title: str, *, shared: bool, org=None) -> None:
    tool = MemoryWriteTool(root, org_root=org, is_org_admin=True)
    tool.run(
        memory_id=memory_id,
        title=title,
        type="metric",
        scope="global",
        summary=f"{title} summary.",
        shared=shared,
    )


def test_empty_when_no_bundles(tmp_path: Path):
    provider = MemoryContextProvider(tmp_path / "memory")
    assert provider.context_block() == ""
    assert provider.bundles() == []


def test_block_lists_user_bundle(tmp_path: Path):
    user_root = tmp_path / "user" / "memory"
    _seed(user_root, "active_user", "Active user", shared=False)

    provider = MemoryContextProvider(user_root)
    assert [b.name for b in provider.bundles()] == ["memory"]
    block = provider.context_block()
    assert "Available memory" in block
    assert "Active user" in block
    assert "curated automatically" in block  # footer


def test_block_separates_org_shared_section(tmp_path: Path):
    user_root = tmp_path / "user" / "memory"
    org_root = tmp_path / "org" / "memory"
    _seed(user_root, "output_prefs", "Output prefs", shared=False)
    _seed(user_root, "active_user", "Active user", shared=True, org=org_root)

    provider = MemoryContextProvider(user_root, org_root=org_root)
    assert [b.name for b in provider.org_bundles()] == ["memory"]
    block = provider.context_block()
    assert "Output prefs" in block  # private
    assert "Organisation-wide shared memory" in block  # org section header
    assert "(read-only" in block  # org section header
