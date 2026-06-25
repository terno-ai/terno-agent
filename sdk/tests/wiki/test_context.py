"""KnowledgeContextProvider: pre-turn injection block."""

from __future__ import annotations

from pathlib import Path

from terno_agent.okf.bundle import KnowledgeBundle
from terno_agent.okf.concept import Concept
from terno_agent.okf.context import KnowledgeContextProvider
from terno_agent.okf.paths import bundle_dir


def test_empty_when_no_bundles(workdir: Path):
    assert KnowledgeContextProvider(workdir).context_block() == ""
    assert KnowledgeContextProvider(workdir).bundles() == []


def test_block_lists_bundles_and_index(workdir: Path):
    b = KnowledgeBundle(bundle_dir(workdir, "sales_db"), name="sales_db")
    b.write_concept(Concept("tables/users", "users", "table", summary="Users."))
    b.rebuild_index()

    provider = KnowledgeContextProvider(workdir)
    assert [x.name for x in provider.bundles()] == ["sales_db"]
    block = provider.context_block()
    assert "Datasource knowledge" in block
    assert "sales_db" in block
    assert "users" in block
    assert "curated automatically" in block  # footer hint
