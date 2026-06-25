"""Bundle write/read + index regeneration."""

from __future__ import annotations

from pathlib import Path

from terno_agent.okf.bundle import KnowledgeBundle
from terno_agent.okf.concept import Concept


def _bundle(tmp_path: Path) -> KnowledgeBundle:
    return KnowledgeBundle(tmp_path / "sales_db", name="sales_db")


def test_write_creates_subdirs_and_read_roundtrip(tmp_path: Path):
    b = _bundle(tmp_path)
    b.write_concept(
        Concept("tables/users", "users", "table", summary="Users.", body="hi")
    )
    assert (b.root / "tables" / "users.md").exists()
    got = b.read_concept("tables/users")
    assert got is not None and got.title == "users" and got.body == "hi"
    assert b.read_concept("tables/missing") is None


def test_rebuild_index_lists_and_links(tmp_path: Path):
    b = _bundle(tmp_path)
    b.write_concept(Concept("overview", "overview", "datasource", summary="DS."))
    b.write_concept(Concept("tables/users", "users", "table", summary="Users."))
    b.write_concept(Concept("tables/orders", "orders", "table", summary="Orders."))
    b.rebuild_index()

    assert b.exists()
    root = (b.root / "index.md").read_text()
    # Root concept linked relatively; subdir grouped under a heading.
    assert "[overview](overview.md)" in root
    assert "## tables/" in root
    assert "[users](tables/users.md)" in root
    assert "Users." in root

    sub = (b.root / "tables" / "index.md").read_text()
    assert "[users](users.md)" in sub
    assert "[orders](orders.md)" in sub


def test_list_concepts_excludes_index(tmp_path: Path):
    b = _bundle(tmp_path)
    b.write_concept(Concept("tables/users", "users", "table"))
    b.rebuild_index()
    ids = {c.concept_id for c in b.list_concepts()}
    assert ids == {"tables/users"}  # index.md files are not concepts


def test_index_text_strips_frontmatter(tmp_path: Path):
    b = _bundle(tmp_path)
    b.write_concept(Concept("tables/users", "users", "table", summary="Users."))
    b.rebuild_index()
    text = b.index_text()
    assert "---" not in text
    assert "users" in text
