"""DatasourceKnowledgeAgent: introspection + LLM enrichment."""

from __future__ import annotations

from pathlib import Path

from terno_agent.db.connection import Database
from terno_agent.okf.builder import DatasourceKnowledgeAgent
from terno_agent.okf.bundle import KnowledgeBundle


def _bundle(tmp_path: Path) -> KnowledgeBundle:
    return KnowledgeBundle(tmp_path / "kb" / "sales_db", name="sales_db")


def test_introspection_only(sqlite_db: Database, tmp_path: Path):
    b = _bundle(tmp_path)
    agent = DatasourceKnowledgeAgent(
        db=sqlite_db, bundle=b, llm=None, today="2026-06-24"
    )
    report = agent.build()
    assert set(report.tables_written) == {"users", "orders"}
    assert report.enriched is False

    users = b.read_concept("tables/users")
    assert users is not None
    assert users.metadata["source"] == "introspection"
    assert "| `email` |" in users.body  # column table rendered
    # FK relationship from orders → users links to the sibling concept.
    orders = b.read_concept("tables/orders")
    assert "[users](users.md)" in orders.body
    # Overview + index exist.
    assert b.read_concept("overview") is not None
    assert b.exists()


def test_llm_enrichment_merged(sqlite_db: Database, tmp_path: Path, scripted_llm):
    b = _bundle(tmp_path)
    agent = DatasourceKnowledgeAgent(
        db=sqlite_db, bundle=b, llm=scripted_llm, today="2026-06-24"
    )
    report = agent.build()
    assert report.enriched is True
    assert scripted_llm.calls == 2  # one per table

    users = b.read_concept("tables/users")
    assert users.metadata["source"] == "introspection+llm"
    assert users.summary == "Enriched summary."
    assert "## Overview" in users.body
    assert "1=active, 0=inactive" in users.body  # column desc + note
    assert "## Notes & Gotchas" in users.body


def test_tables_filter_and_skip(sqlite_db: Database, tmp_path: Path):
    b = _bundle(tmp_path)
    agent = DatasourceKnowledgeAgent(db=sqlite_db, bundle=b, llm=None)
    report = agent.build(tables=["users"])
    assert report.tables_written == ["users"]
    assert b.read_concept("tables/orders") is None

    # Re-running without refresh skips the existing concept.
    report2 = agent.build(tables=["users"])
    assert report2.tables_written == []
    assert report2.tables_skipped == ["users"]

    # refresh=True re-writes it.
    report3 = agent.build(tables=["users"], refresh=True)
    assert report3.tables_written == ["users"]


def test_max_tables_truncation_reported(sqlite_db: Database, tmp_path: Path):
    b = _bundle(tmp_path)
    agent = DatasourceKnowledgeAgent(db=sqlite_db, bundle=b, llm=None, max_tables=1)
    report = agent.build()
    assert len(report.tables_written) == 1
    assert len(report.tables_truncated) == 1
    assert set(report.tables_written + report.tables_truncated) == {"users", "orders"}


def test_enrichment_failure_degrades(sqlite_db: Database, tmp_path: Path):
    class BoomLLM:
        model = "boom"

        def complete(self, *a, **k):
            raise RuntimeError("nope")

    b = _bundle(tmp_path)
    agent = DatasourceKnowledgeAgent(db=sqlite_db, bundle=b, llm=BoomLLM())
    report = agent.build()  # must not raise
    assert set(report.tables_written) == {"users", "orders"}
    users = b.read_concept("tables/users")
    assert users.metadata["source"] == "introspection"
