"""OKF tools: build, read, list."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.db.connection import Database
from terno_agent.wiki.tools import (
    BuildDatasourceKnowledgeTool,
    EditConceptTool,
    ListKnowledgeTool,
    ReadConceptTool,
    WriteConceptTool,
)


def test_build_then_read_and_list(sqlite_db: Database, workdir: Path, scripted_llm):
    build = BuildDatasourceKnowledgeTool(
        db=sqlite_db, workdir=workdir, llm=scripted_llm, default_datasource="sales_db"
    )
    out = json.loads(build.run())
    assert out["datasource"] == "sales_db"
    assert set(out["tables_written"]) == {"users", "orders"}
    assert "users" in out["index"]
    assert Path(out["bundle_dir"]).exists()

    read = ReadConceptTool(workdir)
    doc = read.run(datasource="sales_db", concept_id="tables/users")
    assert "title: users" in doc
    assert "Enriched summary." in doc

    listing = ListKnowledgeTool(workdir)
    assert json.loads(listing.run()) == ["sales_db"]
    concepts = json.loads(listing.run(datasource="sales_db"))
    ids = {c["concept_id"] for c in concepts}
    assert {"overview", "tables/users", "tables/orders"} <= ids


def test_write_concept_creates_and_indexes(workdir: Path):
    tool = WriteConceptTool(workdir)
    out = json.loads(
        tool.run(
            datasource="sales_db",
            concept_id="concepts/active_user",
            title="Active user",
            type="metric",
            summary="status = 1",
            body="An active user has status = 1.",
        )
    )
    assert out["concept_id"] == "concepts/active_user"
    # The concept is readable and the index now lists it.
    read = ReadConceptTool(workdir)
    doc = read.run(datasource="sales_db", concept_id="concepts/active_user")
    assert "type: metric" in doc
    listing = json.loads(ListKnowledgeTool(workdir).run(datasource="sales_db"))
    assert any(c["concept_id"] == "concepts/active_user" for c in listing)


def test_write_concept_requires_fields(workdir: Path):
    with pytest.raises(ToolError):
        WriteConceptTool(workdir).run(datasource="d", concept_id="c", title="t")


def test_write_concept_stamps_provenance(workdir: Path):
    WriteConceptTool(workdir).run(
        datasource="sales_db",
        concept_id="tables/users",
        title="Users",
        type="table",
        body="Registered users.",
        source="introspection",
    )
    doc = ReadConceptTool(workdir).run(datasource="sales_db", concept_id="tables/users")
    assert "source: introspection" in doc
    assert "updated:" in doc  # timestamp recorded automatically


def _seed_users(workdir: Path) -> None:
    WriteConceptTool(workdir).run(
        datasource="sales_db",
        concept_id="tables/users",
        title="Users",
        type="table",
        summary="Registered users.",
        body="## Overview\nOne row per user.\n\nstatus is an int.",
    )


def test_edit_concept_append_preserves_body(workdir: Path):
    _seed_users(workdir)
    EditConceptTool(workdir).run(
        datasource="sales_db",
        concept_id="tables/users",
        append="## Notes\nstatus = 1 means active.",
        source="query",
    )
    doc = ReadConceptTool(workdir).run(datasource="sales_db", concept_id="tables/users")
    assert "One row per user." in doc  # original body kept
    assert "status = 1 means active." in doc  # appended fact
    assert "source: query" in doc


def test_edit_concept_replaces_unique_span(workdir: Path):
    _seed_users(workdir)
    EditConceptTool(workdir).run(
        datasource="sales_db",
        concept_id="tables/users",
        old_string="status is an int.",
        new_string="status: 1=active, 0=inactive.",
    )
    doc = ReadConceptTool(workdir).run(datasource="sales_db", concept_id="tables/users")
    assert "1=active, 0=inactive." in doc
    assert "status is an int." not in doc


def test_edit_concept_errors_when_missing(workdir: Path):
    with pytest.raises(ToolError):
        EditConceptTool(workdir).run(
            datasource="sales_db", concept_id="tables/ghost", append="x"
        )


def test_edit_concept_errors_on_ambiguous_old_string(workdir: Path):
    WriteConceptTool(workdir).run(
        datasource="sales_db",
        concept_id="tables/users",
        title="Users",
        type="table",
        body="dup\ndup",
    )
    with pytest.raises(ToolError):
        EditConceptTool(workdir).run(
            datasource="sales_db",
            concept_id="tables/users",
            old_string="dup",
            new_string="x",
        )


def test_edit_concept_requires_a_change(workdir: Path):
    _seed_users(workdir)
    with pytest.raises(ToolError):
        EditConceptTool(workdir).run(datasource="sales_db", concept_id="tables/users")


def test_build_connects_lazily_from_url(sqlite_db: Database, workdir: Path):
    # No live db handed in — only the URL. The tool connects on demand.
    build = BuildDatasourceKnowledgeTool(
        workdir=workdir,
        database_url=str(sqlite_db.url),
        default_datasource="sales_db",
    )
    out = json.loads(build.run())
    assert set(out["tables_written"]) == {"users", "orders"}


def test_build_errors_without_any_datasource(workdir: Path):
    build = BuildDatasourceKnowledgeTool(workdir=workdir, default_datasource="ds")
    with pytest.raises(ToolError):  # no db, no url
        build.run()


def test_build_requires_datasource(sqlite_db: Database, workdir: Path):
    build = BuildDatasourceKnowledgeTool(db=sqlite_db, workdir=workdir, llm=None)
    with pytest.raises(ToolError):
        build.run()  # no default, none passed


def test_read_missing_concept_errors(sqlite_db: Database, workdir: Path):
    build = BuildDatasourceKnowledgeTool(
        db=sqlite_db, workdir=workdir, llm=None, default_datasource="sales_db"
    )
    build.run()
    read = ReadConceptTool(workdir)
    with pytest.raises(ToolError):
        read.run(datasource="sales_db", concept_id="tables/nope")
    with pytest.raises(ToolError):
        ListKnowledgeTool(workdir).run(datasource="ghost")
