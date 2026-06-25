"""OKF tools exposed to the main agent.

* ``build_datasource_knowledge`` — run the enrichment agent to (re)build a
  bundle for a datasource (requires a live DB connection).
* ``read_concept`` — read one concept document by id.
* ``list_datasource_knowledge`` — list bundles, or the concepts in one bundle.

Reads also work through the ordinary ``read_file`` / ``grep`` tools since the
bundle lives on disk under the working directory; these tools just give the
agent clean, schema-aware addressing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema
from terno_agent.wiki.builder import DatasourceKnowledgeAgent
from terno_agent.wiki.bundle import KnowledgeBundle
from terno_agent.wiki.concept import Concept, ConceptError
from terno_agent.wiki.context import KnowledgeContextProvider
from terno_agent.wiki.paths import bundle_dir

if TYPE_CHECKING:
    from terno_agent.db.connection import Database
    from terno_agent.llm.base import LLMClient


@dataclass
class BuildDatasourceKnowledgeTool:
    workdir: Path
    db: Database | None = None
    database_url: str = ""
    llm: LLMClient | None = None
    default_datasource: str = ""
    max_tables: int = 50
    sample_rows: int = 5

    def _resolve_db(self) -> Database:
        """Return a live DB connection, connecting lazily from the URL.

        Registered unconditionally so the model can always choose to build a
        guide; raises a clear, actionable error when no datasource is
        configured rather than the tool being silently absent.
        """
        if self.db is not None:
            return self.db
        if self.database_url:
            from terno_agent.db.connection import Database as _Database

            self.db = _Database(self.database_url)
            return self.db
        raise ToolError(
            "No datasource is configured. Set TERNO_DATABASE_URL (a SQLAlchemy "
            "URL) so the knowledge agent can introspect the database, then try "
            "again."
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="build_datasource_knowledge",
            description=(
                "Build (or refresh) an Open Knowledge Format bundle describing "
                "the connected datasource: one markdown concept per table "
                "(structure + inferred meaning, enum decoding, gotchas) plus an "
                "index. Writes to disk under .terno/knowledge/<datasource>/. Run "
                "this once per datasource (or when the schema changes); afterwards "
                "read the concepts instead of re-introspecting."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "datasource": {
                        "type": "string",
                        "description": (
                            "Name for the bundle/folder. Defaults to the "
                            "configured datasource name."
                        ),
                    },
                    "tables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional subset of tables to document. Omit to "
                            "document all tables (up to the configured cap)."
                        ),
                    },
                    "refresh": {
                        "type": "boolean",
                        "description": (
                            "Re-document tables that already have a concept "
                            "(default false: existing concepts are skipped)."
                        ),
                    },
                },
                "required": [],
            },
        )

    def run(self, **kwargs: Any) -> str:
        datasource = (kwargs.get("datasource") or self.default_datasource or "").strip()
        if not datasource:
            raise ToolError(
                "build_datasource_knowledge needs a 'datasource' name (none "
                "configured by default)."
            )
        tables = kwargs.get("tables") or None
        if tables is not None and not isinstance(tables, list):
            raise ToolError("'tables' must be an array of table names.")
        refresh = bool(kwargs.get("refresh", False))

        db = self._resolve_db()
        bundle = KnowledgeBundle(
            bundle_dir(self.workdir, datasource), name=datasource
        )
        agent = DatasourceKnowledgeAgent(
            db=db,
            bundle=bundle,
            llm=self.llm,
            max_tables=self.max_tables,
            sample_rows=self.sample_rows,
        )
        report = agent.build(tables=tables, refresh=refresh)
        out = report.to_dict()
        out["bundle_dir"] = str(bundle.root)
        out["index"] = bundle.index_text()
        return json.dumps(out)


@dataclass
class ReadConceptTool:
    workdir: Path

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_concept",
            description=(
                "Read one concept document from a datasource knowledge bundle. "
                "concept_id is the file path within the bundle without '.md' "
                "(e.g. 'tables/users', 'overview')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "datasource": {
                        "type": "string",
                        "description": "Bundle/datasource name.",
                    },
                    "concept_id": {
                        "type": "string",
                        "description": "Concept id, e.g. 'tables/users'.",
                    },
                },
                "required": ["datasource", "concept_id"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        datasource = (kwargs.get("datasource") or "").strip()
        concept_id = (kwargs.get("concept_id") or "").strip()
        if not datasource or not concept_id:
            raise ToolError("read_concept requires 'datasource' and 'concept_id'.")
        bundle = KnowledgeBundle(bundle_dir(self.workdir, datasource), name=datasource)
        concept = bundle.read_concept(concept_id)
        if concept is None:
            raise ToolError(
                f"No concept {concept_id!r} in datasource {datasource!r}."
            )
        return concept.render()


@dataclass
class WriteConceptTool:
    workdir: Path

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_concept",
            description=(
                "Create or replace a single concept document in a datasource "
                "bundle, then regenerate the index. Use for knowledge "
                "introspection can't produce (a metric/term definition, a "
                "business rule, a correction, a gotcha from the conversation). "
                "concept_id is the path within the bundle without '.md' "
                "(e.g. 'tables/users', 'concepts/active_user')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "datasource": {"type": "string", "description": "Bundle name."},
                    "concept_id": {
                        "type": "string",
                        "description": "Concept id, e.g. 'concepts/active_user'.",
                    },
                    "title": {"type": "string", "description": "Human title."},
                    "type": {
                        "type": "string",
                        "description": "Concept type, e.g. table|metric|domain|datasource.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "One-line summary for the index.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Markdown body (sections, links, notes).",
                    },
                },
                "required": ["datasource", "concept_id", "title", "type"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        datasource = (kwargs.get("datasource") or "").strip()
        concept_id = (kwargs.get("concept_id") or "").strip()
        title = (kwargs.get("title") or "").strip()
        type_ = (kwargs.get("type") or "").strip()
        if not (datasource and concept_id and title and type_):
            raise ToolError(
                "write_concept requires 'datasource', 'concept_id', 'title', "
                "and 'type'."
            )
        try:
            concept = Concept(
                concept_id=concept_id,
                title=title,
                type=type_,
                summary=(kwargs.get("summary") or "").strip(),
                body=(kwargs.get("body") or "").strip(),
            )
        except ConceptError as exc:
            raise ToolError(str(exc)) from exc
        bundle = KnowledgeBundle(bundle_dir(self.workdir, datasource), name=datasource)
        path = bundle.write_concept(concept)
        bundle.rebuild_index()
        return json.dumps({"concept_id": concept_id, "path": str(path)})


@dataclass
class ListKnowledgeTool:
    workdir: Path

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_datasource_knowledge",
            description=(
                "List available datasource knowledge bundles, or the concepts "
                "within one bundle when 'datasource' is given."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "datasource": {
                        "type": "string",
                        "description": (
                            "Optional bundle name; omit to list all bundles."
                        ),
                    }
                },
                "required": [],
            },
        )

    def run(self, **kwargs: Any) -> str:
        datasource = (kwargs.get("datasource") or "").strip()
        if not datasource:
            provider = KnowledgeContextProvider(self.workdir)
            return json.dumps([b.name for b in provider.bundles()])
        bundle = KnowledgeBundle(bundle_dir(self.workdir, datasource), name=datasource)
        if not bundle.exists():
            raise ToolError(f"No knowledge bundle for datasource {datasource!r}.")
        return json.dumps(
            [
                {"concept_id": c.concept_id, "title": c.title, "summary": c.summary}
                for c in bundle.list_concepts()
            ]
        )


def knowledge_agent_tools(
    workdir: Path,
    *,
    db: Database | None = None,
    database_url: str = "",
    llm: LLMClient | None = None,
    datasource: str = "",
    max_tables: int = 50,
) -> list:
    """The toolset for the knowledge agent: list / read / write / build."""
    return [
        ListKnowledgeTool(workdir),
        ReadConceptTool(workdir),
        WriteConceptTool(workdir),
        BuildDatasourceKnowledgeTool(
            workdir=workdir,
            db=db,
            database_url=database_url,
            llm=llm,
            default_datasource=datasource,
            max_tables=max_tables,
        ),
    ]


__all__ = [
    "BuildDatasourceKnowledgeTool",
    "ListKnowledgeTool",
    "ReadConceptTool",
    "WriteConceptTool",
    "knowledge_agent_tools",
]
