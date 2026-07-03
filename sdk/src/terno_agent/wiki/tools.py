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
import re
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


_SEARCH_DEFAULT_LIMIT = 20
_SNIPPETS_PER_CONCEPT = 5


def _match_concept(rx: "re.Pattern[str]", concept: Concept) -> list[str]:
    """Return labelled snippets where ``rx`` matches a concept's text.

    Title and summary are surfaced as their own lines (they carry the most
    signal for the index); the body is scanned line by line so the caller
    sees the matching context, not the whole document.
    """
    snippets: list[str] = []
    for label, text in (("title", concept.title), ("summary", concept.summary)):
        if text and rx.search(text):
            snippets.append(f"{label}: {text}")
    for lineno, line in enumerate(concept.body.splitlines(), start=1):
        if rx.search(line):
            snippets.append(f"L{lineno}: {line.strip()}")
    return snippets


@dataclass
class SearchKnowledgeTool:
    """OKF-aware content search across a bundle's concept documents.

    Unlike ``read_concept`` (which needs an exact id) this walks every
    concept in every subdirectory and returns the ones whose title, summary,
    or body match — the way to locate relevant knowledge in a nested bundle
    without reading each file. Follow up with ``read_concept`` on the hits.
    """

    workdir: Path
    default_datasource: str = ""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="search_datasource_knowledge",
            description=(
                "Search the concept documents in a datasource knowledge bundle "
                "for a term or regex (case-insensitive). Scans titles, "
                "summaries, and bodies across every subdirectory and returns "
                "the matching concepts with the lines that matched. Use this to "
                "find where relevant knowledge lives without reading every "
                "file, then `read_concept` the returned concept_ids for full "
                "detail."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Text or regular expression to search for "
                            "(matched case-insensitively)."
                        ),
                    },
                    "datasource": {
                        "type": "string",
                        "description": (
                            "Bundle to search. Omit to search across every "
                            "available bundle."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Maximum matching concepts to return "
                            f"(default {_SEARCH_DEFAULT_LIMIT})."
                        ),
                    },
                },
                "required": ["query"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        query = (kwargs.get("query") or "").strip()
        if not query:
            raise ToolError("search_datasource_knowledge requires a 'query'.")
        limit = int(kwargs.get("limit") or _SEARCH_DEFAULT_LIMIT)
        if limit <= 0:
            raise ToolError("limit must be positive.")
        # Treat the query as a regex, but fall back to a literal match so a
        # stray metacharacter (e.g. a bare '(') never errors the tool.
        try:
            rx = re.compile(query, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(query), re.IGNORECASE)

        datasource = (
            kwargs.get("datasource") or self.default_datasource or ""
        ).strip()
        if datasource:
            bundle = KnowledgeBundle(
                bundle_dir(self.workdir, datasource), name=datasource
            )
            if not bundle.exists():
                raise ToolError(
                    f"No knowledge bundle for datasource {datasource!r}."
                )
            bundles = [bundle]
        else:
            bundles = KnowledgeContextProvider(self.workdir).bundles()

        hits: list[dict[str, Any]] = []
        for bundle in bundles:
            for concept in bundle.list_concepts():
                snippets = _match_concept(rx, concept)
                if not snippets:
                    continue
                hits.append(
                    {
                        "datasource": bundle.name,
                        "concept_id": concept.concept_id,
                        "title": concept.title,
                        "summary": concept.summary,
                        "matches": snippets[:_SNIPPETS_PER_CONCEPT],
                    }
                )
                if len(hits) >= limit:
                    return json.dumps(hits)
        return json.dumps(hits)


def knowledge_agent_tools(
    workdir: Path,
    *,
    db: Database | None = None,
    database_url: str = "",
    llm: LLMClient | None = None,
    datasource: str = "",
    max_tables: int = 50,
) -> list:
    """The toolset for the knowledge agent: list / search / read / write / build."""
    return [
        ListKnowledgeTool(workdir),
        SearchKnowledgeTool(workdir, default_datasource=datasource),
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
    "SearchKnowledgeTool",
    "WriteConceptTool",
    "knowledge_agent_tools",
]
