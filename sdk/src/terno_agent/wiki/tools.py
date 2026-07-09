"""Wiki memory tools.

Two toolsets are exposed:

* The **main agent** gets the READ-ONLY set (``list_memory``, ``search_memory``,
  ``read_memory``) so it can recall facts while answering.
* The **wiki memory agent** (the background curator) gets the full set,
  adding ``write_memory`` and ``edit_memory`` so it can record and refine
  facts after a turn.

Storage uses the OKF bundle engine (``KnowledgeBundle`` / ``Concept``): one
markdown file per fact under ``<memory>/<datasource>/`` with a generated
``index.md``. Unlike datasource *knowledge*, memory bundles live directly in
the workspace ``memory`` folders (never under ``.terno``):

* **private** memory → the caller's user folder
  (``USER_WORKSPACE_ROOT/<org>/<user>/memory``);
* **shared** memory → the org folder (``ORG_WORKSPACE_ROOT/<org>/memory``),
  which **only an org admin may write to**.

Everyone in an org can READ the shared folder; the read tools search the user
folder and the org folder together.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema
from terno_agent.wiki.bundle import KnowledgeBundle
from terno_agent.wiki.concept import Concept, ConceptError
from terno_agent.wiki.context import MemoryContextProvider
from terno_agent.wiki.paths import memory_bundle_dir

# Where a fact came from — recorded in frontmatter as `source` so a reader can
# weigh how much to trust it and a future curator knows what may be stale.
_KNOWN_SOURCES = ("introspection", "query", "conversation", "user", "curator")
_DEFAULT_SOURCE = "curator"

# Memory types from terno-ai plus the datasource-knowledge types.
_KNOWN_TYPES = (
    "user", "feedback", "project", "reference",
    "table", "domain", "metric", "datasource",
)


def _utc_now_iso() -> str:
    """Current UTC time as a second-precision ISO string for `updated`."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_scope(scope: str, datasource_name: str) -> dict[str, str]:
    """Return the ``scope``/``datasource_name`` metadata for a memory.

    ``scope: datasource:<id>`` keeps ``datasource_name``; ``global`` (or an
    empty/unknown value) drops it. This is the terno-ai scoping rule — it
    controls a fact's APPLICABILITY (which database), independent of whether
    the file is stored privately or shared with the org.
    """
    scope = (scope or "").strip()
    if scope.startswith("datasource:"):
        out = {"scope": scope}
        if datasource_name.strip():
            out["datasource_name"] = datasource_name.strip()
        return out
    return {"scope": "global"}


def _stamp(metadata: dict[str, Any], source: str, session_id: str) -> dict[str, Any]:
    """Return ``metadata`` with provenance set.

    Every write/edit refreshes ``updated`` and marks ``node_type: memory``.
    ``source`` is only overwritten when a recognised one is supplied.
    ``originSessionId`` is set once (on create) and preserved on later edits.
    """
    stamped = dict(metadata)
    stamped["node_type"] = "memory"
    stamped["updated"] = _utc_now_iso()
    source = (source or "").strip().lower()
    if source in _KNOWN_SOURCES:
        stamped["source"] = source
    elif "source" not in stamped:
        stamped["source"] = _DEFAULT_SOURCE
    if session_id and not stamped.get("originSessionId"):
        stamped["originSessionId"] = session_id
    return stamped


def _write_target(
    *,
    shared: bool,
    user_root: Path,
    org_root: Path | None,
    is_org_admin: bool,
    action: str,
) -> Path:
    """Pick the memory folder for a write/edit, enforcing the org-admin gate.

    ``shared`` routes to the org folder — permitted only for org admins and
    only when an org folder is configured. Otherwise the caller's private user
    folder is used.
    """
    if not shared:
        return user_root
    if not is_org_admin:
        raise ToolError(
            f"Only an org admin may {action} organisation-shared memory. "
            "Store this as private memory instead (shared=false)."
        )
    if org_root is None:
        raise ToolError(
            "No organisation memory folder is configured; cannot write shared "
            "memory."
        )
    return org_root


@dataclass
class MemoryReadTool:
    user_root: Path
    org_root: Path | None = None

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_memory",
            description=(
                "Read one memory file from a datasource memory bundle. Searches "
                "your private memory first, then organisation-shared memory. "
                "memory_id is the file path within the bundle without '.md' "
                "(e.g. 'metrics/active_user', 'datasource', 'domains/identity')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "datasource": {
                        "type": "string",
                        "description": "Memory bundle / datasource name.",
                    },
                    "memory_id": {
                        "type": "string",
                        "description": "Memory id, e.g. 'domains/identity'.",
                    },
                },
                "required": ["datasource", "memory_id"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        datasource = (kwargs.get("datasource") or "").strip()
        memory_id = (kwargs.get("memory_id") or "").strip()
        if not datasource or not memory_id:
            raise ToolError("read_memory requires 'datasource' and 'memory_id'.")
        if memory_id.startswith("/") or ".." in memory_id.split("/"):
            raise ToolError(f"unsafe memory_id {memory_id!r}.")
        bundle = KnowledgeBundle(
            memory_bundle_dir(self.user_root, datasource), name=datasource
        )
        concept = bundle.read_concept(memory_id)
        if concept is None and self.org_root is not None:
            org_bundle = KnowledgeBundle(
                memory_bundle_dir(self.org_root, datasource), name=datasource
            )
            concept = org_bundle.read_concept(memory_id)
        if concept is None:
            raise ToolError(
                f"No memory {memory_id!r} in datasource {datasource!r}."
            )
        return concept.render()


@dataclass
class MemoryListTool:
    user_root: Path
    org_root: Path | None = None

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_memory",
            description=(
                "List available memory bundles (private + organisation-shared), "
                "or the memories within one bundle when 'datasource' is given."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "datasource": {
                        "type": "string",
                        "description": "Optional bundle name; omit to list all.",
                    }
                },
                "required": [],
            },
        )

    def run(self, **kwargs: Any) -> str:
        datasource = (kwargs.get("datasource") or "").strip()
        if not datasource:
            provider = MemoryContextProvider(self.user_root, org_root=self.org_root)
            names = [b.name for b in provider.bundles()]
            names += [b.name for b in provider.org_bundles()]
            return json.dumps(sorted(set(names)))
        concepts = self._list_bundle(datasource)
        return json.dumps(concepts)

    def _list_bundle(self, datasource: str) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        seen: set[str] = set()
        for root, shared in ((self.user_root, False), (self.org_root, True)):
            if root is None:
                continue
            bundle = KnowledgeBundle(
                memory_bundle_dir(root, datasource), name=datasource
            )
            if not bundle.exists():
                continue
            for c in bundle.list_concepts():
                if c.concept_id in seen:
                    continue
                seen.add(c.concept_id)
                found.append(
                    {
                        "memory_id": c.concept_id,
                        "title": c.title,
                        "type": c.type,
                        "scope": c.metadata.get("scope", "global"),
                        "shared": shared,
                        "summary": c.summary,
                    }
                )
        if not found:
            raise ToolError(f"No memory bundle for datasource {datasource!r}.")
        return found


_SEARCH_DEFAULT_LIMIT = 20
_SNIPPETS_PER_MEMORY = 5


def _match(rx: re.Pattern[str], concept: Concept) -> list[str]:
    """Return labelled snippets where ``rx`` matches a memory's text."""
    snippets: list[str] = []
    for label, text in (("title", concept.title), ("summary", concept.summary)):
        if text and rx.search(text):
            snippets.append(f"{label}: {text}")
    for lineno, line in enumerate(concept.body.splitlines(), start=1):
        if rx.search(line):
            snippets.append(f"L{lineno}: {line.strip()}")
    return snippets


@dataclass
class MemorySearchTool:
    user_root: Path
    default_datasource: str = ""
    org_root: Path | None = None

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="search_memory",
            description=(
                "Search memory files for a term or regex (case-insensitive) "
                "across both private and organisation-shared memory. Scans "
                "titles, summaries, and bodies across every subdirectory and "
                "returns the matching memories with the lines that matched. Use "
                "this to find where relevant knowledge lives, then read_memory "
                "the returned memory_ids for detail."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text or regex (matched case-insensitively).",
                    },
                    "datasource": {
                        "type": "string",
                        "description": "Bundle to search. Omit to search all.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            f"Max matching memories (default {_SEARCH_DEFAULT_LIMIT})."
                        ),
                    },
                },
                "required": ["query"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        query = (kwargs.get("query") or "").strip()
        if not query:
            raise ToolError("search_memory requires a 'query'.")
        limit = int(kwargs.get("limit") or _SEARCH_DEFAULT_LIMIT)
        if limit <= 0:
            raise ToolError("limit must be positive.")
        try:
            rx = re.compile(query, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(query), re.IGNORECASE)

        datasource = (
            kwargs.get("datasource") or self.default_datasource or ""
        ).strip()
        bundles = self._bundles(datasource)

        hits: list[dict[str, Any]] = []
        for bundle, shared in bundles:
            for concept in bundle.list_concepts():
                snippets = _match(rx, concept)
                if not snippets:
                    continue
                hits.append(
                    {
                        "datasource": bundle.name,
                        "memory_id": concept.concept_id,
                        "title": concept.title,
                        "scope": concept.metadata.get("scope", "global"),
                        "shared": shared,
                        "summary": concept.summary,
                        "matches": snippets[:_SNIPPETS_PER_MEMORY],
                    }
                )
                if len(hits) >= limit:
                    return json.dumps(hits)
        return json.dumps(hits)

    def _bundles(self, datasource: str) -> list[tuple[KnowledgeBundle, bool]]:
        if datasource:
            out: list[tuple[KnowledgeBundle, bool]] = []
            for root, shared in ((self.user_root, False), (self.org_root, True)):
                if root is None:
                    continue
                bundle = KnowledgeBundle(
                    memory_bundle_dir(root, datasource), name=datasource
                )
                if bundle.exists():
                    out.append((bundle, shared))
            if not out:
                raise ToolError(f"No memory bundle for datasource {datasource!r}.")
            return out
        provider = MemoryContextProvider(self.user_root, org_root=self.org_root)
        return [(b, False) for b in provider.bundles()] + [
            (b, True) for b in provider.org_bundles()
        ]


@dataclass
class MemoryWriteTool:
    user_root: Path
    org_root: Path | None = None
    is_org_admin: bool = False
    session_id: str = ""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_memory",
            description=(
                "Create a NEW memory file (or fully replace one) in a bundle, "
                "then regenerate the index. Use for a durable fact that has no "
                "file yet: a metric/term definition, a business rule, an enum "
                "decoding, a join path, or a stable user preference. To ADD to "
                "or correct an EXISTING memory, use edit_memory instead. "
                "memory_id is the path within the bundle without '.md' "
                "(e.g. 'metrics/active_user'). Set shared=true to store the fact "
                "in organisation-shared memory (org admins only); otherwise it "
                "is saved as your private memory. Records provenance "
                "automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "datasource": {"type": "string", "description": "Bundle name."},
                    "memory_id": {
                        "type": "string",
                        "description": "Memory id, e.g. 'metrics/active_user'.",
                    },
                    "title": {"type": "string", "description": "Human title."},
                    "type": {
                        "type": "string",
                        "description": (
                            "One of: user|feedback|project|reference (memory "
                            "about the user/work) or table|domain|metric|"
                            "datasource (knowledge about the data)."
                        ),
                    },
                    "scope": {
                        "type": "string",
                        "description": (
                            "Applicability: 'datasource:<id>' for a fact "
                            "specific to one database, or 'global' otherwise."
                        ),
                    },
                    "shared": {
                        "type": "boolean",
                        "description": (
                            "true = save to organisation-shared memory (org "
                            "admins only). false/omitted = your private memory."
                        ),
                    },
                    "datasource_name": {
                        "type": "string",
                        "description": "Datasource name (only when scope is a datasource).",
                    },
                    "summary": {
                        "type": "string",
                        "description": "One-line summary for the index.",
                    },
                    "body": {
                        "type": "string",
                        "description": (
                            "Markdown body. For feedback/project include "
                            "**Why:** and **How to apply:** lines."
                        ),
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Where this came from: introspection|query|"
                            "conversation|user. Defaults to 'curator'."
                        ),
                    },
                },
                "required": ["datasource", "memory_id", "title", "type", "scope"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        datasource = (kwargs.get("datasource") or "").strip()
        memory_id = (kwargs.get("memory_id") or "").strip()
        title = (kwargs.get("title") or "").strip()
        type_ = (kwargs.get("type") or "").strip()
        scope = (kwargs.get("scope") or "").strip()
        if not (datasource and memory_id and title and type_ and scope):
            raise ToolError(
                "write_memory requires 'datasource', 'memory_id', 'title', "
                "'type', and 'scope'."
            )
        if memory_id.startswith("/") or ".." in memory_id.split("/"):
            raise ToolError(f"unsafe memory_id {memory_id!r}.")
        shared = bool(kwargs.get("shared"))
        root = _write_target(
            shared=shared,
            user_root=self.user_root,
            org_root=self.org_root,
            is_org_admin=self.is_org_admin,
            action="write",
        )
        metadata = _normalize_scope(scope, kwargs.get("datasource_name") or "")
        metadata["shared"] = shared
        metadata = _stamp(metadata, kwargs.get("source") or "", self.session_id)
        try:
            concept = Concept(
                concept_id=memory_id,
                title=title,
                type=type_,
                summary=(kwargs.get("summary") or "").strip(),
                body=(kwargs.get("body") or "").strip(),
                metadata=metadata,
            )
        except ConceptError as exc:
            raise ToolError(str(exc)) from exc
        bundle = KnowledgeBundle(
            memory_bundle_dir(root, datasource), name=datasource
        )
        path = bundle.write_concept(concept)
        bundle.rebuild_index()
        return json.dumps(
            {"memory_id": memory_id, "shared": shared, "path": str(path)}
        )


@dataclass
class MemoryEditTool:
    """Targeted, additive edits to an EXISTING memory file."""

    user_root: Path
    org_root: Path | None = None
    is_org_admin: bool = False
    session_id: str = ""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="edit_memory",
            description=(
                "Make a targeted, additive edit to an EXISTING memory without "
                "rewriting the whole file, then regenerate the index. Prefer "
                "this over write_memory when the memory already exists. Provide "
                "'append' to add a markdown block to the end, and/or "
                "'old_string'+'new_string' to replace an exact, unique span "
                "(empty new_string deletes it). You may also update "
                "title/summary/type/scope. Set shared=true to edit the "
                "organisation-shared copy (org admins only). Fails if the memory "
                "does not exist or if old_string is missing/not unique. "
                "Refreshes provenance."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "datasource": {"type": "string", "description": "Bundle name."},
                    "memory_id": {
                        "type": "string",
                        "description": "Existing memory id, e.g. 'domains/identity'.",
                    },
                    "shared": {
                        "type": "boolean",
                        "description": (
                            "true = edit the organisation-shared copy (org "
                            "admins only). false/omitted = your private memory."
                        ),
                    },
                    "append": {
                        "type": "string",
                        "description": "Markdown block to append to the body.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact body text to replace (must be unique).",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement (empty deletes). Required with old_string.",
                    },
                    "title": {"type": "string", "description": "Optional new title."},
                    "summary": {"type": "string", "description": "Optional new summary."},
                    "type": {"type": "string", "description": "Optional new type."},
                    "scope": {
                        "type": "string",
                        "description": "Optional new scope ('datasource:<id>' or 'global').",
                    },
                    "datasource_name": {
                        "type": "string",
                        "description": "Datasource name (when scope is a datasource).",
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Where the change came from. Preserves existing if omitted."
                        ),
                    },
                },
                "required": ["datasource", "memory_id"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        datasource = (kwargs.get("datasource") or "").strip()
        memory_id = (kwargs.get("memory_id") or "").strip()
        if not (datasource and memory_id):
            raise ToolError("edit_memory requires 'datasource' and 'memory_id'.")
        if memory_id.startswith("/") or ".." in memory_id.split("/"):
            raise ToolError(f"unsafe memory_id {memory_id!r}.")

        append = kwargs.get("append")
        old_string = kwargs.get("old_string")
        new_string = kwargs.get("new_string")
        title = kwargs.get("title")
        summary = kwargs.get("summary")
        type_ = kwargs.get("type")
        scope = kwargs.get("scope")
        if not any(
            v not in (None, "")
            for v in (append, old_string, title, summary, type_, scope)
        ):
            raise ToolError(
                "edit_memory needs something to change: 'append', "
                "'old_string'/'new_string', or a title/summary/type/scope update."
            )

        shared = bool(kwargs.get("shared"))
        root = _write_target(
            shared=shared,
            user_root=self.user_root,
            org_root=self.org_root,
            is_org_admin=self.is_org_admin,
            action="edit",
        )
        bundle = KnowledgeBundle(
            memory_bundle_dir(root, datasource), name=datasource
        )
        concept = bundle.read_concept(memory_id)
        if concept is None:
            raise ToolError(
                f"No memory {memory_id!r} in datasource {datasource!r} to edit. "
                "Use write_memory to create it."
            )

        body = concept.body
        if old_string is not None and old_string != "":
            if new_string is None:
                raise ToolError(
                    "edit_memory: 'new_string' is required when 'old_string' is given."
                )
            count = body.count(old_string)
            if count == 0:
                raise ToolError(
                    "edit_memory: 'old_string' not found in the memory body."
                )
            if count > 1:
                raise ToolError(
                    f"edit_memory: 'old_string' is not unique ({count} matches). "
                    "Include more surrounding context."
                )
            body = body.replace(old_string, new_string, 1)

        if append not in (None, ""):
            block = append.strip()
            body = f"{body.rstrip()}\n\n{block}" if body.strip() else block

        def _override(value: Any, current: str) -> str:
            return value.strip() if isinstance(value, str) and value.strip() else current

        metadata = dict(concept.metadata)
        if isinstance(scope, str) and scope.strip():
            # Recompute scope/datasource_name, dropping a stale datasource_name
            # when moving to global.
            metadata.pop("datasource_name", None)
            metadata.update(
                _normalize_scope(scope, kwargs.get("datasource_name") or "")
            )
        metadata["shared"] = shared
        metadata = _stamp(metadata, kwargs.get("source") or "", self.session_id)

        try:
            edited = Concept(
                concept_id=memory_id,
                title=_override(title, concept.title),
                type=_override(type_, concept.type),
                summary=(
                    summary.strip()
                    if isinstance(summary, str) and summary != ""
                    else concept.summary
                ),
                body=body,
                metadata=metadata,
            )
        except ConceptError as exc:
            raise ToolError(str(exc)) from exc

        path = bundle.write_concept(edited)
        bundle.rebuild_index()
        return json.dumps(
            {"memory_id": memory_id, "shared": shared, "path": str(path)}
        )


def memory_read_tools(
    user_root: Path,
    *,
    datasource: str = "",
    org_root: Path | None = None,
) -> list:
    """Read-only memory toolset for the MAIN agent.

    Searches/reads the caller's private user folder and, when ``org_root`` is
    given, the org-shared folder too.
    """
    return [
        MemoryListTool(user_root, org_root=org_root),
        MemorySearchTool(
            user_root,
            default_datasource=datasource,
            org_root=org_root,
        ),
        MemoryReadTool(user_root, org_root=org_root),
    ]


def memory_agent_tools(
    user_root: Path,
    *,
    datasource: str = "",
    org_root: Path | None = None,
    is_org_admin: bool = False,
    session_id: str = "",
) -> list:
    """Full memory toolset for the wiki memory agent (curator).

    Writes default to the user's private folder. ``write_memory``/``edit_memory``
    with ``shared=true`` target the org folder, which is permitted only when
    ``is_org_admin`` is set and ``org_root`` is configured.
    """
    return [
        MemoryListTool(user_root, org_root=org_root),
        MemorySearchTool(user_root, default_datasource=datasource, org_root=org_root),
        MemoryReadTool(user_root, org_root=org_root),
        MemoryWriteTool(
            user_root,
            org_root=org_root,
            is_org_admin=is_org_admin,
            session_id=session_id,
        ),
        MemoryEditTool(
            user_root,
            org_root=org_root,
            is_org_admin=is_org_admin,
            session_id=session_id,
        ),
    ]


__all__ = [
    "MemoryEditTool",
    "MemoryListTool",
    "MemoryReadTool",
    "MemorySearchTool",
    "MemoryWriteTool",
    "memory_agent_tools",
    "memory_read_tools",
]
