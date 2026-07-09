"""Wiki memory tools.

Two toolsets are exposed:

* The **main agent** gets the READ-ONLY set (``list_memory``, ``search_memory``,
  ``read_memory``) so it can recall facts while answering.
* The **wiki memory agent** (the background curator) gets the full set,
  adding ``write_memory`` and ``edit_memory`` so it can record and refine
  facts after a turn.

Storage uses the OKF bundle engine (``KnowledgeBundle`` / ``Concept``): one
markdown file per fact with a generated ``index.md``. There is exactly ONE
memory bundle per workspace ``memory`` folder — the folder itself is the bundle
root. Learned facts are FLAT files that live directly in that memory directory,
one file per fact (e.g. ``customer.md``, ``active-user.md``), never under
``.terno``, never in a per-datasource subfolder, and never nested in any
subdirectory. Files cross-link to each other by name; the directory stays flat.

* **private** memory → the caller's user folder
  (``.../user_workspace/memory``);
* **shared** memory → the org folder (``.../org_workspace/memory``), which
  **only an org admin may write to**.

Everyone in an org can READ the shared folder; the read tools search the user
folder and the org folder together. A fact's APPLICABILITY to a specific
database is recorded in its ``scope`` frontmatter (``datasource:<id>`` vs
``global``), not by its location.
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

#: Display name of the single memory bundle (used in index titles / headers).
DEFAULT_BUNDLE_NAME = "memory"

# Where a fact came from — recorded in frontmatter as `source` so a reader can
# weigh how much to trust it and a future curator knows what may be stale.
_KNOWN_SOURCES = ("introspection", "query", "conversation", "user", "curator")
_DEFAULT_SOURCE = "curator"

# Memory types from terno-ai plus the datasource-knowledge types.
_KNOWN_TYPES = (
    "user", "feedback", "project", "reference",
    "table", "domain", "metric", "datasource",
)

_SEARCH_DEFAULT_LIMIT = 20
_SNIPPETS_PER_MEMORY = 5


def _utc_now_iso() -> str:
    """Current UTC time as a second-precision ISO string for `updated`."""
    # `timezone.utc` (not `datetime.UTC`) — the latter is Python 3.11+, but this
    # project supports 3.10. noqa keeps ruff UP017 from "modernising" it.
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()  # noqa: UP017


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


def _flat_memory_id(memory_id: str) -> str:
    """Return a safe, FLAT memory id — a single file name, never a subpath.

    Memory is a flat set of files living directly in the memory folder; it must
    never grow subdirectories. Any ``/`` the caller supplies is folded to ``-``
    so a write can only ever create a top-level file, and traversal (absolute
    paths, ``..``, backslashes) is rejected outright.
    """
    mid = (memory_id or "").strip()
    if not mid:
        raise ToolError("memory_id must be non-empty.")
    if mid.startswith("/") or "\\" in mid or ".." in mid.split("/"):
        raise ToolError(f"unsafe memory_id {memory_id!r}.")
    return mid.strip("/").replace("/", "-")


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
class _MemoryToolBase:
    """Shared state: the two memory folders, each a single OKF bundle."""

    user_root: Path
    org_root: Path | None = None
    name: str = DEFAULT_BUNDLE_NAME

    def _bundle(self, root: Path) -> KnowledgeBundle:
        return KnowledgeBundle(Path(root).resolve(), name=self.name)

    def _roots(self) -> list[tuple[Path, bool]]:
        """The (root, shared) folders to read, private first."""
        roots: list[tuple[Path, bool]] = [(self.user_root, False)]
        if self.org_root is not None:
            roots.append((self.org_root, True))
        return roots


@dataclass
class MemoryReadTool(_MemoryToolBase):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_memory",
            description=(
                "Read one memory file. Searches your private memory first, then "
                "organisation-shared memory. memory_id is the file name within "
                "the memory folder without '.md' (e.g. 'active-user', "
                "'customer', 'identity') — a single flat name, never a path."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Flat memory id (no '/'), e.g. 'customer'.",
                    },
                },
                "required": ["memory_id"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        memory_id = (kwargs.get("memory_id") or "").strip()
        if not memory_id:
            raise ToolError("read_memory requires 'memory_id'.")
        memory_id = _flat_memory_id(memory_id)
        for root, _shared in self._roots():
            concept = self._bundle(root).read_concept(memory_id)
            if concept is not None:
                return concept.render()
        raise ToolError(f"No memory {memory_id!r}.")


@dataclass
class MemoryListTool(_MemoryToolBase):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_memory",
            description=(
                "List all memory files (private + organisation-shared) with "
                "their id, title, type and scope."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def run(self, **kwargs: Any) -> str:
        rows: list[dict[str, Any]] = []
        for root, shared in self._roots():
            for c in self._bundle(root).list_concepts():
                rows.append(
                    {
                        "memory_id": c.concept_id,
                        "title": c.title,
                        "type": c.type,
                        "scope": c.metadata.get("scope", "global"),
                        "shared": shared,
                        "summary": c.summary,
                    }
                )
        return json.dumps(rows)


@dataclass
class MemorySearchTool(_MemoryToolBase):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="search_memory",
            description=(
                "Search memory files for a term or regex (case-insensitive) "
                "across both private and organisation-shared memory. Scans "
                "titles, summaries, and bodies and returns the matching "
                "memories with the lines that matched. Use this to find where "
                "relevant knowledge lives, then read_memory the returned "
                "memory_ids for detail."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text or regex (matched case-insensitively).",
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

        hits: list[dict[str, Any]] = []
        for root, shared in self._roots():
            for concept in self._bundle(root).list_concepts():
                snippets = _match(rx, concept)
                if not snippets:
                    continue
                hits.append(
                    {
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


@dataclass
class MemoryWriteTool(_MemoryToolBase):
    is_org_admin: bool = False
    session_id: str = ""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_memory",
            description=(
                "Create a NEW memory file (or fully replace one), then "
                "regenerate the index. Use for a durable fact that has no file "
                "yet: a metric/term definition, a business rule, an enum "
                "decoding, a join path, or a stable user preference. To ADD to "
                "or correct an EXISTING memory, use edit_memory instead. "
                "memory_id is the file name without '.md' (e.g. "
                "'active-user') — a single flat name, never a nested path; "
                "memory files are never placed in subdirectories. Set "
                "shared=true to store the fact in "
                "organisation-shared memory (org admins only); otherwise it is "
                "saved as your private memory. Records provenance automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Flat memory id (no '/'), e.g. 'active-user'.",
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
                "required": ["memory_id", "title", "type", "scope"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        memory_id = (kwargs.get("memory_id") or "").strip()
        title = (kwargs.get("title") or "").strip()
        type_ = (kwargs.get("type") or "").strip()
        scope = (kwargs.get("scope") or "").strip()
        if not (memory_id and title and type_ and scope):
            raise ToolError(
                "write_memory requires 'memory_id', 'title', 'type', and 'scope'."
            )
        memory_id = _flat_memory_id(memory_id)
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
        bundle = self._bundle(root)
        path = bundle.write_concept(concept)
        bundle.rebuild_index()
        return json.dumps(
            {"memory_id": memory_id, "shared": shared, "path": str(path)}
        )


@dataclass
class MemoryEditTool(_MemoryToolBase):
    """Targeted, additive edits to an EXISTING memory file."""

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
                    "memory_id": {
                        "type": "string",
                        "description": "Existing flat memory id, e.g. 'identity'.",
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
                "required": ["memory_id"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        memory_id = (kwargs.get("memory_id") or "").strip()
        if not memory_id:
            raise ToolError("edit_memory requires 'memory_id'.")
        memory_id = _flat_memory_id(memory_id)

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
        bundle = self._bundle(root)
        concept = bundle.read_concept(memory_id)
        if concept is None:
            raise ToolError(
                f"No memory {memory_id!r} to edit. Use write_memory to create it."
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
    org_root: Path | None = None,
    name: str = DEFAULT_BUNDLE_NAME,
) -> list:
    """Read-only memory toolset for the MAIN agent.

    Searches/reads the caller's private user folder and, when ``org_root`` is
    given, the org-shared folder too.
    """
    return [
        MemoryListTool(user_root, org_root=org_root, name=name),
        MemorySearchTool(user_root, org_root=org_root, name=name),
        MemoryReadTool(user_root, org_root=org_root, name=name),
    ]


def memory_agent_tools(
    user_root: Path,
    *,
    org_root: Path | None = None,
    is_org_admin: bool = False,
    session_id: str = "",
    name: str = DEFAULT_BUNDLE_NAME,
) -> list:
    """Full memory toolset for the wiki memory agent (curator).

    Writes default to the user's private folder. ``write_memory``/``edit_memory``
    with ``shared=true`` target the org folder, which is permitted only when
    ``is_org_admin`` is set and ``org_root`` is configured.
    """
    return [
        MemoryListTool(user_root, org_root=org_root, name=name),
        MemorySearchTool(user_root, org_root=org_root, name=name),
        MemoryReadTool(user_root, org_root=org_root, name=name),
        MemoryWriteTool(
            user_root,
            org_root=org_root,
            name=name,
            is_org_admin=is_org_admin,
            session_id=session_id,
        ),
        MemoryEditTool(
            user_root,
            org_root=org_root,
            name=name,
            is_org_admin=is_org_admin,
            session_id=session_id,
        ),
    ]


__all__ = [
    "DEFAULT_BUNDLE_NAME",
    "MemoryEditTool",
    "MemoryListTool",
    "MemoryReadTool",
    "MemorySearchTool",
    "MemoryWriteTool",
    "memory_agent_tools",
    "memory_read_tools",
]
