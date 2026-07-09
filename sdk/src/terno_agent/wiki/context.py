"""Pre-turn memory injection for the main agent.

``MemoryContextProvider`` reads the caller's workspace ``memory`` folder — a
single OKF bundle whose ``index.md`` lists every learned fact — and renders a
compact block. The block is prepended to the main agent's per-turn
``extra_context`` so the agent knows the memory exists and can pull detail with
``read_memory`` / ``search_memory`` (or ``read_file`` / ``grep``).

The memory folder IS the bundle root (facts live directly under it, e.g.
``memory/tables/customer.md``), never under ``.terno``. When ``org_root`` is
set, the organisation-shared memory folder is also shown, in a separate
section.

This is the non-RAG recall path: the index is injected verbatim; there is
no embedding or vector search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from terno_agent.wiki.bundle import KnowledgeBundle
from terno_agent.wiki.tools import DEFAULT_BUNDLE_NAME

_HEADER = "## Available memory (file-based, persists across sessions)"
_ORG_HEADER = (
    "### Organisation-wide shared memory"
    " (read-only unless you are an org admin)"
)
_FOOTER = (
    "This memory is curated automatically each turn — treat it as authoritative "
    "background and prefer it over re-deriving the schema. Apply a "
    "`datasource:<id>` memory only when it matches the database you are "
    "querying; `global` memory always applies. For full detail, read a memory "
    "with `read_memory` / `search_memory` (or `read_file` / `grep`)."
)


@dataclass
class MemoryContextProvider:
    #: The caller's private user memory folder (workspace ``.../memory``).
    user_root: Path
    #: The org-shared memory folder, or None.
    org_root: Path | None = field(default=None)
    #: Display name of the bundle (index title / section header).
    name: str = DEFAULT_BUNDLE_NAME

    def _bundle(self, root: Path | None) -> KnowledgeBundle | None:
        if root is None:
            return None
        bundle = KnowledgeBundle(Path(root).resolve(), name=self.name)
        return bundle if bundle.exists() else None

    def bundles(self) -> list[KnowledgeBundle]:
        bundle = self._bundle(self.user_root)
        return [bundle] if bundle is not None else []

    def org_bundles(self) -> list[KnowledgeBundle]:
        bundle = self._bundle(self.org_root)
        return [bundle] if bundle is not None else []

    def context_block(self) -> str:
        """Return a formatted block, or '' when no memory exists."""
        user_bundle = self._bundle(self.user_root)
        org_bundle = self._bundle(self.org_root)
        if user_bundle is None and org_bundle is None:
            return ""
        lines = [_HEADER, ""]
        if user_bundle is not None:
            index = user_bundle.index_text().strip()
            if index:
                lines.append(index)
            lines.append("")
        if org_bundle is not None:
            lines.append(_ORG_HEADER)
            lines.append("")
            index = org_bundle.index_text().strip()
            if index:
                lines.append(index)
            lines.append("")
        lines.append(_FOOTER)
        return "\n".join(lines).rstrip("\n")


__all__ = ["MemoryContextProvider"]
