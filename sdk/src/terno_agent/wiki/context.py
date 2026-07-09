"""Pre-turn memory injection for the main agent.

``MemoryContextProvider`` scans the on-disk memory bundles in the caller's
workspace ``memory`` folder and renders a compact block listing what memory is
available. The block is prepended to the main agent's per-turn
``extra_context`` so the agent knows the memory exists and can pull detail
with ``read_memory`` / ``search_memory`` (or ``read_file`` / ``grep``).

Memory bundles are OKF bundles that live *directly* under the memory folder
(``<memory>/<datasource>/index.md``), NOT under ``.terno``. When ``org_root``
is set, organisation-shared bundles are also scanned from that folder and
shown in a separate section.

This is the non-RAG recall path: the index is injected verbatim; there is
no embedding or vector search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from terno_agent.wiki.bundle import KnowledgeBundle

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
    #: The caller's private user memory folder (workspace ``.../<user>/memory``).
    user_root: Path
    #: The org-shared memory folder (workspace ``.../<org>/memory``), or None.
    org_root: Path | None = field(default=None)

    def _scan_root(self, root: Path | None) -> list[KnowledgeBundle]:
        if root is None or not root.exists():
            return []
        found: list[KnowledgeBundle] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            bundle = KnowledgeBundle(child, name=child.name)
            if bundle.exists():
                found.append(bundle)
        return found

    def bundles(self) -> list[KnowledgeBundle]:
        return self._scan_root(self.user_root)

    def org_bundles(self) -> list[KnowledgeBundle]:
        return self._scan_root(self.org_root)

    def context_block(self) -> str:
        """Return a formatted block, or '' when no bundles exist."""
        bundles = self.bundles()
        org_bundles = self.org_bundles()
        if not bundles and not org_bundles:
            return ""
        lines = [_HEADER, ""]
        for bundle in bundles:
            lines.append(f"### `{bundle.name}`")
            index = bundle.index_text().strip()
            if index:
                lines.append(index)
            lines.append("")
        if org_bundles:
            lines.append(_ORG_HEADER)
            lines.append("")
            for bundle in org_bundles:
                lines.append(f"#### `{bundle.name}` (shared)")
                index = bundle.index_text().strip()
                if index:
                    lines.append(index)
                lines.append("")
        lines.append(_FOOTER)
        return "\n".join(lines).rstrip("\n")


__all__ = ["MemoryContextProvider"]
