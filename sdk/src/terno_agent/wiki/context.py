"""Pre-turn memory injection for the main agent.

``MemoryContextProvider`` scans the on-disk memory bundles under
``<workdir>/.terno/knowledge`` and renders a compact block listing what
memory is available. The block is prepended to the main agent's per-turn
``extra_context`` so the agent knows the memory exists and can pull detail
with ``read_memory`` / ``search_memory`` (or ``read_file`` / ``grep``).

This is the non-RAG recall path: the index is injected verbatim; there is
no embedding or vector search.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from terno_agent.wiki.bundle import KnowledgeBundle
from terno_agent.wiki.paths import knowledge_root

_HEADER = "## Available memory (file-based, persists across sessions)"
_FOOTER = (
    "This memory is curated automatically each turn — treat it as authoritative "
    "background and prefer it over re-deriving the schema. Apply a "
    "`datasource:<id>` memory only when it matches the database you are "
    "querying; `global` memory always applies. For full detail, read a memory "
    "with `read_memory` / `search_memory` (or `read_file` / `grep`)."
)


@dataclass(slots=True)
class MemoryContextProvider:
    workdir: Path

    def bundles(self) -> list[KnowledgeBundle]:
        root = knowledge_root(self.workdir)
        if not root.exists():
            return []
        found: list[KnowledgeBundle] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            bundle = KnowledgeBundle(child, name=child.name)
            if bundle.exists():
                found.append(bundle)
        return found

    def context_block(self) -> str:
        """Return a formatted block, or '' when no bundles exist."""
        bundles = self.bundles()
        if not bundles:
            return ""
        lines = [_HEADER, ""]
        for bundle in bundles:
            lines.append(f"### `{bundle.name}`")
            index = bundle.index_text().strip()
            if index:
                lines.append(index)
            lines.append("")
        lines.append(_FOOTER)
        return "\n".join(lines).rstrip("\n")


__all__ = ["MemoryContextProvider"]
