"""Pre-turn knowledge injection for the main agent.

`KnowledgeContextProvider` scans the on-disk knowledge bundles under
``<workdir>/.terno/knowledge`` and renders a compact block listing what
datasource knowledge is available. The block is prepended to the main
agent's per-turn ``extra_context`` (alongside memory recall) so the agent
knows the knowledge exists and can pull detail with ``read_concept`` /
``grep`` / ``read_file``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from terno_agent.wiki.bundle import KnowledgeBundle
from terno_agent.wiki.paths import knowledge_root

_HEADER = "## Datasource knowledge (Open Knowledge Format)"
_FOOTER = (
    "This knowledge is curated automatically each turn — treat it as "
    "authoritative background and prefer it over re-deriving the schema. For "
    "full detail, read the concept files under the path above with "
    "`read_file` / `grep`."
)


@dataclass(slots=True)
class KnowledgeContextProvider:
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
            lines.append(f"### `{bundle.name}` — {bundle.root}")
            index = bundle.index_text().strip()
            if index:
                lines.append(index)
            lines.append("")
        lines.append(_FOOTER)
        return "\n".join(lines).rstrip("\n")


__all__ = ["KnowledgeContextProvider"]
