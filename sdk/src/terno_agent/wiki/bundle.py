"""On-disk reader/writer for an OKF knowledge bundle.

A bundle is a directory tree of markdown files (one per :class:`Concept`)
plus generated ``index.md`` listings — one at the root and one per
subdirectory — that provide progressive disclosure for a consuming agent.

    bundle = KnowledgeBundle(root, name="sales_db")
    bundle.write_concept(Concept("tables/users", "Users", "table", ...))
    bundle.rebuild_index()
    text = bundle.read_concept("tables/users").render()

The bundle never embeds anything; consumers read it with ordinary file
tools (``read_file`` / ``grep``) or the thin OKF read tools.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

from terno_agent.wiki.concept import Concept, ConceptError

INDEX_FILENAME = "index.md"

# One lock per bundle root, shared across every KnowledgeBundle instance that
# points at the same directory. The background memory curator writes on a
# daemon thread; without this, two overlapping turns (or the main agent
# reading the index while a curator rewrites it) could tear index.md. Locks
# live process-wide keyed by the resolved root path.
_ROOT_LOCKS: dict[str, threading.RLock] = {}
_ROOT_LOCKS_GUARD = threading.Lock()


def _lock_for(root: Path) -> threading.RLock:
    key = str(root)
    with _ROOT_LOCKS_GUARD:
        lock = _ROOT_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _ROOT_LOCKS[key] = lock
        return lock


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + os.replace).

    A reader — in this or another process — either sees the old file or the
    new one, never a partial write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        # Best-effort cleanup; never leak a temp file on failure.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class KnowledgeBundle:
    def __init__(self, root: Path, *, name: str | None = None) -> None:
        self.root = Path(root).resolve()
        self.name = name or self.root.name
        self._lock = _lock_for(self.root)

    # ----- predicates ---------------------------------------------------- #

    def exists(self) -> bool:
        return (self.root / INDEX_FILENAME).exists()

    def _path_for(self, concept_id: str) -> Path:
        rel = concept_id.strip().strip("/")
        return self.root / f"{rel}.md"

    # ----- writes -------------------------------------------------------- #

    def write_concept(self, concept: Concept) -> Path:
        path = self._path_for(concept.concept_id)
        with self._lock:
            _atomic_write(path, concept.render())
        return path

    # ----- reads --------------------------------------------------------- #

    def read_concept(self, concept_id: str) -> Concept | None:
        path = self._path_for(concept_id)
        if not path.exists():
            return None
        try:
            return Concept.parse(concept_id, path.read_text(encoding="utf-8"))
        except (OSError, ConceptError):
            return None

    def list_concepts(self) -> list[Concept]:
        """All concepts in the bundle (generated ``index.md`` files excluded)."""
        concepts: list[Concept] = []
        if not self.root.exists():
            return concepts
        for md in sorted(self.root.rglob("*.md")):
            if md.name == INDEX_FILENAME:
                continue
            concept_id = md.relative_to(self.root).with_suffix("").as_posix()
            try:
                concepts.append(
                    Concept.parse(concept_id, md.read_text(encoding="utf-8"))
                )
            except (OSError, ConceptError):
                continue
        return concepts

    # ----- index generation --------------------------------------------- #

    def rebuild_index(self) -> Path:
        """Regenerate the root ``index.md`` and every subdirectory ``index.md``.

        Serialized against concurrent writers on the same bundle and written
        atomically, so a reader never sees a half-rebuilt index.
        """
        with self._lock:
            concepts = self.list_concepts()
            groups: dict[str, list[Concept]] = {}
            for c in concepts:
                parent = c.concept_id.rsplit("/", 1)[0] if "/" in c.concept_id else ""
                groups.setdefault(parent, []).append(c)

            self.root.mkdir(parents=True, exist_ok=True)
            root_index = self.root / INDEX_FILENAME
            _atomic_write(root_index, self._render_root_index(groups))

            # Per-subdirectory listings (relative links scoped to that dir).
            for parent, items in groups.items():
                if not parent:
                    continue
                sub_index = self.root / parent / INDEX_FILENAME
                _atomic_write(sub_index, self._render_sub_index(parent, items))
            return root_index

    def _render_root_index(self, groups: dict[str, list[Concept]]) -> str:
        lines = [
            "---",
            f"title: {self.name} — knowledge",
            "type: index",
            "---",
            "",
            f"# {self.name}",
            "",
            "Open Knowledge Format bundle. Each entry below links to a concept "
            "document; read it for detail.",
            "",
        ]
        # Root-level concepts first.
        for c in sorted(groups.get("", []), key=lambda c: c.concept_id):
            link = f"{c.concept_id.rsplit('/', 1)[-1]}.md"
            lines.append(self._list_item(c, link))
        if groups.get(""):
            lines.append("")
        # Then each subdirectory as a section.
        for parent in sorted(p for p in groups if p):
            lines.append(f"## {parent}/")
            lines.append("")
            for c in sorted(groups[parent], key=lambda c: c.concept_id):
                link = f"{c.concept_id}.md"
                lines.append(self._list_item(c, link))
            lines.append("")
        return "\n".join(lines).rstrip("\n") + "\n"

    def _render_sub_index(self, parent: str, items: list[Concept]) -> str:
        lines = [
            "---",
            f"title: {parent}",
            "type: index",
            "---",
            "",
            f"# {parent}",
            "",
        ]
        for c in sorted(items, key=lambda c: c.concept_id):
            link = f"{c.concept_id.rsplit('/', 1)[-1]}.md"
            lines.append(self._list_item(c, link))
        return "\n".join(lines).rstrip("\n") + "\n"

    @staticmethod
    def _list_item(concept: Concept, link: str) -> str:
        label = concept.title or concept.concept_id
        suffix = f" — {concept.summary}" if concept.summary else ""
        return f"- [{label}]({link}){suffix}"

    # ----- consumption helper ------------------------------------------- #

    def index_text(self) -> str:
        """Return the root ``index.md`` text (frontmatter stripped), or ''."""
        path = self.root / INDEX_FILENAME
        if not path.exists():
            return ""
        from terno_agent.wiki.frontmatter import parse as parse_frontmatter

        _, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        return body


__all__ = ["INDEX_FILENAME", "KnowledgeBundle"]
