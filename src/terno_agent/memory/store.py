"""Persistent memory store.

Each memory is a markdown file with YAML-ish frontmatter (matching the
Claude Code auto-memory format) plus a ``.vectors.jsonl`` sidecar that
holds embeddings. Two physical scopes:

* **global** — ``~/.terno_agent/memory/`` (``user``, ``feedback``)
* **workdir** — ``<workdir>/.terno/memory/`` (``project``, ``reference``)

The store synchronizes the markdown file, the ``MEMORY.md`` index, and
the vector record on every write.
"""

from __future__ import annotations

import re
from pathlib import Path
from threading import Lock
from typing import Any

from terno_agent.memory.paths import (
    global_memory_dir,
    resolve_dir_for_type,
    workdir_memory_dir,
)
from terno_agent.memory.types import MemoryEntry, MemoryScope, MemoryType
from terno_agent.rag.embeddings import EmbeddingClient, EmbeddingError
from terno_agent.rag.vector_store import FileVectorStore, Hit

INDEX_FILENAME = "MEMORY.md"
VECTOR_FILENAME = ".vectors.jsonl"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:64] or "memory"


class MemoryStore:
    """File-backed memory store with two scopes and per-scope vector indexes."""

    def __init__(
        self,
        workdir: Path,
        embedder: EmbeddingClient | None = None,
    ) -> None:
        self.workdir = Path(workdir).resolve()
        self.embedder = embedder
        self._global_dir = global_memory_dir()
        self._workdir_dir = workdir_memory_dir(self.workdir)
        for d in (self._global_dir, self._workdir_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._vectors: dict[MemoryScope, FileVectorStore] = {
            MemoryScope.GLOBAL: FileVectorStore(self._global_dir / VECTOR_FILENAME),
            MemoryScope.WORKDIR: FileVectorStore(self._workdir_dir / VECTOR_FILENAME),
        }
        self._lock = Lock()

    # ----- scope helpers ----------------------------------------------- #

    def dir_for(self, type_: MemoryType) -> Path:
        return resolve_dir_for_type(type_, self.workdir)

    def vector_store_for(self, scope: MemoryScope) -> FileVectorStore:
        return self._vectors[scope]

    @property
    def vector_stores(self) -> list[FileVectorStore]:
        return [self._vectors[MemoryScope.GLOBAL], self._vectors[MemoryScope.WORKDIR]]

    # ----- I/O on markdown files --------------------------------------- #

    def _file_for(self, name: str) -> Path | None:
        """Find which scope dir holds the memory file for ``name``."""
        candidate_global = self._global_dir / f"{name}.md"
        if candidate_global.exists():
            return candidate_global
        candidate_workdir = self._workdir_dir / f"{name}.md"
        if candidate_workdir.exists():
            return candidate_workdir
        return None

    @staticmethod
    def _render_markdown(entry: MemoryEntry) -> str:
        return (
            "---\n"
            f"name: {entry.name}\n"
            f"description: {entry.description}\n"
            "metadata:\n"
            f"  type: {entry.type.value}\n"
            "---\n\n"
            f"{entry.body.rstrip()}\n"
        )

    @staticmethod
    def _parse_markdown(text: str) -> MemoryEntry | None:
        if not text.startswith("---"):
            return None
        try:
            _, frontmatter, body = text.split("---", 2)
        except ValueError:
            return None
        meta: dict[str, str] = {}
        type_str = ""
        in_metadata = False
        for raw in frontmatter.splitlines():
            line = raw.rstrip()
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            if not line.startswith(" "):
                in_metadata = False
            if line.startswith("metadata:"):
                in_metadata = True
                continue
            if in_metadata:
                if ":" in line:
                    key, _, value = line.strip().partition(":")
                    if key.strip() == "type":
                        type_str = value.strip()
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        name = meta.get("name", "").strip()
        description = meta.get("description", "").strip()
        if not name or not type_str:
            return None
        try:
            type_ = MemoryType(type_str)
        except ValueError:
            return None
        return MemoryEntry(
            name=name,
            description=description,
            type=type_,
            body=body.strip("\n"),
        )

    # ----- CRUD -------------------------------------------------------- #

    def save(self, entry: MemoryEntry) -> Path:
        """Write the markdown file, update the index, and refresh the vector."""
        if not _NAME_RE.match(entry.name):
            entry = MemoryEntry(
                name=_slugify(entry.name),
                description=entry.description,
                type=entry.type,
                body=entry.body,
            )
        target_dir = self.dir_for(entry.type)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{entry.name}.md"
        with self._lock:
            # If a memory with this name exists in the *other* scope (e.g.
            # type changed), remove the old file so we don't end up with two.
            other = self._file_for(entry.name)
            if other is not None and other != path:
                other.unlink(missing_ok=True)
                for vs in self._vectors.values():
                    vs.delete(entry.name)

            path.write_text(self._render_markdown(entry), encoding="utf-8")
            self._rewrite_index(entry.scope)

            if self.embedder is not None:
                try:
                    vector = self.embedder.embed([entry.embedding_text()])[0]
                except EmbeddingError:
                    vector = []
                if vector:
                    self._vectors[entry.scope].upsert(
                        key=entry.name,
                        text=entry.embedding_text(),
                        vector=vector,
                        metadata={
                            "type": entry.type.value,
                            "description": entry.description,
                            "scope": entry.scope.value,
                            "path": str(path),
                        },
                    )
        return path

    def delete(self, name: str) -> bool:
        with self._lock:
            path = self._file_for(name)
            if path is None:
                return False
            path.unlink(missing_ok=True)
            for vs in self._vectors.values():
                vs.delete(name)
            # Rewrite both indexes — we don't know which scope it lived in.
            for scope in MemoryScope:
                self._rewrite_index(scope)
            return True

    def read(self, name: str) -> MemoryEntry | None:
        path = self._file_for(name)
        if path is None:
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        return self._parse_markdown(text)

    def list_all(self, type_: MemoryType | None = None) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        for d in (self._global_dir, self._workdir_dir):
            if not d.exists():
                continue
            for md in sorted(d.glob("*.md")):
                if md.name == INDEX_FILENAME:
                    continue
                try:
                    entry = self._parse_markdown(md.read_text(encoding="utf-8"))
                except OSError:
                    continue
                if entry is None:
                    continue
                if type_ is not None and entry.type is not type_:
                    continue
                entries.append(entry)
        return entries

    # ----- retrieval --------------------------------------------------- #

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if self.embedder is None or not query.strip():
            return []
        try:
            vectors = self.embedder.embed([query])
        except EmbeddingError:
            return []
        if not vectors:
            return []
        qv = vectors[0]
        merged: list[Hit] = []
        for vs in self._vectors.values():
            merged.extend(vs.query(qv, k=k))
        merged.sort(key=lambda h: h.score, reverse=True)
        return merged[:k]

    # ----- index ------------------------------------------------------- #

    def _rewrite_index(self, scope: MemoryScope) -> None:
        scope_dir = self._global_dir if scope is MemoryScope.GLOBAL else self._workdir_dir
        lines: list[str] = []
        for md in sorted(scope_dir.glob("*.md")):
            if md.name == INDEX_FILENAME:
                continue
            try:
                entry = self._parse_markdown(md.read_text(encoding="utf-8"))
            except OSError:
                continue
            if entry is None:
                continue
            desc = entry.description or "(no description)"
            lines.append(f"- [{entry.name}]({md.name}) — {desc}")
        index_path = scope_dir / INDEX_FILENAME
        if lines:
            index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            index_path.unlink(missing_ok=True)

    # ----- introspection ---------------------------------------------- #

    @property
    def global_dir(self) -> Path:
        return self._global_dir

    @property
    def workdir_dir(self) -> Path:
        return self._workdir_dir

    def describe(self) -> dict[str, Any]:
        return {
            "global_dir": str(self._global_dir),
            "workdir_dir": str(self._workdir_dir),
            "count_global": len(self._vectors[MemoryScope.GLOBAL]),
            "count_workdir": len(self._vectors[MemoryScope.WORKDIR]),
        }


__all__ = ["INDEX_FILENAME", "VECTOR_FILENAME", "MemoryStore"]
