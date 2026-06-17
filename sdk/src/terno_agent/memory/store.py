"""Persistent memory store.

Each memory is a markdown file with YAML-ish frontmatter (matching the
Claude Code auto-memory format) plus a ``.vectors.jsonl`` sidecar that
holds embeddings. Everything lives under
``<workdir>/.terno/memory`` (overridable with ``TERNO_MEMORY_HOME``).
The store synchronizes the markdown file, the ``MEMORY.md`` index, and
the vector record on every write.
"""

from __future__ import annotations

import re
from pathlib import Path
from threading import Lock
from typing import Any

from terno_agent.memory.paths import memory_dir
from terno_agent.memory.types import MemoryEntry, MemoryType
from terno_agent.rag.embeddings import EmbeddingClient, EmbeddingError
from terno_agent.rag.vector_store import FileVectorStore, Hit, VectorStore

INDEX_FILENAME = "MEMORY.md"
VECTOR_FILENAME = ".vectors.jsonl"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:64] or "memory"


class MemoryStore:
    """File-backed memory store with a single ``.terno/memory`` directory."""

    def __init__(
        self,
        workdir: Path,
        embedder: EmbeddingClient | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.workdir = Path(workdir).resolve()
        self.embedder = embedder
        self._dir = memory_dir(self.workdir)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Default to the local JSONL store; callers can inject Milvus (or any
        # other VectorStore) when a shared/remote backend is configured.
        self._vectors = vector_store or FileVectorStore(self._dir / VECTOR_FILENAME)
        self._lock = Lock()

    # ----- I/O on markdown files --------------------------------------- #

    def _file_for(self, name: str) -> Path | None:
        candidate = self._dir / f"{name}.md"
        return candidate if candidate.exists() else None

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
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{entry.name}.md"
        with self._lock:
            path.write_text(self._render_markdown(entry), encoding="utf-8")
            self._rewrite_index()

            if self.embedder is not None:
                try:
                    vector = self.embedder.embed([entry.embedding_text()])[0]
                except EmbeddingError:
                    vector = []
                if vector:
                    self._vectors.upsert(
                        key=entry.name,
                        text=entry.embedding_text(),
                        vector=vector,
                        metadata={
                            "type": entry.type.value,
                            "description": entry.description,
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
            self._vectors.delete(name)
            self._rewrite_index()
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
        if not self._dir.exists():
            return entries
        for md in sorted(self._dir.glob("*.md")):
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
        hits = self._vectors.query(qv, k=k)
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    # ----- index ------------------------------------------------------- #

    def _rewrite_index(self) -> None:
        lines: list[str] = []
        for md in sorted(self._dir.glob("*.md")):
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
        index_path = self._dir / INDEX_FILENAME
        if lines:
            index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            index_path.unlink(missing_ok=True)

    # ----- introspection ---------------------------------------------- #

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def vector_store(self) -> VectorStore:
        return self._vectors

    def describe(self) -> dict[str, Any]:
        return {
            "dir": str(self._dir),
            "count": len(self._vectors),
        }


__all__ = ["INDEX_FILENAME", "VECTOR_FILENAME", "MemoryStore"]
