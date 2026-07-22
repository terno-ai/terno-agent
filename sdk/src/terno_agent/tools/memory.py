"""Per-turn memory context — the dynamic half of terno-ai's memory design.

Memory here works exactly like terno-ai: the agent reads and writes its memory
with the ordinary file tools (``read_file`` / ``write_file`` / ``edit_file`` /
``grep``), and the whole memory *protocol* — the frontmatter shape, the
``MEMORY.md`` upkeep, types, scope, and dedup rules — lives in the system
prompt. There is deliberately NO memory-specific tool and NO storage engine.

This module supplies only the *dynamic* context that can't live in a static
prompt: where the memory folders are on disk this session, what their
``MEMORY.md`` indexes currently contain, and the current session id (stamped as
``originSessionId`` when the agent creates a memory). It is injected into the
main agent's per-turn context so the prompt stays stable and cacheable.

Mirrors ``terno/agent/memory.py`` in the terno-ai repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: The index file the agent maintains at the root of each memory folder.
MEMORY_INDEX_FILENAME = "MEMORY.md"


def read_memory_index(memory_dir: Path | str) -> str | None:
    """Return the stripped contents of ``<memory_dir>/MEMORY.md`` or ``None``."""
    path = Path(memory_dir) / MEMORY_INDEX_FILENAME
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace").strip() or None
    except OSError:
        return None
    return None


_FOOTER = (
    "Treat these facts as authoritative background and prefer them over "
    "re-deriving the schema. Apply a `datasource:<id>` memory only when it "
    "matches the database you are querying; `global` memory always applies. "
    "Open a memory file with `read_file` (or `grep` the folder) for full detail, "
    "and record durable new facts there per the Memory section of your "
    "instructions."
)


@dataclass
class MemoryContextProvider:
    """Builds the per-turn memory context block for the main agent.

    ``user_root`` is the caller's private memory folder; ``org_root`` (optional)
    is the organisation-shared folder. The block always names the folders — so
    the agent knows where to write its first memory — and appends each
    ``MEMORY.md`` index when it exists.
    """

    user_root: Path
    org_root: Path | None = field(default=None)
    session_id: str = ""

    def context_block(self) -> str:
        sections: list[str] = []

        user_index = read_memory_index(self.user_root)
        sections.append(
            "Your private memory (read/write) lives in "
            "`/workspace/user_workspace/memory` inside the sandbox.\n"
            "Its MEMORY.md index:\n\n"
            + (user_index or "(empty — no memories saved yet)")
        )

        if self.org_root is not None:
            org_index = read_memory_index(self.org_root)
            sections.append(
                "Organisation-shared memory lives in "
                "`/workspace/org_workspace/memory` inside the sandbox "
                "(read-only unless you are an org admin).\n"
                "Its MEMORY.md index:\n\n"
                + (org_index or "(empty — no shared memories yet)")
            )

        if self.session_id:
            sections.append(
                f"currentSessionId: {self.session_id}\n"
                "(use this as `originSessionId` when creating a new memory)"
            )

        header = "## Persistent memory (file-based, survives across sessions)"
        return f"{header}\n\n" + "\n\n".join(sections) + "\n\n" + _FOOTER


__all__ = [
    "MEMORY_INDEX_FILENAME",
    "MemoryContextProvider",
    "read_memory_index",
]
