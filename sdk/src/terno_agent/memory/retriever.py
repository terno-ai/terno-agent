"""Pre-turn memory retrieval.

Embeds the user's incoming message, fetches top-k relevant memories from
both scopes, and formats them into a block ready to drop into
``BaseAgent.run(extra_context=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from terno_agent.memory.store import MemoryStore

_HEADER = "## Relevant memories (recalled from past sessions)"


@dataclass(slots=True)
class MemoryRetriever:
    store: MemoryStore
    k: int = 5

    def fetch_relevant(self, user_message: str) -> str:
        """Return a formatted block or empty string if no hits."""
        if not user_message.strip():
            return ""
        hits = self.store.search(user_message, k=self.k)
        if not hits:
            return ""
        lines = [_HEADER, ""]
        for h in hits:
            type_ = h.metadata.get("type", "?")
            desc = h.metadata.get("description") or ""
            lines.append(f"- [{h.key}] ({type_}) {desc}".rstrip())
        lines.append("")
        lines.append(
            "Treat these as background — they may be out of date. Call "
            "search_memory or read_memory for the full body if any look "
            "relevant."
        )
        return "\n".join(lines)


__all__ = ["MemoryRetriever"]
