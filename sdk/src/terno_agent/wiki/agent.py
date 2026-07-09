"""The wiki memory agent — a per-turn background memory curator.

``MemoryAgent`` is a small, self-contained agent loop (its own system prompt,
its own tools, its own history) that the main assistant runs once per turn,
AFTER it has answered. Given the finished turn as evidence and the current
state of the memory bundle, it decides whether to write or edit memory files —
without touching the main assistant's conversation or loop.

It is deliberately built on ``BaseAgent`` (not ``TernoAgent``) so it gets only
its curated memory toolset and never recurses into another agent. The main
assistant only ever READS memory; this agent is the only writer.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from terno_agent.agents.base import AgentRun, BaseAgent
from terno_agent.core.messages import AssistantMessage, SystemMessage, UserMessage
from terno_agent.wiki.bundle import KnowledgeBundle
from terno_agent.wiki.prompts import MEMORY_AGENT_PROMPT
from terno_agent.wiki.tools import memory_agent_tools

if TYPE_CHECKING:
    from terno_agent.agents.base import Trace
    from terno_agent.core.cancel import CancelToken
    from terno_agent.llm.base import LLMClient


class MemoryAgent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        user_root: Path,
        datasource: str,
        org_root: Path | None = None,
        is_org_admin: bool = False,
        session_id: str = "",
        max_iterations: int = 8,
        cancel_token: CancelToken | None = None,
    ) -> None:
        self.user_root = Path(user_root).resolve()
        self.org_root = Path(org_root).resolve() if org_root is not None else None
        self.is_org_admin = is_org_admin
        # ``datasource`` is only the memory bundle's display name now — the
        # folder itself is the single bundle (no per-datasource subdir).
        self.datasource = datasource
        tools = memory_agent_tools(
            self.user_root,
            org_root=self.org_root,
            is_org_admin=is_org_admin,
            session_id=session_id,
            name=datasource,
        )
        self._agent = BaseAgent(
            llm,
            MEMORY_AGENT_PROMPT,
            tools,
            cancel_token=cancel_token,
        )
        self._agent.max_iterations = max_iterations

    # ----- per-turn entry points ---------------------------------------- #

    def run_turn(
        self,
        user_task: str,
        *,
        assistant_answer: str | None = None,
        queries: list[str] | None = None,
    ) -> AgentRun:
        """Curate memory from one finished turn. Isolated from the main loop.

        ``user_task`` plus the optional ``assistant_answer`` and ``queries``
        that executed are passed as evidence. The curator distils durable facts
        from them; it does not archive the conversation.
        """
        bundle = KnowledgeBundle(self.user_root, name=self.datasource)
        exists = bundle.exists()
        state = bundle.index_text().strip() if exists else "(no bundle yet)"
        if self.org_root is not None:
            org_bundle = KnowledgeBundle(self.org_root, name=self.datasource)
            if org_bundle.exists():
                shared_state = org_bundle.index_text().strip()
                if shared_state:
                    state = (
                        f"{state}\n\n[organisation-shared memory]\n{shared_state}"
                    )
        evidence = [f'The user asked the main assistant:\n"{user_task}"']
        if assistant_answer:
            evidence.append(f"The assistant answered:\n{assistant_answer.strip()}")
        if queries:
            joined = "\n".join(f"- {q.strip()}" for q in queries if q.strip())
            if joined:
                evidence.append(f"SQL that ran this turn:\n{joined}")
        task = (
            "\n\n".join(evidence) + "\n\n"
            f"Datasource / memory bundle: {self.datasource}\n"
            f"Bundle exists: {exists}\n"
            f"Current memory index:\n{state}\n\n"
            "From this evidence, decide what durable memory (if any) is worth "
            "recording so similar questions are answered better next time, then "
            "act using your tools. If memory already suffices, make no changes."
        )
        # Fresh history each turn keeps this bounded and focused.
        self._agent.clear_history()
        return self._agent.run(task)

    def curate_async(
        self,
        user_task: str,
        *,
        assistant_answer: str | None = None,
        queries: list[str] | None = None,
    ) -> threading.Thread:
        """Fire-and-forget: run ``run_turn`` in a daemon thread.

        The user is never blocked, and any failure is swallowed so memory
        curation can never break the main flow.
        """

        def _safe() -> None:
            try:
                self.run_turn(
                    user_task,
                    assistant_answer=assistant_answer,
                    queries=queries,
                )
            except Exception:  # never break the host on a curation failure
                pass

        thread = threading.Thread(target=_safe, name="wiki-memory-agent", daemon=True)
        thread.start()
        return thread

    # ----- introspection ------------------------------------------------ #

    @property
    def usage(self):
        return self._agent.usage


def latest_exchange(trace: Trace) -> tuple[str, str]:
    """Pull the last user task and final assistant answer out of a trace.

    Returns ``("", "")`` when the trace has no usable exchange. Kept simple:
    memory is distilled from what the user asked and what the assistant
    concluded, not from every intermediate tool call.
    """
    user_task = ""
    assistant_answer = ""
    for msg in trace:
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, UserMessage):
            content = msg.content
            if isinstance(content, str):
                user_task = content
            else:
                parts = [getattr(p, "text", "") for p in content]
                user_task = "\n\n".join(t for t in parts if t)
        elif isinstance(msg, AssistantMessage):
            if msg.content and msg.content.strip():
                assistant_answer = msg.content.strip()
    return user_task, assistant_answer


__all__ = ["MemoryAgent", "latest_exchange"]
