"""Tool that spawns a fresh `TernoAgent` as a subagent.

The subagent shares the parent's LLM client, working directory, task
store, and event hook, but starts with a caller-supplied system prompt
and no message history. The tool returns the subagent's final answer
as plain text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from terno_agent.core.cancel import CancelToken
from terno_agent.core.events import EventHook
from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema
from terno_agent.llm.base import LLMClient
from terno_agent.sandbox.base import Sandbox

if TYPE_CHECKING:
    from terno_agent.core.hooks import PreToolUseHook
    from terno_agent.mcp.manager import McpManager
    from terno_agent.tools.ask_user import AskCallback
    from terno_agent.tools.tasks import TaskStore


@dataclass
class SpawnAgentTool:
    llm: LLMClient
    workdir: Path
    task_store: TaskStore
    sandbox: Sandbox | None = None
    mcp_manager: McpManager | None = None
    bash_timeout_s: int = 120
    run_python_timeout_s: int = 30
    max_iterations: int | None = None
    on_event: EventHook | None = None
    cancel_token: CancelToken | None = None
    ask_callback: AskCallback | None = None
    permission_hook: PreToolUseHook | None = None

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="Agent",
            # Ported from the reference harness, keeping its "When to use"
            # guidance verbatim. Dropped: agent-type registry / subagent_type,
            # SendMessage continuation, worktree+remote isolation, per-call model
            # and effort overrides, and background execution with notifications —
            # Terno has none of those, and every one of them named a tool or
            # parameter that does not exist here.
            description=(
                "Launch a new agent to handle complex, multi-step tasks. The"
                " agent gets a system prompt you write and the same tools you"
                " have (recursively), and returns its final answer as a string."
                " It runs synchronously.\n"
                "\n"
                "## When to use\n"
                "\n"
                "Reach for this when you have independent work to run in"
                " parallel, or when answering would mean reading across several"
                " files — delegate it and you keep the conclusion, not the file"
                " dumps. For a single-fact lookup where you already know the"
                " file, symbol, or value, search directly. Once you've delegated"
                " a search, don't also run it yourself — wait for the result.\n"
                "\n"
                "- The agent's final report is not shown to the user — relay what"
                " matters.\n"
                "- The agent does not see your conversation, so `prompt` and"
                " `task` must be self-contained."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "System prompt for the subagent. Describes its "
                            "role, scope, and any rules it should follow."
                        ),
                    },
                    "task": {
                        "type": "string",
                        "description": (
                            "The initial user-message task for the subagent. "
                            "If omitted, the prompt is used as the task."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        prompt = (kwargs.get("prompt") or "").strip()
        if not prompt:
            raise ToolError("spawn_agent requires a 'prompt' argument.")
        task = (kwargs.get("task") or prompt).strip()

        from terno_agent.agents.terno import TernoAgent

        subagent = TernoAgent(
            self.llm,
            system_prompt=prompt,
            workdir=self.workdir,
            task_store=self.task_store,
            sandbox=self.sandbox,
            mcp_manager=self.mcp_manager,
            bash_timeout_s=self.bash_timeout_s,
            run_python_timeout_s=self.run_python_timeout_s,
            max_iterations=self.max_iterations,
            on_event=self.on_event,
            cancel_token=self.cancel_token,
            ask_callback=self.ask_callback,
            permission_hook=self.permission_hook,
        )
        result = subagent.run(task)
        return result.answer


__all__ = ["SpawnAgentTool"]
