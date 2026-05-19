"""The single Terno agent.

`TernoAgent` is a `BaseAgent` wired with the standard Terno toolset
(file ops, bash, task list, subagent spawner) and the canonical
`SYSTEM_PROMPT`. Subagents spawned via the `spawn_agent` tool are also
`TernoAgent` instances — they share the same LLM, task store, and
working directory, but get a caller-supplied system prompt.
"""

from __future__ import annotations

import sys
from pathlib import Path

from terno_agent.agents.base import AgentRun, BaseAgent
from terno_agent.config import Config
from terno_agent.core.exceptions import ConfigError, SandboxError
from terno_agent.llm.base import LLMClient
from terno_agent.llm.factory import create_llm_client
from terno_agent.prompts.prompt import SYSTEM_PROMPT
from terno_agent.sandbox.base import Sandbox
from terno_agent.sandbox.factory import create_sandbox
from terno_agent.tools.code_exec import RunPythonTool
from terno_agent.tools.files import EditFileTool, ReadFileTool, WriteFileTool
from terno_agent.tools.shell import BashTool
from terno_agent.tools.subagent import SpawnAgentTool
from terno_agent.tools.tasks import (
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskStore,
    TaskUpdateTool,
)


class TernoAgent(BaseAgent):
    name = "terno"
    max_iterations = 32

    def __init__(
        self,
        llm: LLMClient,
        *,
        system_prompt: str | None = None,
        workdir: Path | None = None,
        task_store: TaskStore | None = None,
        sandbox: Sandbox | None = None,
        bash_timeout_s: int = 120,
        run_python_timeout_s: int = 30,
        on_event=None,
    ) -> None:
        self.workdir = (workdir or Path.cwd()).resolve()
        self.task_store = task_store or TaskStore()
        self.sandbox = sandbox

        tools: list = [
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            BashTool(workdir=self.workdir, default_timeout_s=bash_timeout_s),
            TaskCreateTool(self.task_store),
            TaskListTool(self.task_store),
            TaskGetTool(self.task_store),
            TaskUpdateTool(self.task_store),
            SpawnAgentTool(
                llm=llm,
                workdir=self.workdir,
                task_store=self.task_store,
                sandbox=sandbox,
                bash_timeout_s=bash_timeout_s,
                run_python_timeout_s=run_python_timeout_s,
                on_event=on_event,
            ),
        ]
        if sandbox is not None:
            tools.append(RunPythonTool(sandbox, timeout_s=run_python_timeout_s))

        super().__init__(
            llm,
            system_prompt or SYSTEM_PROMPT,
            tools,
            on_event=on_event,
        )

    # ----- Construction helpers ----------------------------------------- #

    @classmethod
    def from_env(cls, *, on_event=None) -> TernoAgent:
        return cls.from_config(Config.from_env(), on_event=on_event)

    @classmethod
    def from_config(cls, config: Config, *, on_event=None) -> TernoAgent:
        if not config.llm_api_key:
            raise ConfigError(
                "No LLM API key configured. Set ANTHROPIC_API_KEY or "
                "OPENAI_API_KEY (or TERNO_LLM_API_KEY)."
            )
        llm = create_llm_client(
            provider=config.llm_provider,
            model=config.llm_model,
            api_key=config.llm_api_key,
        )
        sandbox: Sandbox | None = None
        if config.sandbox != "none":
            try:
                sandbox = create_sandbox(config.sandbox)
            except SandboxError as exc:
                print(
                    f"warning: sandbox {config.sandbox!r} unavailable ({exc}); "
                    "run_python tool will be disabled. Set TERNO_SANDBOX=none "
                    "to silence this warning, or TERNO_SANDBOX=local to use "
                    "the local subprocess sandbox.",
                    file=sys.stderr,
                )
        return cls(llm, sandbox=sandbox, on_event=on_event)

    # ----- Convenience --------------------------------------------------- #

    def ask(self, task: str) -> AgentRun:
        return self.run(task)
