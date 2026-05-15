"""Orchestrator agent.

The orchestrator is itself a `BaseAgent` whose tools are calls to specialist
sub-agents. The LLM picks which specialist to invoke; we instantiate
specialists lazily on the first call so they can share the orchestrator's
LLM client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from terno_agent.agents.base import AgentRun, BaseAgent
from terno_agent.agents.coder import CoderAgent
from terno_agent.agents.database import DatabaseAgent
from terno_agent.config import Config
from terno_agent.core.exceptions import ConfigError, ToolError
from terno_agent.core.tool import ToolSchema
from terno_agent.db.connection import Database
from terno_agent.llm.base import LLMClient
from terno_agent.llm.factory import create_llm_client
from terno_agent.prompts.orchestrator import ORCHESTRATOR_PROMPT
from terno_agent.sandbox.base import Sandbox
from terno_agent.sandbox.factory import create_sandbox


@dataclass
class _DatabaseDelegateTool:
    agent: DatabaseAgent

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="ask_database_agent",
            description=(
                "Delegate a task to the Database specialist. It can introspect "
                "the schema and run read-only SQL. Returns the specialist's "
                "natural-language answer."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Self-contained task or question for the DB specialist.",
                    }
                },
                "required": ["task"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        task = kwargs.get("task")
        if not task:
            raise ToolError("ask_database_agent requires a 'task' argument.")
        return self.agent.run(task).answer


@dataclass
class _CoderDelegateTool:
    agent: CoderAgent

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="ask_coder_agent",
            description=(
                "Delegate a task to the Coder specialist. It writes Python and "
                "runs it in a sandbox. Pass any rows it needs to operate on "
                "via 'input_data' as a JSON/CSV string — it cannot see prior "
                "tool results."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What the coder should do."},
                    "input_data": {
                        "type": "string",
                        "description": "Optional data the coder should operate on (JSON/CSV/text).",
                    },
                },
                "required": ["task"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        task = kwargs.get("task")
        if not task:
            raise ToolError("ask_coder_agent requires a 'task' argument.")
        data = kwargs.get("input_data")
        full = task if not data else f"{task}\n\nInput data:\n{data}"
        return self.agent.run(full).answer


class Orchestrator(BaseAgent):
    name = "orchestrator"
    max_iterations = 16

    def __init__(
        self,
        llm: LLMClient,
        *,
        database: Database,
        sandbox: Sandbox | None,
        on_event=None,
    ) -> None:
        self.database = database
        self.sandbox = sandbox
        self._db_agent = DatabaseAgent(llm, database, on_event=on_event)
        tools = [_DatabaseDelegateTool(self._db_agent)]
        if sandbox is not None:
            self._coder_agent = CoderAgent(llm, sandbox, on_event=on_event)
            tools.append(_CoderDelegateTool(self._coder_agent))
        else:
            self._coder_agent = None

        system = ORCHESTRATOR_PROMPT
        if sandbox is None:
            system += "\n\nNote: the coder agent is disabled in this session — answer using SQL only."
        super().__init__(llm, system, tools, on_event=on_event)

    # ----- Construction helpers -------------------------------------------------

    @classmethod
    def from_env(cls, on_event=None) -> "Orchestrator":
        return cls.from_config(Config.from_env(), on_event=on_event)

    @classmethod
    def from_config(cls, config: Config, *, on_event=None) -> "Orchestrator":
        if not config.database_url:
            raise ConfigError("TERNO_DATABASE_URL is required.")
        llm = create_llm_client(
            provider=config.llm_provider,
            model=config.llm_model,
            api_key=config.llm_api_key,
        )
        database = Database(config.database_url)
        sandbox: Sandbox | None
        if config.sandbox == "none":
            sandbox = None
        else:
            sandbox = create_sandbox(config.sandbox)
        return cls(llm, database=database, sandbox=sandbox, on_event=on_event)

    # ----- Convenience ---------------------------------------------------------

    def ask(self, question: str) -> AgentRun:
        return self.run(question)
