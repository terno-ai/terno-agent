from __future__ import annotations

from terno_agent.agents.base import BaseAgent
from terno_agent.llm.base import LLMClient
from terno_agent.prompts.coder import CODER_PROMPT
from terno_agent.sandbox.base import Sandbox
from terno_agent.tools.code_exec import RunPythonTool


class CoderAgent(BaseAgent):
    name = "coder"

    def __init__(
        self,
        llm: LLMClient,
        sandbox: Sandbox,
        *,
        timeout_s: int = 30,
        on_event=None,
    ) -> None:
        tools = [RunPythonTool(sandbox, timeout_s=timeout_s)]
        super().__init__(llm, CODER_PROMPT, tools, on_event=on_event)
