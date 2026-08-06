"""System-prompt assembly.

The prompt is built as an ordered list of cacheable blocks (see `builder`)
rather than one string. `SYSTEM_PROMPT` remains available as a lazily-rendered
flat string for callers that haven't moved to blocks yet.
"""

from terno_agent.prompts.builder import (
    SystemBlock,
    SystemPrompt,
    build_system_prompt,
    render_session_block,
    render_system_prompt,
)
from terno_agent.prompts.context import GitSnapshot, PromptContext
from terno_agent.prompts.prompt import TOOL_GUIDE


def __getattr__(name: str) -> object:
    # Rendering probes git and the filesystem, so defer it until first use
    # instead of paying for it at import time.
    if name == "SYSTEM_PROMPT":
        return render_system_prompt()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SYSTEM_PROMPT",
    "TOOL_GUIDE",
    "GitSnapshot",
    "PromptContext",
    "SystemBlock",
    "SystemPrompt",
    "build_system_prompt",
    "render_session_block",
    "render_system_prompt",
]
