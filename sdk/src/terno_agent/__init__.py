"""terno-agent: a coding agent CLI + SDK with file ops, shell, sandboxed
Python, task tracking, subagents, and pluggable MCP servers."""

from terno_agent.config import Config
from terno_agent.core.events import TaskListUpdate
from terno_agent.core.permissions import (
    PermissionDecision,
    PermissionMode,
    PermissionPolicy,
    PermissionRequest,
)
from terno_agent.sdk import Agent
from terno_agent.tools.schedules import (
    InMemoryScheduleStore,
    Schedule,
    ScheduleStore,
)
from terno_agent.tools.tasks import (
    InMemoryTaskStore,
    Task,
    TaskStore,
)

__version__ = "0.2.0"
__all__ = [
    "Agent",
    "Config",
    "InMemoryScheduleStore",
    "InMemoryTaskStore",
    "PermissionDecision",
    "PermissionMode",
    "PermissionPolicy",
    "PermissionRequest",
    "Schedule",
    "ScheduleStore",
    "Task",
    "TaskListUpdate",
    "TaskStore",
    "__version__",
]
