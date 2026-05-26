"""Public namespace shim — re-exports the terno-agent SDK.

This lets users write:

    from terno import Agent

instead of `from terno_agent import Agent`. Both work and reference the
same objects.
"""

from terno_agent import __version__
from terno_agent.config import Config
from terno_agent.core.permissions import (
    PermissionDecision,
    PermissionMode,
    PermissionPolicy,
    PermissionRequest,
)
from terno_agent.sdk import Agent

__all__ = [
    "Agent",
    "Config",
    "PermissionDecision",
    "PermissionMode",
    "PermissionPolicy",
    "PermissionRequest",
    "__version__",
]
