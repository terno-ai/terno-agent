"""Model Context Protocol (MCP) client integration for terno-agent.

`McpManager` is the public entry point. It loads a `.mcp.json` file,
spawns sessions for each declared server (over stdio or HTTP), and
exposes each remote tool as a sync `Tool` the rest of terno can use.

All MCP work runs on a background asyncio event loop owned by the
manager — the agent's own run loop stays synchronous.
"""

from terno_agent.mcp.config import (
    HttpServerConfig,
    McpServerConfig,
    StdioServerConfig,
    load_mcp_config,
)
from terno_agent.mcp.manager import McpManager
from terno_agent.mcp.tool import McpTool

__all__ = [
    "HttpServerConfig",
    "McpManager",
    "McpServerConfig",
    "McpTool",
    "StdioServerConfig",
    "load_mcp_config",
]
