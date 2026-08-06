"""Run MCP servers *inside* the session's sandbox container.

The default :class:`~terno_agent.mcp.session.McpSession` spawns each stdio
MCP server as a subprocess of the **host** process (via the ``mcp`` SDK's
``stdio_client``). When the agent is driven by a host that injects a
per-user/per-session sandbox (e.g. terno-ai's ``TernoReplSandbox``, which
proxies ``run_python`` into that user's container), we instead want the MCP
server to run *inside that container* — so stdio servers are children of the
container's persistent Python process and remote calls egress from inside the
container. This mirrors terno-ai's own in-sandbox MCP design.

Rather than re-implement an in-container MCP client, this session reuses the
one terno-ai already bakes into its sandbox image at ``/opt/sandbox/
mcp_client.py`` — importable as ``mcp_client`` with a module-global
``mcp_manager`` singleton. Because the sandbox's ``run_python`` shares one
persistent Python namespace across calls, that singleton (and every MCP
connection it owns) survives between calls, so multiple ``SandboxMcpSession``
objects — one per configured server, as :class:`McpManager` creates them —
all drive the *same* in-container client.

Transport to the in-container client is code injection: each operation ships a
small Python snippet through ``Sandbox.run_python`` that imports
``mcp_manager`` and calls ``connect_all`` / ``get_tool_schemas`` /
``call_tool``, then prints a sentinel-framed JSON envelope we parse back out of
stdout. The public methods are ``async`` and blocking work is offloaded with
``asyncio.to_thread`` so the manager's shared bridge loop is never blocked —
this makes the class a structural drop-in for :class:`McpSession`, requiring no
changes to :class:`McpManager` or :class:`McpTool`.

Prerequisites (all satisfied by terno-ai's container image; auto-detected
otherwise so unsupported sandboxes fall back to the host path — see
``sandbox_mcp_supported``): a **stateful** ``run_python`` whose namespace
persists across calls (rules out :class:`LocalSandbox`), and the ``mcp_client``
module plus any stdio runtimes (``npx`` / ``uvx``) present inside the sandbox.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from typing import Any

from terno_agent.core.exceptions import ConfigError, ToolError
from terno_agent.mcp.config import HttpServerConfig, McpServerConfig, StdioServerConfig

# Marker the injected snippet prints before its JSON envelope so we can locate
# the result amid any other stdout the sandbox emits. Distinct from terno-ai's
# own ``__TERNO_MCP_RESULT__`` so the two never collide.
_SENTINEL = "__TERNO_AGENT_MCP__"

# Extra PATH entries so stdio runners (npx / uvx / pipx-installed servers)
# resolve inside the container regardless of the kernel's inherited PATH.
_SANDBOX_PATH = (
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:"
    "/home/runner/.local/bin:/opt/sandbox/bin"
)

# run_python budgets (seconds). Kept just under the manager's bridge timeouts
# (connect=30, list_tools=15) so a slow call surfaces as our own clean error
# rather than a bridge cancellation.
_CONNECT_TIMEOUT_S = 28
_LIST_TIMEOUT_S = 13
_DEFAULT_CALL_TIMEOUT_S = 115


class SandboxMcpSession:
    """One MCP server, connected and called from inside the sandbox.

    Satisfies the same structural protocol as :class:`McpSession`
    (``connect`` / ``list_tools`` / ``call_tool`` / ``aclose``), so
    :class:`McpManager` treats the two interchangeably.
    """

    def __init__(self, config: McpServerConfig, sandbox: Any) -> None:
        self.config = config
        self._sandbox = sandbox
        self._container_config = _to_container_config(config)
        self._lock = asyncio.Lock()
        self._connected = False
        call_timeout = getattr(config, "timeout_s", None) or _DEFAULT_CALL_TIMEOUT_S
        self._call_timeout_s = min(int(call_timeout), _DEFAULT_CALL_TIMEOUT_S)

    # ----- lifecycle ----------------------------------------------------- #

    async def connect(self) -> None:
        cfg_b64 = _b64(json.dumps(self._container_config))
        code = _CONNECT_TEMPLATE.replace("__CFG_B64__", cfg_b64)
        envelope = await self._run(code, timeout_s=_CONNECT_TIMEOUT_S)
        if not envelope.get("ok"):
            raise ConfigError(
                f"mcp server '{self.config.name}' (in-sandbox): "
                f"{envelope.get('fatal') or 'connect failed'}"
            )
        errors = envelope.get("errors") or {}
        err = errors.get(self.config.name)
        if err:
            raise ConfigError(
                f"mcp server '{self.config.name}' (in-sandbox): {err}"
            )
        self._connected = True

    async def list_tools(self) -> list[Any]:
        if not self._connected:
            raise ToolError(
                f"mcp session for '{self.config.name}' is not connected"
            )
        server_b64 = _b64(self.config.name)
        code = _LIST_TEMPLATE.replace("__SERVER_B64__", server_b64)
        envelope = await self._run(code, timeout_s=_LIST_TIMEOUT_S)
        if not envelope.get("ok"):
            raise ToolError(
                f"mcp server '{self.config.name}' (in-sandbox) list_tools: "
                f"{envelope.get('fatal') or 'failed'}"
            )
        # Adapt terno-ai's schema keys (tool_name/input_schema) to the shape
        # McpManager reads from the mcp SDK (name/inputSchema).
        tools: list[dict[str, Any]] = []
        for s in envelope.get("tools") or []:
            tools.append(
                {
                    "name": s.get("tool_name") or "tool",
                    "description": s.get("description") or "",
                    "inputSchema": s.get("input_schema")
                    or {"type": "object", "properties": {}},
                }
            )
        return tools

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        if not self._connected:
            raise ToolError(
                f"mcp session for '{self.config.name}' is not connected"
            )
        code = (
            _CALL_TEMPLATE.replace("__SERVER_B64__", _b64(self.config.name))
            .replace("__TOOL_B64__", _b64(name))
            .replace("__ARGS_B64__", _b64(json.dumps(args or {})))
        )
        async with self._lock:
            envelope = await self._run(code, timeout_s=self._call_timeout_s)
        if not envelope.get("ok"):
            raise ToolError(
                f"mcp {self.config.name}.{name} (in-sandbox): "
                f"{envelope.get('fatal') or 'call failed'}"
            )
        # The in-container client already flattens content to a string (and
        # returns 'Error: ...' text on failure rather than raising), matching
        # terno-ai's own MCPTool contract. Wrap it so McpTool.run can render
        # it through format_content unchanged.
        return _CallResult(str(envelope.get("output") or ""))

    async def aclose(self) -> None:
        # The in-container client is a shared singleton owning connections for
        # every server; tearing it down here would break sibling sessions.
        # Its lifetime is the container's, which the host manages. Nothing to
        # do on our side beyond marking this handle closed.
        self._connected = False

    # ----- internals ----------------------------------------------------- #

    async def _run(self, code: str, *, timeout_s: int) -> dict[str, Any]:
        """Execute ``code`` in the sandbox and parse the sentinel envelope."""
        try:
            result = await asyncio.to_thread(
                self._sandbox.run_python, code, timeout_s=timeout_s
            )
        except Exception as exc:  # sandbox transport failure
            return {"ok": False, "fatal": f"sandbox run_python failed: {exc}"}
        stdout = getattr(result, "stdout", "") or ""
        parsed = _parse_envelope(stdout)
        if parsed is None:
            stderr = (getattr(result, "stderr", "") or "").strip()
            detail = stderr or "no result envelope in sandbox output"
            return {"ok": False, "fatal": detail}
        return parsed


class _CallResult:
    """Minimal stand-in for an mcp SDK ``CallToolResult``.

    ``McpTool.run`` reads ``.content`` (a list of blocks) and ``.isError``.
    We already have a flat string, so expose it as a single text block and
    never flag ``isError`` — the in-container client surfaces failures as
    ``Error: ...`` text, matching terno-ai, and we preserve that behaviour.
    """

    __slots__ = ("content", "isError")

    def __init__(self, text: str) -> None:
        self.content = [{"type": "text", "text": text}]
        self.isError = False


# --------------------------------------------------------------------------- #
# Capability detection + factory
# --------------------------------------------------------------------------- #


def sandbox_mcp_supported(sandbox: Any) -> bool:
    """Return True when ``sandbox`` can host MCP in-container.

    Probes the live sandbox once: the in-container MCP client only works if
    ``run_python`` keeps a persistent namespace (so the singleton survives)
    and the ``mcp_client`` module is importable inside it. A single cheap
    ``run_python`` answers both — a stateless or module-less sandbox simply
    fails to print the marker, and the caller falls back to the host path.
    Any error is treated as "unsupported" so detection never blocks startup.
    """
    if sandbox is None:
        return False
    probe = (
        "import importlib.util as u;"
        "print('__MCP_CLIENT_OK__' if u.find_spec('mcp_client') "
        "else '__MCP_CLIENT_NO__')"
    )
    try:
        result = sandbox.run_python(probe, timeout_s=15)
    except Exception:
        return False
    return "__MCP_CLIENT_OK__" in (getattr(result, "stdout", "") or "")


def sandbox_mcp_session_factory(sandbox: Any) -> Callable[[McpServerConfig], SandboxMcpSession]:
    """Build a ``session_factory`` for :class:`McpManager` bound to ``sandbox``."""

    def factory(cfg: McpServerConfig) -> SandboxMcpSession:
        return SandboxMcpSession(cfg, sandbox)

    return factory


# --------------------------------------------------------------------------- #
# Config translation
# --------------------------------------------------------------------------- #


def _to_container_config(cfg: McpServerConfig) -> dict[str, Any]:
    """Translate an SDK ``McpServerConfig`` into the dict shape terno-ai's
    in-container ``mcp_client`` expects (``transport_type`` + fields)."""
    if isinstance(cfg, HttpServerConfig):
        ttype = "sse" if cfg.transport == "sse" else "streamable_http"
        return {
            "name": cfg.name,
            "transport_type": ttype,
            "remote_url": cfg.url,
            "headers": dict(cfg.headers),
        }
    if not isinstance(cfg, StdioServerConfig):  # pragma: no cover - exhaustive
        raise ConfigError(f"mcp server '{getattr(cfg, 'name', '?')}': unknown config type")

    command, args = _resolve_stdio(cfg)
    return {
        "name": cfg.name,
        "transport_type": "stdio",
        "command": command,
        "args": list(args),
        "env": dict(cfg.env),
    }


def _resolve_stdio(cfg: StdioServerConfig) -> tuple[str, list[str]]:
    """Pick the (command, args) to run inside the container.

    Unlike the host runner (``mcp/runner.py``) we do NOT ``shutil.which`` —
    resolution happens against the *container's* PATH at spawn time. The
    ``docker`` runner is unsupported here (it would need docker-in-docker);
    such servers should use ``command`` or a package runner instead.
    """
    if cfg.command is not None:
        return cfg.command, list(cfg.args)

    runner = cfg.runner
    if runner is None:  # pragma: no cover - parser guarantees one or the other
        raise ConfigError(f"mcp server '{cfg.name}': missing command/runner")

    match runner.type:
        case "command":
            if not runner.command:
                raise ConfigError(f"mcp server '{cfg.name}': runner.command missing")
            return runner.command, list(runner.args)
        case "uvx":
            return "uvx", [runner.package or "", *runner.args]
        case "npx":
            return "npx", ["-y", runner.package or "", *runner.args]
        case "auto":
            pkg = runner.package
            if not pkg:
                raise ConfigError(
                    f"mcp server '{cfg.name}': runner.type='auto' needs a "
                    "'package' for in-sandbox MCP (image-based servers are "
                    "not supported inside the sandbox)"
                )
            if runner.package_type == "npm" or pkg.startswith("@") or "/" in pkg:
                return "npx", ["-y", pkg, *runner.args]
            return "uvx", [pkg, *runner.args]
        case _:  # "docker" or anything else
            raise ConfigError(
                f"mcp server '{cfg.name}': runner.type={runner.type!r} is not "
                "supported for in-sandbox MCP; use 'command', 'uvx', or 'npx'"
            )


# --------------------------------------------------------------------------- #
# Envelope parsing + injected snippets
# --------------------------------------------------------------------------- #


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _parse_envelope(stdout: str) -> dict[str, Any] | None:
    """Return the last sentinel-framed JSON object in ``stdout``, or None."""
    if not stdout:
        return None
    for line in reversed(stdout.splitlines()):
        if _SENTINEL not in line:
            continue
        payload = line.split(_SENTINEL, 1)[1].strip()
        try:
            obj = json.loads(payload)
        except (ValueError, TypeError):
            return None
        return obj if isinstance(obj, dict) else None
    return None


# Prepended to every snippet: widen PATH and define the emit helper. Kept as a
# plain string (no f-string) so the dict/format braces below stay literal.
_PREAMBLE = f'''
import os, json, base64, sys, importlib.util
for _p in {_SANDBOX_PATH!r}.split(os.pathsep):
    _parts = [x for x in os.environ.get("PATH", "").split(os.pathsep) if x]
    if _p and _p not in _parts:
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + _p
_SENT = "{_SENTINEL}"
def _emit(d):
    sys.stdout.write(_SENT + json.dumps(d) + "\\n")
    sys.stdout.flush()
'''


_CONNECT_TEMPLATE = _PREAMBLE + '''
try:
    if importlib.util.find_spec("mcp_client") is None:
        _emit({"ok": False, "fatal": "mcp_client module not available inside sandbox"})
    else:
        from mcp_client import mcp_manager
        mcp, _ = mcp_manager.get_or_create_client()
        _cfg = json.loads(base64.b64decode("__CFG_B64__").decode())
        _live = getattr(mcp, "_sessions", {}) or {}
        _to_connect = [] if _cfg["name"] in _live else [_cfg]
        errors = mcp.run_async(mcp.connect_all(_to_connect)) if _to_connect else {}
        _emit({"ok": True, "errors": errors or {}})
except Exception as e:
    _emit({"ok": False, "fatal": "connect failed: " + repr(e)})
'''


_LIST_TEMPLATE = _PREAMBLE + '''
try:
    from mcp_client import mcp_manager
    mcp, _ = mcp_manager.get_or_create_client()
    _server = base64.b64decode("__SERVER_B64__").decode()
    _schemas = mcp.get_tool_schemas() or []
    _emit({"ok": True, "tools": [s for s in _schemas if s.get("server_name") == _server]})
except Exception as e:
    _emit({"ok": False, "fatal": "list_tools failed: " + repr(e)})
'''


_CALL_TEMPLATE = _PREAMBLE + '''
try:
    from mcp_client import mcp_manager
    mcp, _ = mcp_manager.get_or_create_client()
    _server = base64.b64decode("__SERVER_B64__").decode()
    _tool = base64.b64decode("__TOOL_B64__").decode()
    _args = json.loads(base64.b64decode("__ARGS_B64__").decode())
    _out = mcp.run_async(mcp.call_tool(_server, _tool, _args))
    _emit({"ok": True, "output": _out if isinstance(_out, str) else str(_out)})
except Exception as e:
    _emit({"ok": False, "fatal": "call failed: " + repr(e)})
'''


__all__ = [
    "SandboxMcpSession",
    "sandbox_mcp_session_factory",
    "sandbox_mcp_supported",
]
