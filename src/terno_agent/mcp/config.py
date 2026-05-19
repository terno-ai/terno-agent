"""Parse and validate `.mcp.json`.

The on-disk format is Claude-Code-compatible: a top-level
``{"mcpServers": {"<name>": { ... }}}`` map. Each entry is one of three
shapes:

A. Raw stdio — ``command`` + ``args`` + optional ``env``.
B. Higher-level stdio — ``runner`` block that lets terno pick uvx /
   npx / docker.
C. Web — ``url`` + optional ``transport`` and ``headers``.

A server entry must declare exactly one of ``command``, ``runner``,
or ``url``. ``${VAR}`` references in ``env``, ``headers``, and
``runner.args`` are expanded from the process environment at load
time; missing variables raise `ConfigError` naming the server.

Discovery order when ``path`` is omitted: ``$TERNO_MCP_CONFIG``, then
``./.mcp.json``, then ``~/.terno/mcp.json``. First hit wins. A missing
file is not an error — it just yields an empty list.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from terno_agent.core.exceptions import ConfigError

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_VALID_RUNNER_TYPES = {"auto", "uvx", "npx", "docker", "command"}
_VALID_TRANSPORTS = {"sse", "http"}
_VALID_PACKAGE_TYPES = {"python", "npm"}


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DockerMount:
    source: str
    target: str
    readonly: bool = False


@dataclass(frozen=True, slots=True)
class DockerOptions:
    mounts: tuple[DockerMount, ...] = ()
    env_passthrough: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunnerBlock:
    type: str  # "auto" | "uvx" | "npx" | "docker" | "command"
    package: str | None = None
    package_type: str | None = None  # "python" | "npm"
    image: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    cwd: str | None = None
    timeout_s: int | None = None
    docker: DockerOptions = field(default_factory=DockerOptions)


@dataclass(frozen=True, slots=True)
class StdioServerConfig:
    name: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    runner: RunnerBlock | None = None
    cwd: str | None = None
    timeout_s: int | None = None


@dataclass(frozen=True, slots=True)
class HttpServerConfig:
    name: str
    url: str
    transport: str = "sse"  # auto-resolved at load time
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: int | None = None


McpServerConfig = StdioServerConfig | HttpServerConfig


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def load_mcp_config(path: Path | str | None = None) -> list[McpServerConfig]:
    """Load and validate `.mcp.json`. Returns `[]` if no file is found."""
    resolved = _resolve_path(path)
    if resolved is None:
        return []
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"mcp config {resolved} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read mcp config {resolved}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"mcp config {resolved}: top-level must be an object")
    servers = raw.get("mcpServers")
    if servers is None:
        return []
    if not isinstance(servers, dict):
        raise ConfigError(f"mcp config {resolved}: 'mcpServers' must be an object")

    out: list[McpServerConfig] = []
    for name, entry in servers.items():
        if not isinstance(name, str) or not name:
            raise ConfigError(f"mcp config {resolved}: server name must be a non-empty string")
        if not isinstance(entry, dict):
            raise ConfigError(
                f"mcp config {resolved}: server '{name}' must be an object"
            )
        out.append(_parse_server(name, entry))
    return out


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _resolve_path(path: Path | str | None) -> Path | None:
    if path:
        p = Path(path).expanduser()
        return p if p.exists() else None
    for candidate in (
        os.environ.get("TERNO_MCP_CONFIG"),
        "./.mcp.json",
        "~/.terno/mcp.json",
    ):
        if not candidate:
            continue
        p = Path(candidate).expanduser()
        if p.exists():
            return p
    return None


def _parse_server(name: str, entry: dict[str, Any]) -> McpServerConfig:
    has_url = "url" in entry
    has_command = "command" in entry
    has_runner = "runner" in entry
    declared = sum((has_url, has_command, has_runner))
    if declared != 1:
        raise ConfigError(
            f"mcp server '{name}': must declare exactly one of "
            f"'url', 'command', or 'runner' (found {declared})"
        )

    if has_url:
        return _parse_http(name, entry)
    return _parse_stdio(name, entry, has_command=has_command)


def _parse_http(name: str, entry: dict[str, Any]) -> HttpServerConfig:
    url = entry["url"]
    if not isinstance(url, str) or not url:
        raise ConfigError(f"mcp server '{name}': 'url' must be a non-empty string")
    transport = entry.get("transport")
    if transport is None:
        transport = "sse" if url.rstrip("/").endswith("/sse") else "http"
    if transport not in _VALID_TRANSPORTS:
        raise ConfigError(
            f"mcp server '{name}': transport must be one of {sorted(_VALID_TRANSPORTS)}"
        )
    headers = _interp_str_map(entry.get("headers") or {}, name, "headers")
    return HttpServerConfig(
        name=name,
        url=url,
        transport=transport,
        headers=headers,
        timeout_s=_optional_int(entry.get("timeout_s"), name, "timeout_s"),
    )


def _parse_stdio(
    name: str, entry: dict[str, Any], *, has_command: bool
) -> StdioServerConfig:
    env = _interp_str_map(entry.get("env") or {}, name, "env")
    cwd = entry.get("cwd")
    timeout_s = _optional_int(entry.get("timeout_s"), name, "timeout_s")

    if has_command:
        command = entry["command"]
        if not isinstance(command, str) or not command:
            raise ConfigError(f"mcp server '{name}': 'command' must be a non-empty string")
        args = _tuple_of_strings(entry.get("args") or [], name, "args")
        return StdioServerConfig(
            name=name,
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            timeout_s=timeout_s,
        )

    runner_raw = entry["runner"]
    if not isinstance(runner_raw, dict):
        raise ConfigError(f"mcp server '{name}': 'runner' must be an object")
    runner = _parse_runner(name, runner_raw)
    return StdioServerConfig(
        name=name,
        runner=runner,
        env=env,
        cwd=cwd,
        timeout_s=timeout_s,
    )


def _parse_runner(name: str, raw: dict[str, Any]) -> RunnerBlock:
    rtype = raw.get("type", "auto")
    if rtype not in _VALID_RUNNER_TYPES:
        raise ConfigError(
            f"mcp server '{name}': runner.type must be one of {sorted(_VALID_RUNNER_TYPES)}"
        )
    package = raw.get("package")
    package_type = raw.get("package_type")
    if package_type is not None and package_type not in _VALID_PACKAGE_TYPES:
        raise ConfigError(
            f"mcp server '{name}': runner.package_type must be one of "
            f"{sorted(_VALID_PACKAGE_TYPES)}"
        )
    image = raw.get("image")
    command = raw.get("command")
    args = _tuple_of_strings(raw.get("args") or [], name, "runner.args", interp=True)
    cwd = raw.get("cwd")
    timeout_s = _optional_int(raw.get("timeout_s"), name, "runner.timeout_s")

    if rtype == "command" and not command:
        raise ConfigError(f"mcp server '{name}': runner.type='command' requires 'command'")
    if rtype in {"uvx", "npx"} and not package:
        raise ConfigError(
            f"mcp server '{name}': runner.type={rtype!r} requires 'package'"
        )
    if rtype == "docker" and not image:
        raise ConfigError(f"mcp server '{name}': runner.type='docker' requires 'image'")
    if rtype == "auto" and not (package or image):
        raise ConfigError(
            f"mcp server '{name}': runner.type='auto' requires 'package' or 'image'"
        )

    docker = _parse_docker_options(name, raw.get("docker") or {})

    return RunnerBlock(
        type=rtype,
        package=package,
        package_type=package_type,
        image=image,
        command=command,
        args=args,
        cwd=cwd,
        timeout_s=timeout_s,
        docker=docker,
    )


def _parse_docker_options(name: str, raw: dict[str, Any]) -> DockerOptions:
    if not isinstance(raw, dict):
        raise ConfigError(f"mcp server '{name}': runner.docker must be an object")
    mounts_raw = raw.get("mounts") or []
    if not isinstance(mounts_raw, list):
        raise ConfigError(f"mcp server '{name}': runner.docker.mounts must be a list")
    mounts: list[DockerMount] = []
    for i, m in enumerate(mounts_raw):
        if not isinstance(m, dict):
            raise ConfigError(
                f"mcp server '{name}': runner.docker.mounts[{i}] must be an object"
            )
        source = m.get("source")
        target = m.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            raise ConfigError(
                f"mcp server '{name}': runner.docker.mounts[{i}] needs string "
                "'source' and 'target'"
            )
        mounts.append(DockerMount(source=source, target=target, readonly=bool(m.get("readonly"))))
    passthrough = _tuple_of_strings(
        raw.get("env_passthrough") or [], name, "runner.docker.env_passthrough"
    )
    return DockerOptions(mounts=tuple(mounts), env_passthrough=passthrough)


def _interp_str_map(raw: Any, server: str, where: str) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ConfigError(f"mcp server '{server}': '{where}' must be an object")
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            raise ConfigError(f"mcp server '{server}': '{where}' keys must be strings")
        if not isinstance(v, str):
            raise ConfigError(
                f"mcp server '{server}': '{where}.{k}' must be a string"
            )
        out[k] = _interpolate(v, server, f"{where}.{k}")
    return out


def _tuple_of_strings(
    raw: Any, server: str, where: str, *, interp: bool = False
) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ConfigError(f"mcp server '{server}': '{where}' must be a list of strings")
    out: list[str] = []
    for i, v in enumerate(raw):
        if not isinstance(v, str):
            raise ConfigError(
                f"mcp server '{server}': '{where}[{i}]' must be a string"
            )
        out.append(_interpolate(v, server, f"{where}[{i}]") if interp else v)
    return tuple(out)


def _optional_int(raw: Any, server: str, where: str) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
        raise ConfigError(f"mcp server '{server}': '{where}' must be a positive integer")
    return raw


def _interpolate(text: str, server: str, where: str) -> str:
    def _sub(m: re.Match[str]) -> str:
        var = m.group(1)
        if var not in os.environ:
            raise ConfigError(
                f"mcp server '{server}': '{where}' references ${{{var}}} but "
                "the environment variable is not set"
            )
        return os.environ[var]

    return _VAR_RE.sub(_sub, text)


__all__ = [
    "DockerMount",
    "DockerOptions",
    "HttpServerConfig",
    "McpServerConfig",
    "RunnerBlock",
    "StdioServerConfig",
    "load_mcp_config",
]
