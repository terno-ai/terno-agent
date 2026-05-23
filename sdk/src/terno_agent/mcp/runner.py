"""Resolve a `StdioServerConfig` to a concrete subprocess invocation.

Pure decision logic — the only side effect is `shutil.which` to probe
for installed runtimes (uvx / npx / docker). The output is a
`RunnerSpec` that the session layer uses to spawn the MCP server.

Web servers don't go through here; they're handled directly by
`session.McpSession`.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass

from terno_agent.core.exceptions import ConfigError
from terno_agent.mcp.config import DockerOptions, RunnerBlock, StdioServerConfig


@dataclass(frozen=True, slots=True)
class RunnerSpec:
    """A concrete stdio subprocess invocation."""

    argv: tuple[str, ...]
    env: dict[str, str]
    cwd: str | None = None


WhichFn = Callable[[str], str | None]


def resolve(
    cfg: StdioServerConfig,
    *,
    which: WhichFn | None = None,
) -> RunnerSpec:
    """Resolve a stdio server config to an argv + env + cwd.

    `which` is injectable for testing; defaults to `shutil.which`.
    """
    w = which or shutil.which

    # Raw command path (Claude-Code-compatible).
    if cfg.command is not None:
        if not w(cfg.command):
            raise ConfigError(
                f"mcp server '{cfg.name}': command {cfg.command!r} not found on PATH"
            )
        return RunnerSpec(argv=(cfg.command, *cfg.args), env=dict(cfg.env), cwd=cfg.cwd)

    runner = cfg.runner
    if runner is None:  # pragma: no cover - config parser guarantees one or the other
        raise ConfigError(f"mcp server '{cfg.name}': missing command/runner")

    match runner.type:
        case "command":
            return _resolve_command(cfg, runner, w)
        case "uvx":
            return _resolve_uvx(cfg, runner, w)
        case "npx":
            return _resolve_npx(cfg, runner, w)
        case "docker":
            return _resolve_docker(cfg, runner, w)
        case "auto":
            return _resolve_auto(cfg, runner, w)
        case _:  # pragma: no cover - parser already validates
            raise ConfigError(f"mcp server '{cfg.name}': unknown runner.type {runner.type!r}")


# --------------------------------------------------------------------------- #
# Per-type resolvers
# --------------------------------------------------------------------------- #


def _resolve_command(
    cfg: StdioServerConfig, runner: RunnerBlock, w: WhichFn
) -> RunnerSpec:
    assert runner.command is not None
    if not w(runner.command):
        raise ConfigError(
            f"mcp server '{cfg.name}': runner.command {runner.command!r} not on PATH"
        )
    return RunnerSpec(
        argv=(runner.command, *runner.args),
        env=dict(cfg.env),
        cwd=runner.cwd or cfg.cwd,
    )


def _resolve_uvx(
    cfg: StdioServerConfig, runner: RunnerBlock, w: WhichFn
) -> RunnerSpec:
    _require(w, "uvx", cfg.name)
    assert runner.package is not None
    return RunnerSpec(
        argv=("uvx", runner.package, *runner.args),
        env=dict(cfg.env),
        cwd=runner.cwd or cfg.cwd,
    )


def _resolve_npx(
    cfg: StdioServerConfig, runner: RunnerBlock, w: WhichFn
) -> RunnerSpec:
    _require(w, "npx", cfg.name)
    assert runner.package is not None
    return RunnerSpec(
        argv=("npx", "-y", runner.package, *runner.args),
        env=dict(cfg.env),
        cwd=runner.cwd or cfg.cwd,
    )


def _resolve_docker(
    cfg: StdioServerConfig, runner: RunnerBlock, w: WhichFn
) -> RunnerSpec:
    _require(w, "docker", cfg.name)
    assert runner.image is not None
    argv = _docker_argv(cfg.name, runner.image, runner.docker, runner.args, cfg.env)
    return RunnerSpec(argv=argv, env={}, cwd=runner.cwd or cfg.cwd)


def _resolve_auto(
    cfg: StdioServerConfig, runner: RunnerBlock, w: WhichFn
) -> RunnerSpec:
    has_docker = bool(w("docker"))
    has_uvx = bool(w("uvx"))
    has_npx = bool(w("npx"))

    # Preference 1: docker, if both an image and docker are present.
    if runner.image and has_docker:
        argv = _docker_argv(cfg.name, runner.image, runner.docker, runner.args, cfg.env)
        return RunnerSpec(argv=argv, env={}, cwd=runner.cwd or cfg.cwd)

    package = runner.package
    if package is None:
        raise ConfigError(
            f"mcp server '{cfg.name}': runner.type='auto' with no package and "
            "docker unavailable"
        )

    # Preference 2: explicit package_type.
    if runner.package_type == "python":
        _require(w, "uvx", cfg.name)
        return RunnerSpec(
            argv=("uvx", package, *runner.args),
            env=dict(cfg.env),
            cwd=runner.cwd or cfg.cwd,
        )
    if runner.package_type == "npm":
        _require(w, "npx", cfg.name)
        return RunnerSpec(
            argv=("npx", "-y", package, *runner.args),
            env=dict(cfg.env),
            cwd=runner.cwd or cfg.cwd,
        )

    # Preference 3: name heuristic — @scoped / contains-slash → npm; else uvx.
    looks_npm = package.startswith("@") or "/" in package
    if looks_npm and has_npx:
        return RunnerSpec(
            argv=("npx", "-y", package, *runner.args),
            env=dict(cfg.env),
            cwd=runner.cwd or cfg.cwd,
        )
    if has_uvx:
        return RunnerSpec(
            argv=("uvx", package, *runner.args),
            env=dict(cfg.env),
            cwd=runner.cwd or cfg.cwd,
        )
    if has_npx:
        return RunnerSpec(
            argv=("npx", "-y", package, *runner.args),
            env=dict(cfg.env),
            cwd=runner.cwd or cfg.cwd,
        )
    raise ConfigError(
        f"mcp server '{cfg.name}': runner.type='auto' but no usable runtime "
        "found (uvx, npx, or docker)"
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _require(w: WhichFn, tool: str, server: str) -> None:
    if not w(tool):
        raise ConfigError(f"mcp server '{server}': {tool!r} not found on PATH")


def _docker_argv(
    server: str,
    image: str,
    docker_opts: DockerOptions,
    args: tuple[str, ...],
    env: dict[str, str],
) -> tuple[str, ...]:
    out: list[str] = ["docker", "run", "--rm", "-i"]
    for m in docker_opts.mounts:
        spec = f"{m.source}:{m.target}"
        if m.readonly:
            spec += ":ro"
        out += ["-v", spec]
    for k, v in env.items():
        out += ["-e", f"{k}={v}"]
    for k in docker_opts.env_passthrough:
        if k in os.environ:
            out += ["-e", f"{k}={os.environ[k]}"]
        else:
            out += ["-e", k]
    out += ["--name", f"terno-mcp-{server}-{os.getpid()}", image]
    out += list(args)
    return tuple(out)


__all__ = ["RunnerSpec", "resolve"]
