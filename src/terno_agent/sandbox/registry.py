"""Plugin registry for sandbox backends.

Built-ins (``docker``, ``local``) are seeded directly. Third-party
backends register via the ``terno_agent.sandboxes`` entry-point group
in their own ``pyproject.toml``::

    [project.entry-points."terno_agent.sandboxes"]
    qemu = "terno_qemu:QemuSandbox"

Power users can also point ``TERNO_SANDBOX`` at a fully-qualified
``module.path:ClassName`` to avoid publishing a package — the factory
handles that case before consulting the registry.

A plugin sandbox just needs to satisfy the `Sandbox` Protocol from
`terno_agent.sandbox.base` (one method: ``run_python``). Entry-point
loading is idempotent and tolerant: a single broken plugin emits a
stderr warning and is skipped without affecting other backends.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from importlib import metadata
from typing import Any

from terno_agent.core.exceptions import ConfigError
from terno_agent.sandbox.base import Sandbox

SandboxFactory = Callable[..., Sandbox]

ENTRY_POINT_GROUP = "terno_agent.sandboxes"

_registry: dict[str, SandboxFactory] = {}
_entry_points_loaded = False


# --------------------------------------------------------------------------- #
# Built-in lazy factories
# --------------------------------------------------------------------------- #


def _docker_factory(**options: Any) -> Sandbox:
    from terno_agent.sandbox.docker import DockerSandbox

    return DockerSandbox(**options)


def _local_factory(**options: Any) -> Sandbox:
    from terno_agent.sandbox.local import LocalSandbox

    return LocalSandbox(**options)


_BUILTIN: dict[str, SandboxFactory] = {
    "docker": _docker_factory,
    "local": _local_factory,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def register_sandbox(name: str, factory: SandboxFactory) -> None:
    """Register a sandbox backend under ``name``.

    Used by entry-point loading and available to third-party code that
    prefers programmatic registration. Re-registering the same name
    overwrites the previous factory.
    """
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("sandbox name must be a non-empty string")
    if not callable(factory):
        raise ConfigError(f"sandbox factory for {name!r} must be callable")
    _registry[name.strip().lower()] = factory


def available_sandboxes() -> list[str]:
    """Return the sorted list of registered backend names (built-ins + plugins)."""
    _load_entry_points()
    return sorted(set(_BUILTIN) | set(_registry))


def lookup(kind: str) -> SandboxFactory:
    """Return the factory registered for ``kind`` (loads EPs on first call).

    Raises `ConfigError` with the list of available names on miss.
    """
    _load_entry_points()
    key = kind.strip().lower()
    if key in _registry:
        return _registry[key]
    if key in _BUILTIN:
        return _BUILTIN[key]
    raise ConfigError(
        f"Unknown sandbox {kind!r}. Available: {', '.join(available_sandboxes())}. "
        "Install a plugin package or set TERNO_SANDBOX to "
        "'module.path:ClassName' for ad-hoc backends."
    )


def resolve_import_string(spec: str) -> SandboxFactory:
    """Resolve ``"pkg.mod:Cls"`` into a callable sandbox factory.

    The returned object must be callable (classes satisfy this); the
    factory call site applies ``**options`` to it.
    """
    if ":" not in spec:
        raise ConfigError(
            f"sandbox import string {spec!r} must be of the form 'module.path:ClassName'"
        )
    module_name, _, attr = spec.partition(":")
    module_name = module_name.strip()
    attr = attr.strip()
    if not module_name or not attr:
        raise ConfigError(
            f"sandbox import string {spec!r} must be of the form 'module.path:ClassName'"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(
            f"sandbox {spec!r}: could not import {module_name!r} ({exc})"
        ) from exc
    try:
        target = getattr(module, attr)
    except AttributeError as exc:
        raise ConfigError(
            f"sandbox {spec!r}: module {module_name!r} has no attribute {attr!r}"
        ) from exc
    if not callable(target):
        raise ConfigError(f"sandbox {spec!r}: {attr!r} is not callable")
    return target


def _load_entry_points() -> None:
    """Populate `_registry` from declared entry points. Idempotent."""
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    try:
        eps = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:
        print(
            f"warning: sandbox entry-point discovery failed: {exc}",
            file=sys.stderr,
        )
        return
    for ep in eps:
        try:
            factory = ep.load()
        except Exception as exc:
            print(
                f"warning: sandbox plugin {ep.name!r} failed to load: {exc}",
                file=sys.stderr,
            )
            continue
        if not callable(factory):
            print(
                f"warning: sandbox plugin {ep.name!r} did not export a callable",
                file=sys.stderr,
            )
            continue
        _registry[ep.name.strip().lower()] = factory


def _reset_for_tests() -> None:
    """Test-only: forget loaded entry points and clear registered plugins."""
    global _entry_points_loaded
    _entry_points_loaded = False
    _registry.clear()


__all__ = [
    "ENTRY_POINT_GROUP",
    "SandboxFactory",
    "available_sandboxes",
    "lookup",
    "register_sandbox",
    "resolve_import_string",
]
