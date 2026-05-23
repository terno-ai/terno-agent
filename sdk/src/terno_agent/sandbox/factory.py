"""Sandbox factory.

Looks up a backend by name via the registry, or resolves an
``"module.path:ClassName"`` import string for ad-hoc backends, and
instantiates it with the supplied keyword options.

Backend-construction errors (e.g. Docker daemon missing) propagate as
``SandboxError`` so callers can warn and continue without a sandbox.
Unknown names and bad option kwargs surface as ``ConfigError``.
"""

from __future__ import annotations

from terno_agent.core.exceptions import ConfigError
from terno_agent.sandbox.base import Sandbox
from terno_agent.sandbox.registry import lookup, resolve_import_string


def create_sandbox(kind: str, **options: object) -> Sandbox:
    """Return a `Sandbox` for the requested kind.

    Built-ins (``docker``, ``local``) ship with the package. Plugins
    register via the ``terno_agent.sandboxes`` entry-point group, or
    can be supplied inline as ``"package.module:ClassName"``.
    """
    if not isinstance(kind, str) or not kind.strip():
        raise ConfigError("sandbox kind must be a non-empty string")

    factory = (
        resolve_import_string(kind) if ":" in kind else lookup(kind)
    )
    try:
        return factory(**options)
    except TypeError as exc:
        raise ConfigError(
            f"sandbox {kind!r} rejected options {sorted(options)}: {exc}"
        ) from exc


__all__ = ["create_sandbox"]
