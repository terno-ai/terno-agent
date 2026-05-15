from __future__ import annotations

from terno_agent.core.exceptions import ConfigError
from terno_agent.sandbox.base import Sandbox


def create_sandbox(kind: str, **options: object) -> Sandbox:
    """Return a `Sandbox` for the requested kind.

    Supported kinds: ``docker``, ``local``. Future kinds (``e2b``, ``modal``,
    ``firejail``) just register here.
    """
    kind = kind.lower().strip()
    if kind == "docker":
        from terno_agent.sandbox.docker import DockerSandbox

        return DockerSandbox(**options)  # type: ignore[arg-type]
    if kind == "local":
        from terno_agent.sandbox.local import LocalSandbox

        return LocalSandbox(**options)  # type: ignore[arg-type]
    raise ConfigError(f"Unknown sandbox kind: {kind!r}. Supported: docker, local.")
