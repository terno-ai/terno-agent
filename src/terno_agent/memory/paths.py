"""Resolve the per-project memory directory.

All memory lives in ``<workdir>/.terno/memory``. The legacy global
``~/.terno_agent/memory`` location is no longer used; tests still set
``TERNO_MEMORY_HOME`` to redirect the dir into a tmp path.
"""

from __future__ import annotations

import os
from pathlib import Path

HOME_ENV_VAR = "TERNO_MEMORY_HOME"
# Backwards-compatible alias used by older tests.
GLOBAL_ENV_VAR = HOME_ENV_VAR


def memory_dir(workdir: Path) -> Path:
    """Return the memory dir for ``workdir``.

    Overridable via ``TERNO_MEMORY_HOME`` (used by tests). Defaults to
    ``<workdir>/.terno/memory``.
    """
    override = os.getenv(HOME_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return (Path(workdir).resolve() / ".terno" / "memory").resolve()


__all__ = [
    "GLOBAL_ENV_VAR",
    "HOME_ENV_VAR",
    "memory_dir",
]
