"""Resolve on-disk locations for Open Knowledge Format (OKF) bundles.

A *bundle* is one datasource's knowledge — a directory of markdown files.
All bundles live under ``<workdir>/.terno/knowledge`` (mirroring the
``.terno/memory`` and ``.terno/attachments`` conventions). The location is
overridable via ``TERNO_KNOWLEDGE_HOME`` (used by tests to redirect into a
temp path).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

HOME_ENV_VAR = "TERNO_KNOWLEDGE_HOME"

_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def slugify(name: str) -> str:
    """Lowercase slug safe as a single path segment.

    Underscores are preserved (they are common, valid datasource/table
    names); any other non-alphanumeric run collapses to a single ``-``.
    """
    s = _SLUG_RE.sub("-", name.strip().lower()).strip("-_")
    return s[:64] or "datasource"


def knowledge_root(workdir: Path) -> Path:
    """Return the directory holding all knowledge bundles for ``workdir``.

    Overridable via ``TERNO_KNOWLEDGE_HOME``. Defaults to
    ``<workdir>/.terno/knowledge``.
    """
    override = os.getenv(HOME_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return (Path(workdir).resolve() / ".terno" / "knowledge").resolve()


def bundle_dir(workdir: Path, datasource: str) -> Path:
    """Return the bundle directory for ``datasource`` under ``workdir``."""
    return knowledge_root(workdir) / slugify(datasource)


__all__ = ["HOME_ENV_VAR", "bundle_dir", "knowledge_root", "slugify"]
