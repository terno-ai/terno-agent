"""Resolve on-disk locations for wiki knowledge and workspace memory.

Two distinct storage models live here:

* **OKF knowledge bundles** — one datasource's schema knowledge, a directory
  of markdown files under ``<workdir>/.terno/knowledge`` (built by
  :class:`~terno_agent.wiki.builder.DatasourceKnowledgeAgent`). Overridable
  via ``TERNO_KNOWLEDGE_HOME`` (used by tests).

* **Workspace memory** — the cross-session memory the wiki curator writes.
  It uses the same OKF bundle format as knowledge, but it does NOT live under
  ``.terno``; it lives directly in the user/org *workspace* ``memory`` folders
  that terno-ai bind-mounts into the sandbox and lists in the file-browser UI:

    - user memory:  ``USER_WORKSPACE_ROOT/<org>/<user>/memory``
    - org memory:   ``ORG_WORKSPACE_ROOT/<org>/memory``

  Memory bundles live as subdirectories directly under those folders
  (``<memory>/<datasource>/index.md`` + one file per fact). The roots come
  from ``TERNO_USER_WORKSPACE_ROOT`` / ``TERNO_ORG_WORKSPACE_ROOT`` (mirroring
  terno-ai's ``USER_WORKSPACE_ROOT`` / ``ORG_WORKSPACE_ROOT``) or are passed
  explicitly. Only org admins may write to the org memory folder.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

HOME_ENV_VAR = "TERNO_KNOWLEDGE_HOME"
USER_WORKSPACE_ENV_VAR = "TERNO_USER_WORKSPACE_ROOT"
ORG_WORKSPACE_ENV_VAR = "TERNO_ORG_WORKSPACE_ROOT"

#: The ``memory`` subfolder name inside every workspace root.
MEMORY_DIRNAME = "memory"

_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def slugify(name: str) -> str:
    """Lowercase slug safe as a single path segment.

    Underscores are preserved (they are common, valid datasource/table
    names); any other non-alphanumeric run collapses to a single ``-``.
    """
    s = _SLUG_RE.sub("-", name.strip().lower()).strip("-_")
    return s[:64] or "datasource"


# ----- OKF knowledge bundles (schema knowledge, under .terno) ------------- #


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


# ----- Workspace memory (terno-ai parity, NOT under .terno) --------------- #


def user_memory_dir(user_workspace_root: str | Path, org: str, username: str) -> Path:
    """Return a user's private memory dir: ``<root>/<org>/<user>/memory``.

    ``org``/``username`` are slugified into single, traversal-safe segments so
    an attacker-controlled username can never escape the workspace root.
    """
    root = Path(user_workspace_root).expanduser().resolve()
    return (root / slugify(org) / slugify(username) / MEMORY_DIRNAME).resolve()


def org_memory_dir(org_workspace_root: str | Path, org: str) -> Path:
    """Return an org's shared memory dir: ``<root>/<org>/memory``."""
    root = Path(org_workspace_root).expanduser().resolve()
    return (root / slugify(org) / MEMORY_DIRNAME).resolve()


def memory_bundle_dir(memory_root: str | Path, datasource: str) -> Path:
    """Return the OKF bundle dir for ``datasource`` under a memory folder.

    Unlike :func:`bundle_dir` (which nests under ``.terno/knowledge``), memory
    bundles sit directly under the workspace ``memory`` folder so they are
    visible in the terno-ai file browser: ``<memory_root>/<datasource>``.
    """
    return (Path(memory_root).resolve() / slugify(datasource)).resolve()


def workspace_memory_dirs(
    *,
    user_workspace_root: str | Path | None,
    org_workspace_root: str | Path | None,
    org: str,
    username: str,
) -> tuple[Path | None, Path | None]:
    """Resolve ``(user_dir, org_dir)`` from workspace roots + identity.

    Each dir is ``None`` when its inputs are incomplete (missing root, org, or
    — for the user dir — username), so a partially-configured host degrades to
    "no memory there" rather than writing to a wrong path.
    """
    user_dir: Path | None = None
    org_dir: Path | None = None
    if user_workspace_root and org and username:
        user_dir = user_memory_dir(user_workspace_root, org, username)
    if org_workspace_root and org:
        org_dir = org_memory_dir(org_workspace_root, org)
    return user_dir, org_dir


__all__ = [
    "HOME_ENV_VAR",
    "MEMORY_DIRNAME",
    "ORG_WORKSPACE_ENV_VAR",
    "USER_WORKSPACE_ENV_VAR",
    "bundle_dir",
    "knowledge_root",
    "memory_bundle_dir",
    "org_memory_dir",
    "slugify",
    "user_memory_dir",
    "workspace_memory_dirs",
]
