"""Render and parse the YAML frontmatter block of an OKF markdown file.

A frontmatter block is YAML delimited by ``---`` lines at the very top of
the file, followed by the markdown body::

    ---
    title: Users
    type: table
    summary: Registered end-users.
    ---

    ## Overview
    ...

These helpers are thin wrappers over ``yaml.safe_load`` / ``yaml.safe_dump``.
Keys are emitted in a stable, human-friendly order (``title``, ``type``,
``summary`` first, then the rest alphabetically) so bundles diff cleanly in
version control.
"""

from __future__ import annotations

from typing import Any

import yaml

_PREFERRED_ORDER = ("title", "type", "summary", "updated", "source")


def _ordered(frontmatter: dict[str, Any]) -> list[tuple[str, Any]]:
    keys = list(frontmatter)
    preferred = [k for k in _PREFERRED_ORDER if k in frontmatter]
    rest = sorted(k for k in keys if k not in _PREFERRED_ORDER)
    return [(k, frontmatter[k]) for k in (*preferred, *rest)]


def render(frontmatter: dict[str, Any], body: str) -> str:
    """Serialize ``frontmatter`` + ``body`` into an OKF markdown document."""
    # Dump each key separately to preserve our preferred ordering; safe_dump
    # on a dict would otherwise sort keys alphabetically.
    lines = ["---"]
    for key, value in _ordered(frontmatter):
        if value is None:
            continue
        chunk = yaml.safe_dump(
            {key: value}, default_flow_style=False, allow_unicode=True, sort_keys=False
        )
        lines.append(chunk.rstrip("\n"))
    lines.append("---")
    rendered_body = body.strip("\n")
    return "\n".join(lines) + ("\n\n" + rendered_body + "\n" if rendered_body else "\n")


def parse(text: str) -> tuple[dict[str, Any], str]:
    """Split an OKF document into ``(frontmatter, body)``.

    Returns ``({}, text)`` when no frontmatter block is present so callers can
    still recover the body of a malformed file.
    """
    if not text.startswith("---"):
        return {}, text.strip("\n")
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text.strip("\n")
    _, raw_frontmatter, body = parts
    try:
        loaded = yaml.safe_load(raw_frontmatter) or {}
    except yaml.YAMLError:
        loaded = {}
    if not isinstance(loaded, dict):
        loaded = {}
    return loaded, body.strip("\n")


__all__ = ["parse", "render"]
