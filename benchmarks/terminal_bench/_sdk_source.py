"""Helpers for installing the local SDK into benchmark containers."""

from __future__ import annotations

import shutil
from pathlib import Path

CONTAINER_SDK_PATH = "/installed-agent/terno-agent-sdk"
DEFAULT_LOCAL_SDK_PATH = Path(__file__).resolve().parents[2] / "sdk"


def stage_sdk_source(source: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("pyproject.toml", "README.md", "uv.lock"):
        path = source / filename
        if path.exists():
            shutil.copy2(path, target / filename)
    shutil.copytree(
        source / "src",
        target / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    return target
