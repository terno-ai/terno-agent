"""Path resolution for global vs workdir memory dirs."""

from __future__ import annotations

from pathlib import Path

import pytest

from terno_agent.memory.paths import (
    GLOBAL_ENV_VAR,
    global_memory_dir,
    resolve_dir_for_type,
    workdir_memory_dir,
)
from terno_agent.memory.types import MemoryType


def test_global_dir_uses_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "custom_memory"
    monkeypatch.setenv(GLOBAL_ENV_VAR, str(target))
    assert global_memory_dir() == target.resolve()


def test_global_dir_falls_back_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GLOBAL_ENV_VAR, raising=False)
    expected = (Path.home() / ".terno_agent" / "memory").resolve()
    assert global_memory_dir() == expected


def test_workdir_dir_is_under_workdir(tmp_path: Path) -> None:
    assert workdir_memory_dir(tmp_path) == (tmp_path / ".terno" / "memory").resolve()


def test_resolve_dir_for_type_routes_by_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_target = tmp_path / "g"
    monkeypatch.setenv(GLOBAL_ENV_VAR, str(global_target))
    assert resolve_dir_for_type(MemoryType.USER, tmp_path) == global_target.resolve()
    assert resolve_dir_for_type(MemoryType.FEEDBACK, tmp_path) == global_target.resolve()
    assert resolve_dir_for_type(MemoryType.PROJECT, tmp_path) == (
        tmp_path / ".terno" / "memory"
    ).resolve()
    assert resolve_dir_for_type(MemoryType.REFERENCE, tmp_path) == (
        tmp_path / ".terno" / "memory"
    ).resolve()
