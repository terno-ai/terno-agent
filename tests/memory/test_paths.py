"""Path resolution for the single ``.terno/memory`` directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from terno_agent.memory.paths import HOME_ENV_VAR, memory_dir


def test_memory_dir_defaults_to_workdir_dot_terno(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(HOME_ENV_VAR, raising=False)
    assert memory_dir(tmp_path) == (tmp_path / ".terno" / "memory").resolve()


def test_memory_dir_honors_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "custom_memory"
    monkeypatch.setenv(HOME_ENV_VAR, str(target))
    assert memory_dir(tmp_path) == target.resolve()
