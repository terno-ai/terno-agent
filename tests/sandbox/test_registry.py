"""Registry + factory + entry-point discovery for sandbox plugins."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

from terno_agent.config import Config
from terno_agent.core.exceptions import ConfigError
from terno_agent.sandbox import (
    ExecutionResult,
    available_sandboxes,
    create_sandbox,
    register_sandbox,
)
from terno_agent.sandbox import registry as registry_module
from terno_agent.sandbox.local import LocalSandbox


@pytest.fixture(autouse=True)
def _reset_registry():
    """Clear plugin registrations + EP-load flag so tests don't leak."""
    registry_module._reset_for_tests()
    yield
    registry_module._reset_for_tests()


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


@dataclass
class _FakeSandbox:
    options: dict[str, Any]

    def __init__(self, **options: Any) -> None:
        self.options = options

    def run_python(self, code: str, *, timeout_s: int = 30, env=None) -> ExecutionResult:
        return ExecutionResult(stdout=f"ran:{code}", stderr="", exit_code=0)


def _fake_factory(**options: Any) -> _FakeSandbox:
    return _FakeSandbox(**options)


# --------------------------------------------------------------------------- #
# Built-ins
# --------------------------------------------------------------------------- #


def test_create_local_sandbox_builtin():
    sb = create_sandbox("local")
    assert isinstance(sb, LocalSandbox)


def test_available_sandboxes_includes_builtins():
    names = available_sandboxes()
    assert "docker" in names
    assert "local" in names


# --------------------------------------------------------------------------- #
# Unknown / bad inputs
# --------------------------------------------------------------------------- #


def test_unknown_kind_lists_available():
    with pytest.raises(ConfigError) as excinfo:
        create_sandbox("nonexistent-xyz")
    msg = str(excinfo.value)
    assert "nonexistent-xyz" in msg
    assert "docker" in msg
    assert "local" in msg


def test_empty_kind_rejected():
    with pytest.raises(ConfigError):
        create_sandbox("")


def test_bad_kwargs_become_config_error():
    register_sandbox("fake", _fake_factory)
    # _FakeSandbox.__init__ accepts arbitrary kwargs so kwargs DO NOT raise.
    # Use a sandbox class that does NOT accept arbitrary kwargs:
    class _StrictSandbox:
        def __init__(self) -> None:
            pass

        def run_python(self, code, *, timeout_s=30, env=None):
            return ExecutionResult(stdout="", stderr="", exit_code=0)

    register_sandbox("strict", _StrictSandbox)
    with pytest.raises(ConfigError) as excinfo:
        create_sandbox("strict", unknown_kwarg="x")
    assert "strict" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Programmatic registration
# --------------------------------------------------------------------------- #


def test_register_sandbox_works():
    register_sandbox("fake", _fake_factory)
    sb = create_sandbox("fake")
    assert isinstance(sb, _FakeSandbox)


def test_register_passes_options_as_kwargs():
    register_sandbox("fake", _fake_factory)
    sb = create_sandbox("fake", level=3, label="x")
    assert isinstance(sb, _FakeSandbox)
    assert sb.options == {"level": 3, "label": "x"}


def test_register_rejects_non_callable():
    with pytest.raises(ConfigError):
        register_sandbox("bad", "not-a-callable")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Import-string resolution
# --------------------------------------------------------------------------- #


def test_import_string_resolution(monkeypatch):
    # Stash a fake module so the import resolver can find it.
    mod = types.ModuleType("fake_pkg_for_sandbox_test")
    mod.MySandbox = _FakeSandbox  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_pkg_for_sandbox_test", mod)
    sb = create_sandbox("fake_pkg_for_sandbox_test:MySandbox", k="v")
    assert isinstance(sb, _FakeSandbox)
    assert sb.options == {"k": "v"}


def test_import_string_bad_module():
    with pytest.raises(ConfigError) as excinfo:
        create_sandbox("definitely_not_a_real_module_xyz:Cls")
    assert "definitely_not_a_real_module_xyz" in str(excinfo.value)


def test_import_string_missing_attr(monkeypatch):
    mod = types.ModuleType("fake_pkg_for_sandbox_test_2")
    monkeypatch.setitem(sys.modules, "fake_pkg_for_sandbox_test_2", mod)
    with pytest.raises(ConfigError) as excinfo:
        create_sandbox("fake_pkg_for_sandbox_test_2:Missing")
    assert "Missing" in str(excinfo.value)


def test_import_string_format_rejected():
    with pytest.raises(ConfigError):
        create_sandbox("no_colon_here")  # falls into name lookup, not import string


def test_malformed_import_string():
    with pytest.raises(ConfigError):
        create_sandbox(":NoModule")
    with pytest.raises(ConfigError):
        create_sandbox("mod_only:")


# --------------------------------------------------------------------------- #
# Entry-point discovery
# --------------------------------------------------------------------------- #


class _FakeEntryPoint:
    def __init__(self, name: str, target):
        self.name = name
        self._target = target

    def load(self):
        return self._target


def test_entry_points_loaded_lazily(monkeypatch):
    calls = {"n": 0}

    def fake_eps(group):
        calls["n"] += 1
        assert group == "terno_agent.sandboxes"
        return [_FakeEntryPoint("plugin_x", _fake_factory)]

    monkeypatch.setattr(registry_module.metadata, "entry_points", fake_eps)
    # First lookup triggers the discovery.
    sb = create_sandbox("plugin_x")
    assert isinstance(sb, _FakeSandbox)
    # Second lookup must NOT re-scan entry points.
    create_sandbox("plugin_x")
    assert calls["n"] == 1


def test_entry_point_load_failure_warns_and_skips(monkeypatch, capsys):
    class _BoomEP:
        name = "boom"

        def load(self):
            raise RuntimeError("import failed")

    def fake_eps(group):
        return [_BoomEP(), _FakeEntryPoint("good", _fake_factory)]

    monkeypatch.setattr(registry_module.metadata, "entry_points", fake_eps)
    # The good plugin still works even though 'boom' failed.
    sb = create_sandbox("good")
    assert isinstance(sb, _FakeSandbox)
    captured = capsys.readouterr()
    assert "boom" in captured.err
    # 'boom' is NOT registered.
    with pytest.raises(ConfigError):
        create_sandbox("boom")


def test_entry_point_non_callable_skipped(monkeypatch, capsys):
    monkeypatch.setattr(
        registry_module.metadata,
        "entry_points",
        lambda group: [_FakeEntryPoint("notcall", "not-a-function")],
    )
    with pytest.raises(ConfigError):
        create_sandbox("notcall")
    captured = capsys.readouterr()
    assert "notcall" in captured.err


# --------------------------------------------------------------------------- #
# Config integration
# --------------------------------------------------------------------------- #


def test_config_accepts_plugin_name_without_validation():
    # Previously this would raise ConfigError because "qemu" wasn't in the
    # hardcoded set. The registry now owns that check.
    cfg = Config(llm_provider="anthropic", llm_api_key="x", sandbox="qemu")
    assert cfg.sandbox == "qemu"


def test_config_keeps_import_string_case(monkeypatch):
    monkeypatch.setenv("TERNO_SANDBOX", "My_Pkg.Sub:CamelCase")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = Config.from_env()
    assert cfg.sandbox == "My_Pkg.Sub:CamelCase"


def test_config_lowercases_short_names(monkeypatch):
    monkeypatch.setenv("TERNO_SANDBOX", "Local")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = Config.from_env()
    assert cfg.sandbox == "local"


def test_config_parses_sandbox_options(monkeypatch):
    monkeypatch.setenv("TERNO_SANDBOX_OPTIONS", "image=python:3.13 , timeout=60")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = Config.from_env()
    assert cfg.sandbox_options == {"image": "python:3.13", "timeout": "60"}


def test_config_rejects_bad_options_format(monkeypatch):
    monkeypatch.setenv("TERNO_SANDBOX_OPTIONS", "no-equals")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    with pytest.raises(ConfigError):
        Config.from_env()
