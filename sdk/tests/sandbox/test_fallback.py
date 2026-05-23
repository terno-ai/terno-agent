"""Sandbox-initialization fallback behavior."""

from __future__ import annotations

import pytest

from terno_agent.agents import terno as terno_mod
from terno_agent.agents.terno import _init_sandbox
from terno_agent.config import Config
from terno_agent.core.exceptions import SandboxError
from terno_agent.sandbox.base import ExecutionResult


class _StubSandbox:
    def __init__(self, label: str) -> None:
        self.label = label

    def run_python(self, code: str, *, timeout_s: int = 30, env=None) -> ExecutionResult:
        return ExecutionResult(stdout=self.label, stderr="", exit_code=0)


def _scripted_factory(behaviors: dict[str, object]):
    """Return a `create_sandbox` replacement that consults `behaviors[kind]`.

    ``behaviors[kind]`` is either a Sandbox instance to return or an
    Exception instance to raise.
    """

    def fake_create_sandbox(kind: str, **_options):
        if kind not in behaviors:
            raise SandboxError(f"no scripted behavior for {kind!r}")
        outcome = behaviors[kind]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return fake_create_sandbox


def _cfg(**overrides) -> Config:
    base = {
        "llm_provider": "anthropic",
        "llm_api_key": "x",
        "sandbox": "docker",
        "sandbox_fallback": "local",
    }
    base.update(overrides)
    return Config(**base)


def test_primary_succeeds_no_fallback_attempted(monkeypatch, capsys):
    sb = _StubSandbox("docker-ok")
    monkeypatch.setattr(
        terno_mod, "create_sandbox", _scripted_factory({"docker": sb})
    )
    result = _init_sandbox(_cfg())
    assert result is sb
    captured = capsys.readouterr()
    assert captured.err == ""


def test_falls_back_to_local_when_docker_unavailable(monkeypatch, capsys):
    local_sb = _StubSandbox("local-ok")
    monkeypatch.setattr(
        terno_mod,
        "create_sandbox",
        _scripted_factory(
            {
                "docker": SandboxError("daemon unreachable"),
                "local": local_sb,
            }
        ),
    )
    result = _init_sandbox(_cfg())
    assert result is local_sb
    captured = capsys.readouterr()
    # Notice, not warning. No mention of "run_python disabled".
    assert "notice:" in captured.err
    assert "falling back to 'local'" in captured.err
    assert "warning" not in captured.err


def test_fallback_disabled_emits_warning_and_returns_none(monkeypatch, capsys):
    monkeypatch.setattr(
        terno_mod,
        "create_sandbox",
        _scripted_factory({"docker": SandboxError("daemon unreachable")}),
    )
    result = _init_sandbox(_cfg(sandbox_fallback=""))
    assert result is None
    captured = capsys.readouterr()
    assert "warning:" in captured.err
    assert "run_python tool will be disabled" in captured.err


def test_fallback_none_acts_as_disabled(monkeypatch, capsys):
    monkeypatch.setattr(
        terno_mod,
        "create_sandbox",
        _scripted_factory({"docker": SandboxError("nope")}),
    )
    result = _init_sandbox(_cfg(sandbox_fallback="none"))
    assert result is None
    captured = capsys.readouterr()
    assert "warning:" in captured.err
    # Should not have tried "none" as a fallback kind.
    assert "fallback 'none'" not in captured.err


def test_fallback_equal_to_primary_is_a_noop(monkeypatch, capsys):
    monkeypatch.setattr(
        terno_mod,
        "create_sandbox",
        _scripted_factory({"docker": SandboxError("nope")}),
    )
    result = _init_sandbox(_cfg(sandbox_fallback="docker"))
    assert result is None
    captured = capsys.readouterr()
    assert "warning:" in captured.err
    # We should not have logged that 'docker' was tried as a fallback.
    assert "fallback 'docker'" not in captured.err


def test_both_fail_warning_lists_both(monkeypatch, capsys):
    monkeypatch.setattr(
        terno_mod,
        "create_sandbox",
        _scripted_factory(
            {
                "docker": SandboxError("daemon"),
                "local": SandboxError("subprocess broken"),
            }
        ),
    )
    result = _init_sandbox(_cfg())
    assert result is None
    captured = capsys.readouterr()
    assert "warning:" in captured.err
    assert "docker" in captured.err
    assert "fallback 'local'" in captured.err


def test_sandbox_none_short_circuits(monkeypatch, capsys):
    called: list[str] = []

    def should_not_be_called(kind: str, **_options):
        called.append(kind)
        raise AssertionError("create_sandbox must not be called when sandbox='none'")

    monkeypatch.setattr(terno_mod, "create_sandbox", should_not_be_called)
    assert _init_sandbox(_cfg(sandbox="none")) is None
    assert called == []
    assert capsys.readouterr().err == ""


def test_env_default_includes_local_fallback(monkeypatch):
    monkeypatch.delenv("TERNO_SANDBOX_FALLBACK", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = Config.from_env()
    assert cfg.sandbox_fallback == "local"


def test_env_can_disable_fallback(monkeypatch):
    monkeypatch.setenv("TERNO_SANDBOX_FALLBACK", "none")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = Config.from_env()
    assert cfg.sandbox_fallback == "none"
