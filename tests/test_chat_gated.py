"""`terno chat` should refuse to run outside a development install."""

from __future__ import annotations

import argparse

import pytest

from terno_agent import cli as cli_mod


def test_chat_refuses_when_not_editable(monkeypatch, capsys):
    monkeypatch.setattr(cli_mod, "_is_editable_install", lambda: False)
    args = argparse.Namespace(quiet=False)
    rc = cli_mod._cmd_chat(args)
    err = capsys.readouterr().err
    assert rc == 2
    assert "development install" in err
    assert "pip install -e" in err


def test_chat_starts_in_editable_install(monkeypatch):
    """When editable, _cmd_chat builds an Orchestrator and runs the loop.

    We stub `Orchestrator.from_env` to raise a sentinel so we know we reached
    construction (we don't actually want to talk to an LLM in tests).
    """
    monkeypatch.setattr(cli_mod, "_is_editable_install", lambda: True)

    class _Boom(RuntimeError):
        pass

    def _from_env(**_kwargs):
        raise _Boom("reached construction")

    monkeypatch.setattr(cli_mod.Orchestrator, "from_env", classmethod(lambda cls, **kw: _from_env(**kw)))

    args = argparse.Namespace(quiet=True)
    with pytest.raises(_Boom):
        cli_mod._cmd_chat(args)


def test_is_editable_install_matches_current_install():
    """The current test run is itself an editable install (via uv pip install -e .)."""
    assert cli_mod._is_editable_install() is True
