"""`terno chat` builds the agent on entry without any install-time gating."""

from __future__ import annotations

import argparse

import pytest

from terno_agent import cli as cli_mod


def test_chat_constructs_terno_agent(monkeypatch):
    """`_cmd_chat` should build the TernoAgent regardless of install kind.

    We stub `TernoAgent.from_env` to raise a sentinel so we know we
    reached construction (we don't actually want to talk to an LLM).
    """

    class _BoomError(RuntimeError):
        pass

    def _from_env(**_kwargs):
        raise _BoomError("reached construction")

    monkeypatch.setattr(
        cli_mod.TernoAgent,
        "from_env",
        classmethod(lambda cls, **kw: _from_env(**kw)),
    )

    args = argparse.Namespace(quiet=True)
    with pytest.raises(_BoomError):
        cli_mod._cmd_chat(args)
