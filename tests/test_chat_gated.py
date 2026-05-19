"""`terno chat` no longer requires an editable install.

The previous gate has been removed: chat is available in any install
and prompts up-front for optional deep research.
"""

from __future__ import annotations

import argparse

import pytest

from terno_agent import cli as cli_mod


def test_chat_starts_without_editable_gate(monkeypatch):
    """`_cmd_chat` should build the orchestrator regardless of install kind.

    We stub `Orchestrator.from_env` to raise a sentinel so we know we
    reached construction (we don't actually want to talk to an LLM).
    """

    class _BoomError(RuntimeError):
        pass

    def _from_env(**_kwargs):
        raise _BoomError("reached construction")

    monkeypatch.setattr(
        cli_mod.Orchestrator, "from_env",
        classmethod(lambda cls, **kw: _from_env(**kw)),
    )

    args = argparse.Namespace(quiet=True)
    with pytest.raises(_BoomError):
        cli_mod._cmd_chat(args)
