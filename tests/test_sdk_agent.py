"""SDK surface — `from terno import Agent` and constructor behavior."""

from __future__ import annotations

import pytest

import terno
import terno_agent
from terno_agent.config import Config
from terno_agent.sdk import Agent


def test_terno_shim_exports_same_agent():
    assert terno.Agent is terno_agent.Agent
    assert terno.Agent is Agent
    assert terno.__version__ == terno_agent.__version__


def test_agent_kwargs_override_env(monkeypatch):
    """Explicit kwargs should win over env-derived defaults, and the
    constructor must not actually contact an LLM provider."""
    monkeypatch.delenv("TERNO_DATABASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    captured: dict[str, Config] = {}

    def _fake_from_config(cls, config, *, on_event=None):
        captured["config"] = config
        return object()

    monkeypatch.setattr(
        "terno_agent.sdk.Orchestrator.from_config",
        classmethod(_fake_from_config),
    )

    agent = Agent(
        api_key="sk-test",
        database_url="sqlite:///./demo.db",
        provider="anthropic",
        model="claude-opus-4-7",
        sandbox="local",
    )
    cfg = captured["config"]
    assert cfg.llm_api_key == "sk-test"
    assert cfg.database_url == "sqlite:///./demo.db"
    assert cfg.llm_provider == "anthropic"
    assert cfg.llm_model == "claude-opus-4-7"
    assert cfg.sandbox == "local"
    assert agent.config is cfg


def test_agent_run_delegates_to_orchestrator(monkeypatch):
    sentinel = object()

    class _Stub:
        def __init__(self):
            self.calls: list[str] = []

        def run(self, question: str):
            self.calls.append(question)
            return sentinel

        def ask(self, question: str):
            return self.run(question)

    stub = _Stub()
    monkeypatch.setattr(
        "terno_agent.sdk.Orchestrator.from_config",
        classmethod(lambda cls, config, **_kw: stub),
    )

    agent = Agent(
        api_key="sk-test",
        database_url="sqlite:///./demo.db",
        sandbox="local",
    )
    assert agent.run("hello") is sentinel
    assert agent.ask("again") is sentinel
    assert stub.calls == ["hello", "again"]


def test_agent_from_env_uses_config(monkeypatch):
    monkeypatch.setenv("TERNO_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    monkeypatch.setenv("TERNO_DATABASE_URL", "sqlite:///./envdb.db")
    monkeypatch.setenv("TERNO_SANDBOX", "local")

    captured: dict[str, Config] = {}
    monkeypatch.setattr(
        "terno_agent.sdk.Orchestrator.from_config",
        classmethod(lambda cls, config, **_kw: captured.setdefault("c", config)),
    )

    Agent.from_env()
    cfg = captured["c"]
    assert cfg.llm_api_key == "sk-env"
    assert cfg.database_url == "sqlite:///./envdb.db"
    assert cfg.sandbox == "local"


def test_invalid_sandbox_rejected(monkeypatch):
    monkeypatch.setattr(
        "terno_agent.sdk.Orchestrator.from_config",
        classmethod(lambda cls, config, **_kw: object()),
    )
    with pytest.raises(Exception):
        Agent(sandbox="not-a-real-sandbox")
