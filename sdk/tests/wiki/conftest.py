"""Shared fixtures for OKF tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from terno_agent.core.messages import AssistantMessage
from terno_agent.llm.base import LLMResponse


class ScriptedLLM:
    """LLM stub that always returns the same enrichment JSON for any table."""

    model = "scripted"

    def __init__(self, enrichment: dict | None = None) -> None:
        self.calls = 0
        self._payload = enrichment or {
            "summary": "Enriched summary.",
            "overview": "An enriched overview paragraph.",
            "columns": {"status": "1=active, 0=inactive"},
            "notes": ["status is an enum: 1=active, 0=inactive"],
        }

    def complete(self, messages, tools=None, **_kwargs) -> LLMResponse:
        self.calls += 1
        content = "```json\n" + json.dumps(self._payload) + "\n```"
        return LLMResponse(
            message=AssistantMessage(content=content), stop_reason="end_turn"
        )


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def workdir(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path
