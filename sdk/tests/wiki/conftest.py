"""Shared fixtures for OKF tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa

from terno_agent.core.messages import AssistantMessage
from terno_agent.db.connection import Database
from terno_agent.llm.base import LLMResponse


@pytest.fixture
def sqlite_db(tmp_path: Path) -> Database:
    """A small sqlite datasource with two related tables and a few rows."""
    url = f"sqlite:///{tmp_path / 'sales.db'}"
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "CREATE TABLE users ("
                "id INTEGER PRIMARY KEY, email TEXT, status INTEGER)"
            )
        )
        conn.execute(
            sa.text(
                "CREATE TABLE orders ("
                "id INTEGER PRIMARY KEY, "
                "user_id INTEGER REFERENCES users(id), amount REAL)"
            )
        )
        conn.execute(
            sa.text("INSERT INTO users VALUES (1,'a@b.com',1),(2,'c@d.com',0)")
        )
        conn.execute(sa.text("INSERT INTO orders VALUES (1,1,9.5),(2,1,3.0)"))
    return Database(url)


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
