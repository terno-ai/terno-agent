"""SQLModel table models for chat data stored in terno.db."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import field_validator
from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class ChatSession(SQLModel, table=True):
    """A chat conversation persisted in SQLite."""

    __tablename__: ClassVar[str] = "sessions"

    id: str = Field(primary_key=True, min_length=1)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ChatMessage(SQLModel, table=True):
    """One persisted message in a chat session history."""

    __tablename__: ClassVar[str] = "messages"
    _allowed_roles: ClassVar[set[str]] = {"system", "user", "assistant", "tool"}

    session_id: str = Field(foreign_key="sessions.id", primary_key=True, min_length=1)
    idx: int = Field(primary_key=True, ge=0)
    role: str = Field(index=True, min_length=1)
    payload: str = Field(sa_column=Column(Text, nullable=False))
    created_at: datetime = Field(default_factory=_now)

    @field_validator("role", mode="before")
    @classmethod
    def _validate_role(cls, value: Any) -> str:
        role = str(value)
        if role not in cls._allowed_roles:
            raise ValueError(f"unknown message role: {role!r}")
        return role

    @field_validator("payload", mode="before")
    @classmethod
    def _validate_payload(cls, value: Any) -> str:
        if isinstance(value, str):
            loaded = json.loads(value)
            if not isinstance(loaded, dict):
                raise ValueError("message payload must decode to a JSON object")
            return value
        if isinstance(value, dict):
            return json.dumps(value, default=str)
        raise ValueError("message payload must be a dict or JSON object string")

    @classmethod
    def from_payload(
        cls,
        *,
        session_id: str,
        idx: int,
        role: str,
        payload: dict[str, Any],
    ) -> ChatMessage:
        return cls(
            session_id=session_id,
            idx=idx,
            role=role,
            payload=json.dumps(payload, default=str),
        )

    def payload_dict(self) -> dict[str, Any]:
        loaded = json.loads(self.payload)
        if not isinstance(loaded, dict):
            raise ValueError("message payload must decode to a JSON object")
        return loaded
