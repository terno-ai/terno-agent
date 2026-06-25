"""SQLModel persistence for chat sessions and messages."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Session, select
from terno_agent.core.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)

import database
from models import ChatMessage, ChatSession


class SessionStore:
    """Thread-safe SQLite-backed store for chat sessions and their messages."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        database.create_db_and_tables()

    def close(self) -> None:
        return None

    def upsert_session(self, session_id: str) -> None:
        with self._lock, Session(database.engine) as session:
            chat_session = session.get(ChatSession, session_id)
            if chat_session is None:
                session.add(ChatSession(id=session_id))
                session.commit()

    def save_history(self, session_id: str, history: list[Message]) -> None:
        """Replace all stored messages for the session with the given history."""
        messages = [
            ChatMessage.from_payload(
                session_id=session_id,
                idx=idx,
                role=_role_str(message),
                payload=_encode_message(message),
            )
            for idx, message in enumerate(history)
        ]
        with self._lock, Session(database.engine) as session:
            chat_session = session.get(ChatSession, session_id)
            if chat_session is None:
                chat_session = ChatSession(id=session_id)
                session.add(chat_session)

            existing = session.exec(
                select(ChatMessage).where(ChatMessage.session_id == session_id)
            ).all()
            for message in existing:
                session.delete(message)

            for message in messages:
                session.add(message)

            chat_session.updated_at = datetime.now(UTC).replace(tzinfo=None)
            session.add(chat_session)
            session.commit()

    def load_history(self, session_id: str) -> list[Message]:
        return [
            _decode_message(message.role, message.payload_dict())
            for message in self.load_messages(session_id)
        ]

    def load_messages(self, session_id: str) -> list[ChatMessage]:
        with self._lock, Session(database.engine) as session:
            return list(
                session.exec(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.idx)
                ).all()
            )

    def get_session(self, session_id: str) -> ChatSession | None:
        with self._lock, Session(database.engine) as session:
            return session.get(ChatSession, session_id)


def _role_str(message: Message) -> str:
    role = getattr(message, "role", None)
    return getattr(role, "value", str(role))


def _encode_message(message: Message) -> dict[str, Any]:
    data = asdict(message)
    data["role"] = _role_str(message)
    return data


def _decode_message(role: str, payload: dict[str, Any] | str) -> Message:
    data = json.loads(payload) if isinstance(payload, str) else payload
    # Be tolerant of any legacy rows stored as 'Role.USER' before the fix.
    role = role.rsplit(".", 1)[-1].lower()
    if role == "user":
        content = data.get("content", "")
        if isinstance(content, list):
            # Best-effort: only text parts round-trip; richer attachment parts
            # are collapsed to text since they can't be reliably reconstructed
            # without their on-disk paths.
            content = [
                TextPart(text=str(part.get("text", "")))
                for part in content
                if isinstance(part, dict)
            ]
        return UserMessage(content=content)
    if role == "assistant":
        tool_calls = [
            ToolCall(
                id=call["id"],
                name=call["name"],
                arguments=call.get("arguments", {}),
            )
            for call in data.get("tool_calls", [])
        ]
        return AssistantMessage(
            content=data.get("content", ""), tool_calls=tool_calls
        )
    if role == "tool":
        results = [
            ToolResult(
                call_id=result["call_id"],
                content=result["content"],
                is_error=bool(result.get("is_error", False)),
            )
            for result in data.get("results", [])
        ]
        return ToolResultMessage(results=results)
    if role == "system":
        return SystemMessage(content=data.get("content", ""))
    raise ValueError(f"Unknown message role: {role!r}")
