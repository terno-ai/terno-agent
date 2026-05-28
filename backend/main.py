"""FastAPI backend for the Terno SDK demo UI."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from database import DB_PATH, create_db_and_tables
from store import SessionStore
from terno import Agent, Config
from terno_agent.core.events import (
    IterationStart,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    TurnEnd,
)


logger = logging.getLogger("terno_agent.backend")
logging.basicConfig(level=logging.INFO)

api_router = APIRouter(prefix="/api")
ws_router = APIRouter()
BACKEND_ENV_PATH = Path(__file__).with_name(".env")


class ConfigResponse(BaseModel):
    config: dict[str, str | int | bool]


class ClientFrame(BaseModel):
    type: Literal["user_message", "cancel", "clear"]
    text: str | None = None


class ActiveChatSession:
    """Owns one SDK agent and serializes turns for a single chat session."""

    def __init__(self, session_id: str, config: Config, store: SessionStore) -> None:
        self.session_id = session_id
        self.store = store
        self.event_loop: asyncio.AbstractEventLoop | None = None
        self.event_queue: asyncio.Queue[dict[str, Any] | None] | None = None
        self.agent = Agent.from_config(_copy_config(config), on_event=self._on_event)
        self.lock = asyncio.Lock()

        try:
            store.upsert_session(session_id)
            saved = store.load_history(session_id)
        except Exception:  # noqa: BLE001
            logger.exception("failed to load persisted chat session %s", session_id)
        else:
            if saved:
                self.agent.history[:] = saved

    def bind_events(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[dict[str, Any] | None],
    ) -> None:
        self.event_loop = loop
        self.event_queue = queue

    def unbind_events(self) -> None:
        self.event_loop = None
        self.event_queue = None

    def close(self) -> None:
        self.agent.close()

    def _on_event(self, event: Any) -> None:
        if self.event_loop is None or self.event_queue is None:
            return
        payload = _event_to_payload(event)
        if payload is not None:
            self.event_loop.call_soon_threadsafe(self.event_queue.put_nowait, payload)


def create_config() -> Config:
    """Build the SDK config object from the backend's own .env file."""
    load_dotenv(BACKEND_ENV_PATH, override=True)
    return Config.from_env()


def _copy_config(config: Config) -> Config:
    return replace(
        config,
        sandbox_options=dict(config.sandbox_options),
        skill_paths=list(config.skill_paths),
        extra=dict(config.extra),
    )


def _sessions(request_or_websocket: Request | WebSocket) -> dict[str, ActiveChatSession]:
    return request_or_websocket.app.state.sessions


def _get_or_create_session(websocket: WebSocket, session_id: str) -> ActiveChatSession:
    sessions = _sessions(websocket)
    session = sessions.get(session_id)
    if session is None:
        logger.info("creating session %s", session_id)
        session = ActiveChatSession(
            session_id,
            websocket.app.state.terno_config,
            websocket.app.state.session_store,
        )
        sessions[session_id] = session
    return session


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.terno_config = create_config()
    logger.info("using sqlite database at %s", DB_PATH)
    create_db_and_tables()
    app.state.session_store = SessionStore()
    app.state.sessions = {}
    yield
    sessions: dict[str, ActiveChatSession] = app.state.sessions
    await asyncio.gather(
        *(asyncio.to_thread(session.close) for session in sessions.values()),
        return_exceptions=True,
    )
    app.state.session_store.close()


app = FastAPI(title="Terno SDK Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@api_router.get("/config", response_model=ConfigResponse)
async def read_config(request: Request) -> ConfigResponse:
    config: Config = request.app.state.terno_config
    return ConfigResponse(config=_public_config(config))


def _serialize_message(message: Any) -> dict[str, Any]:
    """Convert an SDK message dataclass to a JSON-safe dict."""
    if is_dataclass(message):
        data = asdict(message)
        data["role"] = str(getattr(message, "role", data.get("role", "")))
        return data
    return {"role": "unknown", "content": str(message)}


def _history_payload(agent: Agent) -> list[dict[str, Any]]:
    return [_serialize_message(m) for m in agent.history]


@ws_router.websocket("/chat/{session_id}")
async def chat_ws(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    try:
        UUID(session_id)
    except ValueError:
        await websocket.send_json({"type": "error", "message": "Invalid session id."})
        await websocket.close()
        return

    try:
        session = _get_or_create_session(websocket, session_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("session create failed")
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
        return
    loop = asyncio.get_running_loop()

    await websocket.send_json(
        {"type": "ready", "history": _history_payload(session.agent)}
    )

    try:
        while True:
            try:
                frame = ClientFrame.model_validate(await websocket.receive_json())
            except ValidationError as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
                continue

            kind = frame.type

            if kind == "user_message":
                if session.lock.locked():
                    await websocket.send_json(
                        {"type": "error", "message": "A turn is already in progress."}
                    )
                    continue
                text = (frame.text or "").strip()
                if not text:
                    continue
                await _run_turn(websocket, session, text, loop)

            elif kind == "cancel":
                session.agent.cancel()

            elif kind == "clear":
                if session.lock.locked():
                    await websocket.send_json(
                        {"type": "error", "message": "Cannot clear during a turn."}
                    )
                    continue
                session.agent.clear_history()
                saved = await _save_session_history(session)
                if not saved:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Chat cleared in memory, but database save failed.",
                        }
                    )
                await websocket.send_json(
                    {"type": "cleared", "history": _history_payload(session.agent)}
                )

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown frame type: {kind!r}"}
                )

    except WebSocketDisconnect:
        logger.info("client disconnected from session %s", session_id)


async def _run_turn(
    websocket: WebSocket,
    session: ActiveChatSession,
    text: str,
    loop: asyncio.AbstractEventLoop,
) -> None:
    async with session.lock:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        session.agent.reset_cancel()
        session.bind_events(loop, queue)

        async def forward() -> None:
            while True:
                payload = await queue.get()
                if payload is None:
                    return
                try:
                    await websocket.send_json(payload)
                except Exception:
                    return

        forwarder = asyncio.create_task(forward())

        await websocket.send_json({"type": "turn_start"})

        try:
            await asyncio.to_thread(session.agent.run, text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent run failed")
            await websocket.send_json({"type": "error", "message": str(exc)})
        finally:
            await queue.put(None)
            await forwarder
            session.unbind_events()
            saved = await _save_session_history(session)
            if not saved:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Turn completed, but database save failed.",
                    }
                )

        await websocket.send_json(
            {"type": "turn_complete", "history": _history_payload(session.agent)}
        )


async def _save_session_history(session: ActiveChatSession) -> bool:
    try:
        await asyncio.to_thread(
            session.store.save_history,
            session.session_id,
            list(session.agent.history),
        )
    except Exception:  # noqa: BLE001
        logger.exception("failed to save chat session %s", session.session_id)
        return False
    return True


def _event_to_payload(event: Any) -> dict[str, Any] | None:
    if isinstance(event, IterationStart):
        return {"type": "iteration_start", "iteration": event.iteration}
    if isinstance(event, TextDelta):
        return {"type": "text_delta", "text": event.text}
    if isinstance(event, ToolCallEvent):
        return {
            "type": "tool_call",
            "id": event.call.id,
            "name": event.call.name,
            "arguments": event.call.arguments,
        }
    if isinstance(event, ToolResultEvent):
        return {
            "type": "tool_result",
            "call_id": event.result.call_id,
            "content": event.result.content,
            "is_error": event.result.is_error,
        }
    if isinstance(event, TurnEnd):
        return {"type": "turn_end", "message": _serialize_message(event.message)}
    return None


def _public_config(config: Config) -> dict[str, str | int | bool]:
    return {
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
        "llm_api_key": "configured" if config.llm_api_key else "missing",
        "sandbox": config.sandbox,
        "sandbox_fallback": config.sandbox_fallback or "disabled",
        "mcp_enabled": config.mcp_enabled,
        "skills_enabled": config.skills_enabled,
        "memory_enabled": config.memory_enabled,
        "attachments_enabled": config.attachments_enabled,
    }


app.include_router(api_router)
app.include_router(ws_router)
