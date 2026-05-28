"""SQLite database setup for the FastAPI backend."""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends
from sqlmodel import Session, SQLModel, create_engine


BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env", override=True)

DB_PATH = Path(os.getenv("TERNO_DB_PATH", ".terno/terno.db"))
if not DB_PATH.is_absolute():
    DB_PATH = BACKEND_DIR / DB_PATH
DB_PATH = DB_PATH.expanduser().resolve()
SQLITE_URL = f"sqlite:///{DB_PATH}"

connect_args = {"check_same_thread": False}
engine = create_engine(SQLITE_URL, connect_args=connect_args)


def create_db_and_tables() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Import table models before create_all so SQLModel metadata is populated.
    import models  # noqa: F401

    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
