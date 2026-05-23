"""Local attachment storage.

Attachments are copied into a content-addressed blob directory and indexed in
SQLite. The SQLite layer is intentionally simple: metadata plus extracted text
chunks, enough to assemble bounded prompt context without loading a large file
into memory.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class AttachmentRecord:
    id: str
    source_path: Path
    blob_path: Path
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    text_extracted: bool = False


class AttachmentStore:
    """Content-addressed local attachment store backed by SQLite metadata."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.blob_dir = self.root / "blobs"
        self.db_path = self.root / "attachments.sqlite3"
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def save(
        self,
        path: Path,
        *,
        mime_type: str,
        max_bytes: int | None = None,
    ) -> AttachmentRecord:
        source = Path(path).expanduser().resolve()
        size = source.stat().st_size
        if max_bytes is not None and max_bytes > 0 and size > max_bytes:
            raise ValueError(
                f"Attachment is too large: {source} is {size} bytes; limit is {max_bytes}"
            )

        sha256 = _hash_file(source)
        attachment_id = sha256[:16]
        suffix = source.suffix.lower()
        blob_path = self.blob_dir / f"{sha256}{suffix}"
        if not blob_path.exists():
            shutil.copyfile(source, blob_path)

        with self._connect() as conn:
            conn.execute(
                """
                insert into attachments (
                    id, source_path, blob_path, filename, mime_type, size_bytes, sha256
                )
                values (?, ?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                    source_path=excluded.source_path,
                    blob_path=excluded.blob_path,
                    filename=excluded.filename,
                    mime_type=excluded.mime_type,
                    size_bytes=excluded.size_bytes,
                    sha256=excluded.sha256
                """,
                (
                    attachment_id,
                    str(source),
                    str(blob_path),
                    source.name,
                    mime_type,
                    size,
                    sha256,
                ),
            )
        return AttachmentRecord(
            id=attachment_id,
            source_path=source,
            blob_path=blob_path,
            filename=source.name,
            mime_type=mime_type,
            size_bytes=size,
            sha256=sha256,
        )

    def replace_chunks(self, attachment_id: str, chunks: list[str]) -> None:
        with self._connect() as conn:
            conn.execute("delete from chunks where attachment_id = ?", (attachment_id,))
            conn.executemany(
                "insert into chunks (attachment_id, chunk_index, text) values (?, ?, ?)",
                ((attachment_id, idx, text) for idx, text in enumerate(chunks)),
            )
            conn.execute(
                "update attachments set text_extracted = ? where id = ?",
                (1 if chunks else 0, attachment_id),
            )

    def chunks_for_prompt(
        self,
        attachment_id: str,
        *,
        query: str,
        max_chars: int,
    ) -> list[str]:
        if max_chars <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "select text from chunks where attachment_id = ? order by chunk_index",
                (attachment_id,),
            ).fetchall()
        chunks = [str(row[0]) for row in rows]
        if not chunks:
            return []

        selected = _rank_chunks(chunks, query)
        out: list[str] = []
        used = 0
        for text in selected:
            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining].rstrip()
            if text:
                out.append(text)
                used += len(text)
        return out

    def _init_db(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists attachments (
                    id text primary key,
                    source_path text not null,
                    blob_path text not null,
                    filename text not null,
                    mime_type text not null,
                    size_bytes integer not null,
                    sha256 text not null,
                    text_extracted integer not null default 0,
                    created_at text not null default current_timestamp
                )
                """
            )
            conn.execute(
                """
                create table if not exists chunks (
                    attachment_id text not null,
                    chunk_index integer not null,
                    text text not null,
                    primary key (attachment_id, chunk_index),
                    foreign key (attachment_id) references attachments(id)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank_chunks(chunks: list[str], query: str) -> list[str]:
    terms = {
        token.lower()
        for token in query.replace("_", " ").replace("-", " ").split()
        if len(token) >= 3
    }
    if not terms:
        return chunks

    scored: list[tuple[int, int, str]] = []
    for idx, chunk in enumerate(chunks):
        lower = chunk.lower()
        score = sum(lower.count(term) for term in terms)
        scored.append((score, -idx, chunk))
    scored.sort(reverse=True)
    ranked = [chunk for score, _idx, chunk in scored if score > 0]
    if ranked:
        return ranked
    return chunks


def file_size(path: Path) -> int:
    return os.stat(path).st_size


__all__ = ["AttachmentRecord", "AttachmentStore", "file_size"]
