"""SQLAlchemy-based database wrapper.

The `Database` object owns a single `Engine`. It exposes the small surface the
DB agent actually needs: list tables, describe a table, run a read-only query.

Writes are not blocked at the engine level — callers are expected to run with
a read-only DB user when running against production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, MetaData, create_engine, inspect, text
from sqlalchemy.engine import Result


@dataclass(slots=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    row_count: int
    truncated: bool


class Database:
    def __init__(self, url: str, *, echo: bool = False) -> None:
        self.url = url
        self._engine: Engine = create_engine(url, echo=echo, future=True)
        self._metadata = MetaData()

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def dialect(self) -> str:
        return self._engine.dialect.name

    def list_tables(self, schema: str | None = None) -> list[str]:
        return sorted(inspect(self._engine).get_table_names(schema=schema))

    def describe_table(self, name: str, schema: str | None = None) -> dict[str, Any]:
        insp = inspect(self._engine)
        cols = insp.get_columns(name, schema=schema)
        pks = insp.get_pk_constraint(name, schema=schema)
        fks = insp.get_foreign_keys(name, schema=schema)
        indexes = insp.get_indexes(name, schema=schema)
        return {
            "table": name,
            "schema": schema,
            "columns": [
                {
                    "name": c["name"],
                    "type": str(c["type"]),
                    "nullable": bool(c.get("nullable", True)),
                    "default": c.get("default"),
                }
                for c in cols
            ],
            "primary_key": pks.get("constrained_columns", []),
            "foreign_keys": [
                {
                    "columns": fk.get("constrained_columns"),
                    "references": {
                        "schema": fk.get("referred_schema"),
                        "table": fk.get("referred_table"),
                        "columns": fk.get("referred_columns"),
                    },
                }
                for fk in fks
            ],
            "indexes": [
                {"name": ix["name"], "columns": ix["column_names"], "unique": ix["unique"]}
                for ix in indexes
            ],
        }

    def schema_overview(self, max_tables: int = 50) -> str:
        """Compact textual overview suitable for an LLM system prompt."""
        tables = self.list_tables()
        if not tables:
            return f"(no tables found in {self.dialect} database)"
        shown = tables[:max_tables]
        lines = [f"Dialect: {self.dialect}", f"Tables ({len(tables)}):"]
        for t in shown:
            try:
                info = self.describe_table(t)
                cols = ", ".join(f"{c['name']}:{c['type']}" for c in info["columns"][:12])
                lines.append(f"- {t}({cols}{'...' if len(info['columns']) > 12 else ''})")
            except Exception as exc:  # pragma: no cover - introspection failure
                lines.append(f"- {t} (introspection failed: {exc})")
        if len(tables) > max_tables:
            lines.append(f"... and {len(tables) - max_tables} more")
        return "\n".join(lines)

    def execute(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        max_rows: int = 200,
    ) -> QueryResult:
        with self._engine.begin() as conn:
            res: Result = conn.execute(text(sql), params or {})
            if res.returns_rows:
                fetched = res.fetchmany(max_rows + 1)
                truncated = len(fetched) > max_rows
                rows = [tuple(r) for r in fetched[:max_rows]]
                return QueryResult(
                    columns=list(res.keys()),
                    rows=rows,
                    row_count=len(rows),
                    truncated=truncated,
                )
            return QueryResult(columns=[], rows=[], row_count=res.rowcount, truncated=False)
