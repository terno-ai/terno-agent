"""SQL tools backed by a `Database`."""

from __future__ import annotations

import json
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema
from terno_agent.db.connection import Database, QueryResult


class ListTablesTool:
    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_tables",
            description="List all tables in the connected database.",
            parameters={
                "type": "object",
                "properties": {
                    "schema": {
                        "type": "string",
                        "description": "Optional schema name (Postgres/SQL Server).",
                    }
                },
                "required": [],
            },
        )

    def run(self, **kwargs: Any) -> str:
        tables = self.db.list_tables(schema=kwargs.get("schema"))
        return json.dumps(tables)


class DescribeTableTool:
    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="describe_table",
            description="Return columns, primary key, foreign keys and indexes for a table.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Table name."},
                    "schema": {"type": "string", "description": "Optional schema name."},
                },
                "required": ["name"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        name = kwargs.get("name")
        if not name:
            raise ToolError("describe_table requires a 'name' argument.")
        info = self.db.describe_table(name, schema=kwargs.get("schema"))
        return json.dumps(info, default=str)


class SqlQueryTool:
    """Execute a read-only SQL statement and return results as a table."""

    def __init__(self, db: Database, *, max_rows: int = 200, read_only: bool = True) -> None:
        self.db = db
        self.max_rows = max_rows
        self.read_only = read_only

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="sql_query",
            description=(
                "Run a SQL query against the connected database and return rows. "
                "By default only SELECT/WITH/EXPLAIN are allowed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL statement to execute.",
                    }
                },
                "required": ["sql"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        sql = (kwargs.get("sql") or "").strip()
        if not sql:
            raise ToolError("sql_query requires a 'sql' argument.")
        if self.read_only and not _is_read_only(sql):
            raise ToolError(
                "Refusing to run non-read-only SQL. Only SELECT, WITH and EXPLAIN are allowed."
            )
        try:
            result = self.db.execute(sql, max_rows=self.max_rows)
        except Exception as exc:
            raise ToolError(f"SQL execution failed: {exc}") from exc
        return _render(result)


def _is_read_only(sql: str) -> bool:
    head = sql.lstrip().split(None, 1)[0].lower().rstrip(";")
    return head in {"select", "with", "explain", "show", "describe", "desc"}


def _render(r: QueryResult) -> str:
    if not r.columns:
        return f"OK ({r.row_count} rows affected)"
    out = {
        "columns": r.columns,
        "rows": [list(row) for row in r.rows],
        "row_count": r.row_count,
        "truncated": r.truncated,
    }
    return json.dumps(out, default=str)
