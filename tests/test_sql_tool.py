import json

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.db.connection import Database
from terno_agent.tools.sql import DescribeTableTool, ListTablesTool, SqlQueryTool


@pytest.fixture()
def db():
    d = Database("sqlite:///:memory:")
    with d.engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER)"
        )
        conn.exec_driver_sql("INSERT INTO users (name, age) VALUES ('alice', 30), ('bob', 25)")
    return d


def test_list_tables(db):
    out = json.loads(ListTablesTool(db).run())
    assert out == ["users"]


def test_describe_table(db):
    out = json.loads(DescribeTableTool(db).run(name="users"))
    cols = {c["name"] for c in out["columns"]}
    assert {"id", "name", "age"} <= cols
    assert out["primary_key"] == ["id"]


def test_sql_query_select(db):
    out = json.loads(SqlQueryTool(db).run(sql="SELECT name FROM users ORDER BY name"))
    assert out["columns"] == ["name"]
    assert out["rows"] == [["alice"], ["bob"]]


def test_sql_query_rejects_write(db):
    with pytest.raises(ToolError):
        SqlQueryTool(db).run(sql="DELETE FROM users")


def test_sql_query_allows_write_when_disabled(db):
    SqlQueryTool(db, read_only=False).run(sql="DELETE FROM users WHERE name = 'alice'")
    out = json.loads(SqlQueryTool(db).run(sql="SELECT COUNT(*) AS c FROM users"))
    assert out["rows"][0] == [1]
