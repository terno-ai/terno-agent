from __future__ import annotations

from terno_agent.agents.base import BaseAgent
from terno_agent.db.connection import Database
from terno_agent.llm.base import LLMClient
from terno_agent.prompts.database import DATABASE_PROMPT
from terno_agent.tools.sql import DescribeTableTool, ListTablesTool, SqlQueryTool


class DatabaseAgent(BaseAgent):
    name = "database"

    def __init__(
        self,
        llm: LLMClient,
        db: Database,
        *,
        max_rows: int = 200,
        read_only: bool = True,
        on_event=None,
    ) -> None:
        tools = [
            ListTablesTool(db),
            DescribeTableTool(db),
            SqlQueryTool(db, max_rows=max_rows, read_only=read_only),
        ]
        system = DATABASE_PROMPT + "\n\nDatabase context:\n" + db.schema_overview()
        super().__init__(llm, system, tools, on_event=on_event)
