"""The datasource knowledge-enrichment agent.

`DatasourceKnowledgeAgent` builds an OKF bundle describing a live datasource.
It crawls the schema deterministically (always) and, when an LLM is
available, layers semantic enrichment on top — prose overviews, per-column
descriptions, and gotchas (enum meanings, staleness, caveats).

    agent = DatasourceKnowledgeAgent(db=db, llm=llm, bundle=bundle)
    report = agent.build(tables=None, refresh=False)

The agent never blocks on enrichment: if ``llm`` is ``None`` or a call
fails, it falls back to introspection-only concepts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from terno_agent.core.messages import SystemMessage, UserMessage
from terno_agent.wiki.bundle import KnowledgeBundle
from terno_agent.wiki.concept import Concept
from terno_agent.wiki.paths import slugify

if TYPE_CHECKING:
    from terno_agent.db.connection import Database
    from terno_agent.llm.base import LLMClient

_ENRICH_SYSTEM = (
    "You are a data documentation expert. Given a database table's structure "
    "and a small sample of rows, write concise, accurate knowledge about it. "
    "Reply with ONLY a JSON object (no prose, no code fences) of the form:\n"
    '{"summary": "<one line>", "overview": "<1-3 sentence paragraph>", '
    '"columns": {"<column>": "<meaning>", ...}, '
    '"notes": ["<gotcha / enum meaning / caveat>", ...]}\n'
    "Only include columns you can meaningfully describe. Infer enum/code "
    "meanings from sample values when possible. Keep it factual."
)


@dataclass
class BuildReport:
    datasource: str
    tables_written: list[str] = field(default_factory=list)
    tables_skipped: list[str] = field(default_factory=list)
    tables_truncated: list[str] = field(default_factory=list)
    enriched: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "datasource": self.datasource,
            "tables_written": self.tables_written,
            "tables_skipped": self.tables_skipped,
            "tables_truncated": self.tables_truncated,
            "enriched": self.enriched,
        }


class DatasourceKnowledgeAgent:
    def __init__(
        self,
        *,
        db: Database,
        bundle: KnowledgeBundle,
        llm: LLMClient | None = None,
        max_tables: int = 50,
        sample_rows: int = 5,
        today: str | None = None,
    ) -> None:
        self.db = db
        self.bundle = bundle
        self.llm = llm
        self.max_tables = max_tables
        self.sample_rows = sample_rows
        self._today = today or date.today().isoformat()

    # ----- public API ---------------------------------------------------- #

    def build(
        self, tables: list[str] | None = None, *, refresh: bool = False
    ) -> BuildReport:
        print("-----start-----")
        all_tables = self.db.list_tables()
        wanted = set(tables) if tables else None
        selected = [t for t in all_tables if wanted is None or t in wanted]

        report = BuildReport(datasource=self.bundle.name, enriched=self.llm is not None)
        if self.max_tables and len(selected) > self.max_tables:
            report.tables_truncated = selected[self.max_tables :]
            selected = selected[: self.max_tables]

        for table in selected:
            concept_id = f"tables/{slugify(table)}"
            if not refresh and self.bundle.read_concept(concept_id) is not None:
                report.tables_skipped.append(table)
                continue
            info = self.db.describe_table(table)
            enrichment = self._enrich(table, info) if self.llm is not None else {}
            self.bundle.write_concept(
                self._table_concept(concept_id, table, info, enrichment)
            )
            report.tables_written.append(table)

        self.bundle.write_concept(self._overview_concept(all_tables, selected))
        self.bundle.rebuild_index()
        print(report.to_dict())
        print("-------end-------")
        return report

    # ----- concept construction ----------------------------------------- #

    def _table_concept(
        self,
        concept_id: str,
        table: str,
        info: dict[str, Any],
        enrichment: dict[str, Any],
    ) -> Concept:
        columns = info.get("columns", [])
        pk = info.get("primary_key", [])
        fks = info.get("foreign_keys", [])
        col_desc: dict[str, str] = enrichment.get("columns", {}) or {}

        summary = (enrichment.get("summary") or "").strip() or (
            f"{table} table ({len(columns)} columns)."
        )
        body_parts: list[str] = []

        overview = (enrichment.get("overview") or "").strip()
        if overview:
            body_parts.append(f"## Overview\n\n{overview}")

        body_parts.append(self._columns_section(columns, pk, col_desc))

        rel = self._relationships_section(fks)
        if rel:
            body_parts.append(rel)

        notes = [str(n).strip() for n in (enrichment.get("notes") or []) if str(n).strip()]
        if notes:
            note_lines = "\n".join(f"- {n}" for n in notes)
            body_parts.append(f"## Notes & Gotchas\n\n{note_lines}")

        metadata: dict[str, Any] = {
            "updated": self._today,
            "source": "introspection+llm" if enrichment else "introspection",
            "table": table,
            "columns": len(columns),
        }
        if pk:
            metadata["primary_key"] = list(pk)

        return Concept(
            concept_id=concept_id,
            title=table,
            type="table",
            summary=summary,
            body="\n\n".join(body_parts),
            metadata=metadata,
        )

    @staticmethod
    def _columns_section(
        columns: list[dict[str, Any]], pk: list[str], col_desc: dict[str, str]
    ) -> str:
        lines = [
            "## Columns",
            "",
            "| Column | Type | Nullable | Key | Description |",
            "| --- | --- | --- | --- | --- |",
        ]
        for c in columns:
            name = c.get("name", "")
            ctype = str(c.get("type", ""))
            nullable = "yes" if c.get("nullable", True) else "no"
            key = "PK" if name in pk else ""
            desc = (col_desc.get(name) or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{name}` | {ctype} | {nullable} | {key} | {desc} |")
        return "\n".join(lines)

    @staticmethod
    def _relationships_section(fks: list[dict[str, Any]]) -> str:
        if not fks:
            return ""
        lines = ["## Relationships", ""]
        for fk in fks:
            cols = ", ".join(f"`{c}`" for c in (fk.get("columns") or []))
            ref = fk.get("references", {}) or {}
            ref_table = ref.get("table") or "?"
            ref_cols = ", ".join(f"`{c}`" for c in (ref.get("columns") or []))
            link = f"[{ref_table}]({slugify(ref_table)}.md)" if ref_table != "?" else "?"
            lines.append(f"- {cols} → {link} ({ref_cols})")
        return "\n".join(lines)

    def _overview_concept(
        self, all_tables: list[str], selected: list[str]
    ) -> Concept:
        lines = [
            "## Overview",
            "",
            f"- **Dialect:** {self.db.dialect}",
            f"- **Tables:** {len(all_tables)} total, {len(selected)} documented",
            "",
            "## Tables",
            "",
        ]
        for table in sorted(selected):
            lines.append(f"- [{table}](tables/{slugify(table)}.md)")
        return Concept(
            concept_id="overview",
            title=f"{self.bundle.name} overview",
            type="datasource",
            summary=f"{self.db.dialect} datasource with {len(all_tables)} tables.",
            body="\n".join(lines),
            metadata={"updated": self._today, "source": "introspection"},
        )

    # ----- LLM enrichment ----------------------------------------------- #

    def _enrich(self, table: str, info: dict[str, Any]) -> dict[str, Any]:
        if self.llm is None:
            return {}
        try:
            payload = self._enrich_payload(table, info)
            response = self.llm.complete(
                [SystemMessage(_ENRICH_SYSTEM), UserMessage(payload)],
                tools=None,
                max_tokens=1500,
            )
            return self._parse_enrichment(response.message.content)
        except Exception:
            # Enrichment is best-effort; never block bundle creation.
            return {}

    def _enrich_payload(self, table: str, info: dict[str, Any]) -> str:
        cols = [
            {"name": c.get("name"), "type": str(c.get("type"))}
            for c in info.get("columns", [])
        ]
        parts = [
            f"Table: {table}",
            f"Columns: {json.dumps(cols)}",
            f"Primary key: {json.dumps(info.get('primary_key', []))}",
            f"Foreign keys: {json.dumps(info.get('foreign_keys', []))}",
        ]
        sample = self._sample(table)
        if sample:
            parts.append(f"Sample rows (up to {self.sample_rows}): {json.dumps(sample)}")
        return "\n".join(parts)

    def _sample(self, table: str) -> list[dict[str, Any]]:
        try:
            quoted = self.db.engine.dialect.identifier_preparer.quote(table)
            result = self.db.execute(
                f"SELECT * FROM {quoted} LIMIT {int(self.sample_rows)}",
                max_rows=self.sample_rows,
            )
            return [
                {col: _jsonable(val) for col, val in zip(result.columns, row, strict=False)}
                for row in result.rows
            ]
        except Exception:
            return []

    @staticmethod
    def _parse_enrichment(text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if raw.startswith("```"):
            # Strip ```json ... ``` fences.
            raw = raw.split("\n", 1)[-1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0]
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = ["BuildReport", "DatasourceKnowledgeAgent"]
