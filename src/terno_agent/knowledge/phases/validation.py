"""Phase 4 — Question Generation & SQL Validation.

Generates a layered question bank (Tier 1 lookups through Tier 5
diagnostics) and, for each question, runs 4-5 SQL approaches in
parallel, then asks the user which one is correct. Validated pairs
are saved as canonical examples for the agent to learn from.

Depends on the schema (must exist) and prefers the descriptions
(better questions when they're ready).

Output sink: store.write_example(question, sql, output)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum

from terno_agent.knowledge.base import Phase, Task, TaskResult, TaskStatus
from terno_agent.knowledge.context import TaskContext
from terno_agent.knowledge.prompts import PromptOption, UserPrompt

PHASE = "validation"


class Tier(int, Enum):
    LOOKUP = 1       # single table, simple aggregate
    JOIN = 2         # multi-table grouping
    TIME_SERIES = 3  # date manipulation, windowing
    COMPARATIVE = 4  # subqueries, period comparison
    DIAGNOSTIC = 5   # open-ended decomposition


@dataclass(slots=True)
class GeneratedQuestion:
    tier: Tier
    text: str


class QuestionGenerationTask(Task):
    """Produce a tiered question bank conditioned on schema + descriptions."""

    name = "question_generation"
    description = "Generate Tier 1-5 questions for the connected DB."

    async def run(self, ctx: TaskContext) -> TaskResult:
        await ctx.phase.schema_ready.wait()
        # Descriptions are optional but improve quality; wait briefly if pending.
        if not ctx.phase.descriptions_ready.is_set():
            await ctx.phase.descriptions_ready.wait()

        # TODO(impl): prompt ctx.llm per tier with relevant schema slice
        # + descriptions. Aim for ~5 questions per tier.
        questions: list[GeneratedQuestion] = []
        ctx.phase.artifacts["questions"] = questions
        return TaskResult(
            task=self.name,
            status=TaskStatus.COMPLETED,
            output={"count": len(questions), "tiers": [t.name for t in Tier]},
        )


class SQLValidationTask(Task):
    """For each question, race N candidate SQLs and let the user pick winners."""

    name = "sql_validation"
    description = "Generate N candidate SQLs per question; user picks the correct one(s)."
    depends_on = ("question_generation",)
    approaches: int = 5

    async def run(self, ctx: TaskContext) -> TaskResult:
        questions: list[GeneratedQuestion] = ctx.phase.artifacts.get("questions", [])
        saved = 0
        for q in questions:
            saved += await self._validate_one(ctx, q)
        return TaskResult(
            task=self.name,
            status=TaskStatus.COMPLETED,
            output={"saved_examples": saved},
        )

    async def _validate_one(self, ctx: TaskContext, q: GeneratedQuestion) -> int:
        # TODO(impl): generate N candidate SQLs via ctx.llm and execute
        # them in parallel against ctx.db. Each entry is (sql, result).
        candidates: list[tuple[str, str]] = await asyncio.gather(
            *(self._one_approach(ctx, q, i) for i in range(self.approaches))
        )

        options = [
            PromptOption(
                label=f"Approach {i + 1}",
                value=str(i),
                description=(sql[:120] + "...") if len(sql) > 120 else sql,
            )
            for i, (sql, _output) in enumerate(candidates)
        ]
        response = await ctx.ask(
            UserPrompt.new(
                phase=PHASE,
                task=self.name,
                question=f"[{q.tier.name}] {q.text} — which SQL(s) are correct?",
                options=options,
                multi_select=True,
                allow_text=True,
                text_label="Notes on why an approach is wrong",
            )
        )
        saved = 0
        for idx_str in response.selected:
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            sql, output = candidates[idx]
            ctx.store.write_example(q.text, sql, output)
            saved += 1
        return saved

    async def _one_approach(
        self, ctx: TaskContext, q: GeneratedQuestion, i: int
    ) -> tuple[str, str]:
        # TODO(impl): one LLM call to draft SQL + ctx.db.execute() to run it.
        return ("", "")


class ValidationPhase(Phase):
    name = PHASE
    description = "Phase 4: generate tiered questions and validate SQL with the user."

    def build_tasks(self) -> list[Task]:
        return [QuestionGenerationTask(), SQLValidationTask()]


__all__ = [
    "GeneratedQuestion",
    "QuestionGenerationTask",
    "SQLValidationTask",
    "Tier",
    "ValidationPhase",
]
