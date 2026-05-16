"""Phase 3 — Semantic Annotation.

LLM-drafted natural-language descriptions for tables and columns
(presented to the user for correction), value-level embeddings for
low-cardinality columns (so the agent can resolve "successful payments"
→ status IN ('success', 'Completed')), and a tribal-knowledge capture
for the gotchas that never get written down anywhere.

Depends on the schema landing first: tasks wait on
`PhaseContext.schema_ready` before starting real work.

Output sinks:
    store.write_description(...)  for table/column descriptions
    store.write_embedding(...)    for value-level vectors
    store.append_org_prompt(...)  for gotchas appended to org prompt
"""

from __future__ import annotations

from terno_agent.knowledge.base import Phase, Task, TaskResult, TaskStatus
from terno_agent.knowledge.context import TaskContext
from terno_agent.knowledge.prompts import PromptOption, UserPrompt

PHASE = "annotation"


class DescriptionsTask(Task):
    """LLM drafts descriptions; user accepts, edits, or rejects each."""

    name = "descriptions"
    description = "Draft per-table/column descriptions; collect corrections."

    async def run(self, ctx: TaskContext) -> TaskResult:
        # Cross-phase wait: schema must exist before we can describe it.
        await ctx.phase.schema_ready.wait()
        # Org prompt improves draft quality but isn't strictly required.
        if not ctx.phase.org_prompt_ready.is_set():
            await ctx.phase.org_prompt_ready.wait()

        # TODO(impl): for each table/column, draft via ctx.llm using the
        # org prompt + schema + sample rows, then present this prompt
        # per draft. Persist accepted text via store.write_description.
        sample = UserPrompt.new(
            phase=PHASE,
            task=self.name,
            question="Description for `orders.status` — keep, edit, or reject?",
            options=[
                PromptOption(
                    label="Keep as-is",
                    value="keep",
                    description="Lifecycle state of the order (pending, paid, ...).",
                ),
                PromptOption(label="Edit", value="edit"),
                PromptOption(label="Reject (regenerate)", value="reject"),
            ],
            multi_select=False,
            allow_text=True,
            text_label="If editing, paste the corrected description",
        )
        await ctx.ask(sample)
        ctx.phase.descriptions_ready.set()
        return TaskResult(
            task=self.name,
            status=TaskStatus.COMPLETED,
            output={"described": 0},
        )


class EmbeddingsTask(Task):
    """Index distinct values of low-cardinality columns into a vector store."""

    name = "embeddings"
    description = "Embed low-cardinality column values for fuzzy filter resolution."

    async def run(self, ctx: TaskContext) -> TaskResult:
        await ctx.phase.schema_ready.wait()
        # TODO(impl): pick columns with cardinality <= K from the profile,
        # embed each distinct value, and write via store.write_embedding.
        # This is the "index whole DB" piece — analogous to code indexing.
        return TaskResult(
            task=self.name,
            status=TaskStatus.COMPLETED,
            output={"vectors_written": 0},
        )


class GotchasTask(Task):
    """Capture tribal knowledge that won't surface from schema alone."""

    name = "gotchas"
    description = "Solicit soft-delete rules, TZ quirks, dedupe logic, etc."

    async def run(self, ctx: TaskContext) -> TaskResult:
        # Free-form heavy: options seed common patterns; text field is the point.
        prompt = UserPrompt.new(
            phase=PHASE,
            task=self.name,
            question="Anything that surprises new analysts about this data?",
            options=[
                PromptOption(
                    label="Soft deletes (rows kept; flag/column marks deletion)",
                    value="soft_deletes",
                ),
                PromptOption(label="Timezone inconsistencies", value="tz"),
                PromptOption(label="Duplicate rows needing dedupe", value="dupes"),
                PromptOption(label="Tables that should never be joined directly", value="no_join"),
                PromptOption(label="Known bad data in some date ranges", value="bad_ranges"),
            ],
            multi_select=True,
            allow_text=True,
            text_label="Add anything specific to your data",
        )
        response = await ctx.ask(prompt)
        # TODO(impl): synthesize selections + free text into a paragraph
        # and append to the org prompt via store.append_org_prompt.
        return TaskResult(
            task=self.name,
            status=TaskStatus.COMPLETED,
            output={"tags": list(response.selected), "text": response.text},
        )


class AnnotationPhase(Phase):
    name = PHASE
    description = "Phase 3: describe, embed, and capture tribal knowledge."

    def build_tasks(self) -> list[Task]:
        return [DescriptionsTask(), EmbeddingsTask(), GotchasTask()]


__all__ = [
    "AnnotationPhase",
    "DescriptionsTask",
    "EmbeddingsTask",
    "GotchasTask",
]
