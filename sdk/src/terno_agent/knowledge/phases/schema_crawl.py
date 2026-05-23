"""Phase 2 — Schema Deep Crawl.

Fetches the live schema, then enriches it with PII flags, statistical
profiling, implicit foreign-key discovery, enum semantics, and
staleness/trust signals. The fetched schema is published via
`PhaseContext.schema_ready` so downstream phases (annotation,
validation) can start as soon as it lands.

Output sinks:
    store.upsert_dbi_model(...)   per table/column/relationship
    store.write_description(...)  for profiled column descriptions
"""

from __future__ import annotations

from terno_agent.knowledge.base import Phase, Task, TaskResult, TaskStatus
from terno_agent.knowledge.context import TaskContext
from terno_agent.knowledge.prompts import PromptOption, UserPrompt

PHASE = "schema_crawl"


class FetchSchemaTask(Task):
    """List tables/columns/declared FKs from the live DB."""

    name = "fetch_schema"
    description = "Crawl tables, columns, and declared foreign keys."

    async def run(self, ctx: TaskContext) -> TaskResult:
        # TODO(impl): walk ctx.db.list_tables() + describe_table() and
        # persist each as a dbi_model row. Skipping I/O in the stub.
        schema: dict = {"tables": [], "columns": [], "foreign_keys": []}
        ctx.phase.artifacts["schema"] = schema
        ctx.phase.schema_ready.set()
        return TaskResult(
            task=self.name,
            status=TaskStatus.COMPLETED,
            output={"tables": len(schema["tables"])},
        )


class PIIDetectionTask(Task):
    """Ask the user which tables/columns to hide or obfuscate."""

    name = "pii_detection"
    description = "Suggest PII and confirm which columns to mask."
    depends_on = ("fetch_schema",)

    async def run(self, ctx: TaskContext) -> TaskResult:
        # TODO(impl): heuristically flag candidates (regex on names like
        # email/phone/ssn/credit_card; type sniffing on sample values),
        # then present a multi-select with confidence-ordered suggestions.
        prompt = UserPrompt.new(
            phase=PHASE,
            task=self.name,
            question="Which columns should be obfuscated/hashed?",
            options=[
                PromptOption(label="users.email", value="users.email"),
                PromptOption(label="users.phone", value="users.phone"),
                PromptOption(label="payments.card_number", value="payments.card_number"),
                PromptOption(label="users.address", value="users.address"),
            ],
            multi_select=True,
            allow_text=True,
            text_label="Add other column paths (table.column, comma-separated)",
        )
        response = await ctx.ask(prompt)
        masked = list(response.selected)
        # TODO(impl): upsert mask=True on each chosen column's dbi_model.
        return TaskResult(
            task=self.name,
            status=TaskStatus.COMPLETED,
            output={"masked": masked, "extra": response.text},
        )


class ProfilingTask(Task):
    """Compute per-column statistical profile (cardinality, nulls, top-k)."""

    name = "profiling"
    description = "Per-column stats: cardinality, null rate, min/max/mean, top-K."
    depends_on = ("fetch_schema",)

    async def run(self, ctx: TaskContext) -> TaskResult:
        # TODO(impl): for each (table, column), run SELECTs to compute
        # cardinality, null_rate, min/max/mean, top-3 values, and
        # write a short natural-language summary to store.write_description.
        profiled: list[str] = []
        return TaskResult(
            task=self.name,
            status=TaskStatus.COMPLETED,
            output={"profiled_columns": len(profiled)},
        )


class RelationshipDiscoveryTask(Task):
    """Infer foreign keys that aren't declared in the schema."""

    name = "relationship_discovery"
    description = "Detect implicit FKs by name + value overlap, with confidence."
    depends_on = ("fetch_schema", "profiling")

    async def run(self, ctx: TaskContext) -> TaskResult:
        # TODO(impl): for each (col_a, col_b) candidate (suffix _id, name
        # overlap, type match), estimate value-set Jaccard overlap and
        # score confidence; ask the user to confirm low-confidence ones.
        candidates: list[dict] = []
        if candidates:
            await ctx.ask(
                UserPrompt.new(
                    phase=PHASE,
                    task=self.name,
                    question="Confirm inferred relationships:",
                    options=[
                        PromptOption(
                            label=f"{c['from']} → {c['to']} (conf {c['confidence']:.0%})",
                            value=f"{c['from']}->{c['to']}",
                        )
                        for c in candidates
                    ],
                    multi_select=True,
                )
            )
        return TaskResult(
            task=self.name,
            status=TaskStatus.COMPLETED,
            output={"candidates": len(candidates)},
        )


class FindEnumsTask(Task):
    """Resolve coded values (e.g. status=1) into human labels via the user."""

    name = "find_enums"
    description = "Ask the user to label opaque enum values."
    depends_on = ("profiling",)

    async def run(self, ctx: TaskContext) -> TaskResult:
        # TODO(impl): walk low-cardinality numeric/string columns and ask
        # the user to map each distinct value to a label. One prompt per
        # column keeps the UI tractable.
        # Example shape only; real impl would iterate over discovered enums:
        sample = UserPrompt.new(
            phase=PHASE,
            task=self.name,
            question="What does `orders.status = 1` mean?",
            options=[
                PromptOption(label="pending", value="pending"),
                PromptOption(label="paid", value="paid"),
                PromptOption(label="shipped", value="shipped"),
                PromptOption(label="cancelled", value="cancelled"),
            ],
            multi_select=False,
            allow_text=True,
            text_label="Or describe in your own words",
        )
        await ctx.ask(sample)
        return TaskResult(
            task=self.name,
            status=TaskStatus.COMPLETED,
            output={"enums_resolved": 0},
        )


class StalenessTask(Task):
    """Flag dead/legacy data so downstream queries avoid it."""

    name = "staleness"
    description = "Mark stale tables, fully-null columns, orphan tables."
    depends_on = ("profiling",)

    async def run(self, ctx: TaskContext) -> TaskResult:
        # TODO(impl): max(updated_at) per table → stale_at_threshold;
        # null_rate == 1.0 → deprecated; no inbound/outbound FK → orphan.
        # Persist findings into the dbi_model for each subject.
        return TaskResult(
            task=self.name,
            status=TaskStatus.COMPLETED,
            output={"flagged": 0},
        )


class SchemaCrawlPhase(Phase):
    name = PHASE
    description = "Phase 2: crawl schema, profile, discover relationships, flag dead data."

    def build_tasks(self) -> list[Task]:
        return [
            FetchSchemaTask(),
            PIIDetectionTask(),
            ProfilingTask(),
            RelationshipDiscoveryTask(),
            FindEnumsTask(),
            StalenessTask(),
        ]


__all__ = [
    "FetchSchemaTask",
    "FindEnumsTask",
    "PIIDetectionTask",
    "ProfilingTask",
    "RelationshipDiscoveryTask",
    "SchemaCrawlPhase",
    "StalenessTask",
]
