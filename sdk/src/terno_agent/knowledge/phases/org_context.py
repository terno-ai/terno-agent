"""Phase 1 — Organization Context.

Search the web for the connected company, ask a few targeted follow-ups,
and synthesize an org prompt that downstream phases (annotations,
question generation) condition on.

Output sink: `store.write_org_prompt(...)` and
`PhaseContext.org_prompt_ready` event.
"""

from __future__ import annotations

from terno_agent.knowledge.base import Phase, Task, TaskResult, TaskStatus
from terno_agent.knowledge.context import TaskContext
from terno_agent.knowledge.prompts import PromptOption, UserPrompt

PHASE = "org_context"


class WebSearchTask(Task):
    """Probe the public web for the company and draft an initial blurb."""

    name = "web_search"
    description = "Find the company on the web; draft an initial org blurb."

    async def run(self, ctx: TaskContext) -> TaskResult:
        # TODO(impl): hit a search tool (Tavily/Serper/Brave), pull About/LinkedIn,
        # then summarize via ctx.llm into a 1-2 paragraph draft.
        draft = ""
        ctx.phase.artifacts["org_web_draft"] = draft
        return TaskResult(
            task=self.name,
            status=TaskStatus.COMPLETED,
            output={"draft_chars": len(draft)},
        )


class FollowUpQuestionsTask(Task):
    """Ask the user to confirm/extend what we learned from the web."""

    name = "follow_up"
    description = "Ask the user a few clarifying questions."
    depends_on = ("web_search",)

    async def run(self, ctx: TaskContext) -> TaskResult:
        # TODO(impl): generate the question set from the web draft via LLM
        # so options reflect what's already known vs. still unknown.
        prompt = UserPrompt.new(
            phase=PHASE,
            task=self.name,
            question="Which of these best describe your company?",
            options=[
                PromptOption(label="B2B SaaS", value="b2b_saas"),
                PromptOption(label="B2C app", value="b2c_app"),
                PromptOption(label="Marketplace", value="marketplace"),
                PromptOption(label="EdTech", value="edtech"),
            ],
            multi_select=True,
            allow_text=True,
            text_label="Anything we missed (product lines, geos, segments)",
        )
        response = await ctx.ask(prompt)
        ctx.phase.artifacts["org_follow_up"] = {
            "selected": list(response.selected),
            "text": response.text,
        }
        return TaskResult(
            task=self.name,
            status=TaskStatus.COMPLETED,
            output={"selected": list(response.selected)},
        )


class BuildOrgPromptTask(Task):
    """Synthesize web draft + user answers into the canonical org prompt."""

    name = "build_org_prompt"
    description = "Compose and persist the org prompt."
    depends_on = ("web_search", "follow_up")

    async def run(self, ctx: TaskContext) -> TaskResult:
        # TODO(impl): prompt ctx.llm with the web draft + follow-up answers
        # to produce a structured org prompt (mission, products, jargon).
        org_prompt = ""
        ctx.store.write_org_prompt(org_prompt)
        ctx.phase.artifacts["org_prompt"] = org_prompt
        ctx.phase.org_prompt_ready.set()
        return TaskResult(
            task=self.name,
            status=TaskStatus.COMPLETED,
            output={"length": len(org_prompt)},
        )


class OrgContextPhase(Phase):
    name = PHASE
    description = "Phase 1: gather organization context; write the org prompt."

    def build_tasks(self) -> list[Task]:
        return [WebSearchTask(), FollowUpQuestionsTask(), BuildOrgPromptTask()]


__all__ = [
    "BuildOrgPromptTask",
    "FollowUpQuestionsTask",
    "OrgContextPhase",
    "WebSearchTask",
]
