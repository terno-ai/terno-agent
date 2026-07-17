"""Terminal driver for the knowledge-extraction pipeline.

Renders `UserPrompt`s from the shared `PromptChannel` to a rich
Console and reads back numeric selections + optional text via
`asyncio.to_thread(input, ...)`. The runner and the drainer share
an event loop so phases keep making progress while we block on stdin.
"""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from terno_agent.config import Config
from terno_agent.core.exceptions import ConfigError
from terno_agent.knowledge.agent import KnowledgeExtractionAgent
from terno_agent.knowledge.prompts import PromptChannel, UserPrompt, UserResponse
from terno_agent.knowledge.runner import KnowledgeReport
from terno_agent.knowledge.store import InMemoryStore, KnowledgeStore
from terno_agent.llm.factory import create_llm_client

_PHASE_COLORS = {
    "org_context": "bold blue",
    "schema_crawl": "bold cyan",
    "annotation": "bold green",
    "validation": "bold yellow",
}


def _render_prompt(console: Console, prompt: UserPrompt) -> None:
    color = _PHASE_COLORS.get(prompt.phase, "bold white")
    body = Text()
    body.append(prompt.question + "\n\n", style="bold")
    for i, opt in enumerate(prompt.options, 1):
        body.append(f"  {i}. {opt.label}\n")
        if opt.description:
            body.append(f"     {opt.description}\n", style="dim")
    hints = [
        "Multi-select: comma-separated numbers (e.g. 1,3). Blank = none."
        if prompt.multi_select
        else "Single-select: one number. Blank = none."
    ]
    if prompt.allow_text:
        hints.append(f"Free text: {prompt.text_label or 'optional notes'}.")
    body.append("\n" + "\n".join(hints), style="dim italic")
    console.print(
        Panel(
            body,
            title=f"[{prompt.phase}.{prompt.task}]",
            title_align="left",
            border_style=color,
        )
    )


def _parse_selection(prompt: UserPrompt, line: str) -> tuple[str, ...]:
    if not line.strip():
        return ()
    picks: list[str] = []
    for tok in line.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            idx = int(tok) - 1
        except ValueError:
            continue
        if 0 <= idx < len(prompt.options):
            picks.append(prompt.options[idx].value)
    return tuple(picks)


async def _drain_prompts(
    channel: PromptChannel, console: Console, stop: asyncio.Event
) -> None:
    while not stop.is_set():
        try:
            prompt = await asyncio.wait_for(channel.next_prompt(), timeout=0.25)
        except TimeoutError:
            continue
        _render_prompt(console, prompt)
        line = await asyncio.to_thread(input, "  selection> ")
        text: str | None = None
        if prompt.allow_text:
            entered = await asyncio.to_thread(input, "  text> ")
            text = entered.strip() or None
        channel.submit(
            UserResponse(
                prompt_id=prompt.id,
                selected=_parse_selection(prompt, line),
                text=text,
            )
        )


def _print_report(console: Console, report: KnowledgeReport) -> None:
    console.print()
    for p in report.phases:
        color = _PHASE_COLORS.get(p.phase, "white")
        console.print(f"[{color}]{p.phase}[/]  ok={p.ok}")
        for t in p.tasks:
            mark = "✓" if t.status.value == "completed" else "·"
            err = f"  [red]{t.error}[/]" if t.error else ""
            console.print(f"  {mark} {t.task} [dim]({t.status.value})[/]{err}")
    console.print(f"\n[bold]overall:[/] ok={report.ok}")


async def run_knowledge_extraction(
    *,
    config: Config | None = None,
    store: KnowledgeStore | None = None,
    console: Console | None = None,
    require_database: bool = True,
) -> KnowledgeReport:
    cfg = config or Config.from_env()
    console = console or Console()
    if require_database and not cfg.database_url:
        raise ConfigError("TERNO_DATABASE_URL is required for knowledge extraction.")
    if cfg.database_url:
        raise ConfigError(
            "Database-backed knowledge extraction is no longer available "
            "(terno_agent.db was removed)."
        )
    db = None
    llm = create_llm_client(
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
    )
    agent = KnowledgeExtractionAgent(
        db=db, llm=llm, store=store or InMemoryStore()
    )
    stop = asyncio.Event()
    drainer = asyncio.create_task(_drain_prompts(agent.channel, console, stop))
    try:
        report = await agent.run()
    finally:
        stop.set()
        await drainer
    _print_report(console, report)
    return report


__all__ = ["run_knowledge_extraction"]
