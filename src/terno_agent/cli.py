"""argparse-based CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from terno_agent import __version__
from terno_agent.agents.orchestrator import Orchestrator
from terno_agent.config import Config
from terno_agent.core.events import (
    AgentEvent,
    IterationStart,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    TurnEnd,
)
from terno_agent.core.exceptions import TernoError
from terno_agent.knowledge.cli import run_knowledge_extraction

_AGENT_COLORS = {
    "orchestrator": "bold magenta",
    "database": "bold cyan",
    "coder": "bold green",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except TernoError as exc:
        Console(stderr=True).print(f"[bold red]error:[/] {exc}")
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="terno",
        description="Multi-agent CLI that answers questions about your database.",
    )
    p.add_argument("--version", action="version", version=f"terno-agent {__version__}")
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress agent activity; print only the final answer.",
    )
    sub = p.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="Ask a single question and exit.")
    ask.add_argument("question", nargs="+", help="The question to ask.")
    ask.set_defaults(func=_cmd_ask)

    chat = sub.add_parser("chat", help="Start an interactive REPL.")
    chat.set_defaults(func=_cmd_chat)

    cfg = sub.add_parser("config", help="Show effective configuration.")
    cfg.set_defaults(func=_cmd_config)

    research = sub.add_parser(
        "deep_research",
        aliases=["knowledge"],
        help="Run the deep-research / knowledge-extraction pipeline "
        "(org context, schema crawl, annotation, validation).",
    )
    research.set_defaults(func=_cmd_deep_research)

    return p


def _cmd_ask(args: argparse.Namespace) -> int:
    question = " ".join(args.question)
    console = Console()
    renderer = None if args.quiet else AgentRenderer(console)
    agent = Orchestrator.from_env(on_event=renderer)
    result = agent.ask(question)
    if renderer is not None:
        renderer.finalize()
    if args.quiet:
        print(result.answer)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    console = Console()
    renderer = None if args.quiet else AgentRenderer(console)
    agent = Orchestrator.from_env(on_event=renderer)
    console.print(
        "[bold]terno-agent REPL[/] — type 'exit' or Ctrl-D to quit. "
        "Use [bold]/deep_research[/] to launch knowledge extraction.\n"
    )
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            print()
            return 0
        if not line:
            continue
        lowered = line.lower()
        if lowered in {"exit", "quit", ":q"}:
            return 0
        if lowered in {"/deep_research", "/research", "/knowledge"}:
            _run_deep_research(console)
            console.print()
            continue
        try:
            result = agent.ask(line)
        except TernoError as exc:
            console.print(f"[bold red]error:[/] {exc}")
            continue
        if renderer is not None:
            renderer.finalize()
            renderer.reset()
        if args.quiet:
            print(f"terno> {result.answer}\n")
        else:
            console.print()
    return 0


def _cmd_config(_args: argparse.Namespace) -> int:
    print(Config.from_env().display())
    return 0


def _cmd_deep_research(_args: argparse.Namespace) -> int:
    console = Console()
    console.print("[bold]Deep research[/] — four phases, prompts inline.\n")
    _run_deep_research(console)
    return 0


def _run_deep_research(console: Console) -> None:
    try:
        asyncio.run(run_knowledge_extraction(console=console))
    except TernoError as exc:
        console.print(f"[bold red]deep research failed:[/] {exc}")


# --------------------------------------------------------------------------- #
# Event renderer
# --------------------------------------------------------------------------- #


class AgentRenderer:
    """Render `AgentEvent`s to a rich Console.

    Streams assistant text inline; surrounds tool calls and results with
    syntax-highlighted panels. Designed to be called from any thread.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self._stream_open = False
        self._current_agent: str | None = None

    # callable as the event hook
    def __call__(self, event: AgentEvent) -> None:
        if isinstance(event, IterationStart):
            return
        if isinstance(event, TextDelta):
            self._handle_text(event)
        elif isinstance(event, ToolCallEvent):
            self._close_stream()
            self._render_tool_call(event)
        elif isinstance(event, ToolResultEvent):
            self._close_stream()
            self._render_tool_result(event)
        elif isinstance(event, TurnEnd):
            self._close_stream()

    def finalize(self) -> None:
        self._close_stream()

    def reset(self) -> None:
        self._current_agent = None

    # ----- streaming text ------------------------------------------------- #

    def _handle_text(self, event: TextDelta) -> None:
        if not event.text:
            return
        if not self._stream_open or self._current_agent != event.agent:
            self._close_stream()
            tag = self._agent_tag(event.agent)
            self.console.print(tag, end=" ", highlight=False)
            self._stream_open = True
            self._current_agent = event.agent
        self.console.print(event.text, end="", highlight=False, markup=False)

    def _close_stream(self) -> None:
        if self._stream_open:
            self.console.print()  # newline to end the streaming line
            self._stream_open = False

    def _agent_tag(self, agent: str) -> Text:
        style = _AGENT_COLORS.get(agent, "bold white")
        return Text(f"[{agent}]", style=style)

    # ----- tool call/result panels --------------------------------------- #

    def _render_tool_call(self, event: ToolCallEvent) -> None:
        call = event.call
        body = _format_call_body(call.name, call.arguments)
        title = f"[{event.agent}] → {call.name}"
        self.console.print(
            Panel(
                body,
                title=title,
                title_align="left",
                border_style=_AGENT_COLORS.get(event.agent, "white"),
                padding=(0, 1),
            )
        )

    def _render_tool_result(self, event: ToolResultEvent) -> None:
        result = event.result
        body = _format_result_body(result.content)
        style = "red" if result.is_error else "dim"
        marker = "✗" if result.is_error else "✓"
        title = f"[{event.agent}] {marker} result"
        self.console.print(
            Panel(body, title=title, title_align="left", border_style=style, padding=(0, 1))
        )


# --------------------------------------------------------------------------- #
# Tool-call / result formatting helpers
# --------------------------------------------------------------------------- #


def _format_call_body(name: str, args: dict) -> object:
    """Pick the best rich renderable for the tool's payload."""
    if name == "sql_query" and isinstance(args.get("sql"), str):
        return Syntax(args["sql"].strip(), "sql", theme="ansi_dark", word_wrap=True)
    if name == "run_python" and isinstance(args.get("code"), str):
        return Syntax(args["code"], "python", theme="ansi_dark", word_wrap=True)
    if name in {"ask_database_agent", "ask_coder_agent"}:
        task = args.get("task", "")
        extra = args.get("input_data")
        body = Text(task, no_wrap=False)
        if extra:
            body.append("\n\ninput_data:\n", style="dim")
            body.append(_truncate(extra, 800))
        return body
    pretty = json.dumps(args, indent=2, default=str)
    return Syntax(pretty, "json", theme="ansi_dark", word_wrap=True)


def _format_result_body(content: str) -> object:
    text = content.strip()
    if not text:
        return Text("(empty)", style="dim italic")
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return Text(_truncate(text, 4000))
    if isinstance(parsed, dict) and {"columns", "rows"} <= parsed.keys():
        return _format_query_result(parsed)
    pretty = json.dumps(parsed, indent=2, default=str)
    return Syntax(_truncate(pretty, 4000), "json", theme="ansi_dark", word_wrap=True)


def _format_query_result(payload: dict) -> object:
    columns: list[str] = payload.get("columns") or []
    rows: list[list] = payload.get("rows") or []
    table = Table(show_header=True, header_style="bold cyan", expand=False)
    for col in columns:
        table.add_column(str(col))
    preview = rows[:20]
    for row in preview:
        table.add_row(*[_cell(v) for v in row])
    footer = f"{payload.get('row_count', len(rows))} rows"
    if payload.get("truncated"):
        footer += " (truncated)"
    if len(rows) > len(preview):
        footer += f" — showing first {len(preview)}"
    table.caption = footer
    return table


def _cell(value: object) -> str:
    if value is None:
        return "—"
    text = str(value)
    if len(text) > 80:
        text = text[:77] + "..."
    return text


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


if __name__ == "__main__":
    raise SystemExit(main())
