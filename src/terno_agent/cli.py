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
from rich.text import Text

from terno_agent import __version__
from terno_agent.agents.terno import TernoAgent
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
        description="Terno — an interactive coding agent.",
    )
    p.add_argument("--version", action="version", version=f"terno-agent {__version__}")
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress agent activity; print only the final answer.",
    )
    sub = p.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="Run the agent on a single task and exit.")
    ask.add_argument("task", nargs="+", help="The task to run.")
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
    task = " ".join(args.task)
    console = Console()
    renderer = None if args.quiet else AgentRenderer(console)
    agent = TernoAgent.from_env(on_event=renderer)
    result = agent.ask(task)
    if renderer is not None:
        renderer.finalize()
    if args.quiet:
        print(result.answer)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    console = Console()
    renderer = None if args.quiet else AgentRenderer(console)
    agent = TernoAgent.from_env(on_event=renderer)
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


_AGENT_STYLE = "bold magenta"


class AgentRenderer:
    """Render `AgentEvent`s to a rich Console.

    Streams assistant text inline; surrounds tool calls and results with
    syntax-highlighted panels.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self._stream_open = False
        self._current_agent: str | None = None

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
            tag = Text(f"[{event.agent}]", style=_AGENT_STYLE)
            self.console.print(tag, end=" ", highlight=False)
            self._stream_open = True
            self._current_agent = event.agent
        self.console.print(event.text, end="", highlight=False, markup=False)

    def _close_stream(self) -> None:
        if self._stream_open:
            self.console.print()
            self._stream_open = False

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
                border_style=_AGENT_STYLE,
                padding=(0, 1),
            )
        )

    def _render_tool_result(self, event: ToolResultEvent) -> None:
        body = _format_result_body(event.result.content)
        style = "red" if event.result.is_error else "dim"
        marker = "✗" if event.result.is_error else "✓"
        title = f"[{event.agent}] {marker} result"
        self.console.print(
            Panel(body, title=title, title_align="left", border_style=style, padding=(0, 1))
        )


# --------------------------------------------------------------------------- #
# Tool-call / result formatting helpers
# --------------------------------------------------------------------------- #


def _format_call_body(name: str, args: dict) -> object:
    if name == "bash" and isinstance(args.get("command"), str):
        return Syntax(args["command"], "bash", theme="ansi_dark", word_wrap=True)
    if name == "write_file" and isinstance(args.get("content"), str):
        return Syntax(
            _truncate(args["content"], 2000),
            _lang_for_path(args.get("path", "")),
            theme="ansi_dark",
            word_wrap=True,
        )
    if name == "edit_file":
        parts = [
            f"path: {args.get('path', '')}",
            "--- old",
            _truncate(str(args.get("old_string", "")), 800),
            "+++ new",
            _truncate(str(args.get("new_string", "")), 800),
        ]
        return Text("\n".join(parts))
    if name == "spawn_agent":
        body = Text("prompt:\n", style="bold")
        body.append(_truncate(str(args.get("prompt", "")), 800))
        if args.get("task"):
            body.append("\n\ntask:\n", style="bold")
            body.append(_truncate(str(args.get("task", "")), 800))
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
    pretty = json.dumps(parsed, indent=2, default=str)
    return Syntax(_truncate(pretty, 4000), "json", theme="ansi_dark", word_wrap=True)


def _lang_for_path(path: str) -> str:
    lower = path.lower()
    for ext, lang in (
        (".py", "python"),
        (".js", "javascript"),
        (".ts", "typescript"),
        (".tsx", "tsx"),
        (".jsx", "jsx"),
        (".md", "markdown"),
        (".sh", "bash"),
        (".json", "json"),
        (".yaml", "yaml"),
        (".yml", "yaml"),
        (".toml", "toml"),
        (".sql", "sql"),
    ):
        if lower.endswith(ext):
            return lang
    return "text"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


if __name__ == "__main__":
    raise SystemExit(main())
