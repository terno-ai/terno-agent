"""argparse-based CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import threading
import time
from collections.abc import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from terno_agent import __version__
from terno_agent.agents.base import AgentRun
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

_DOUBLE_CTRLC_WINDOW_S = 2.0


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
    p.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable persistent memory (no recall, no extraction) for this session.",
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
    agent = _build_agent(args, on_event=renderer)
    try:
        result, _exc = _run_turn_with_cancel(agent, task, console)
        if _exc is not None and not isinstance(_exc, TernoError):
            raise _exc
        if _exc is not None:
            console.print(f"[bold red]error:[/] {_exc}")
            return 2
        if renderer is not None:
            renderer.finalize()
        if args.quiet and result is not None:
            print(result.answer)
        return 0
    finally:
        _shutdown_mcp(agent)


def _cmd_chat(args: argparse.Namespace) -> int:
    console = Console()
    renderer = None if args.quiet else AgentRenderer(console)
    agent = _build_agent(args, on_event=renderer)
    console.print(
        "[bold]terno-agent REPL[/] — type 'exit' or Ctrl-D to quit. "
        "Use [bold]/deep_research[/] to launch knowledge extraction. "
        "Hit Ctrl-C once to stop a running turn; twice in 2s to quit.\n"
    )
    last_ctrlc = 0.0
    try:
        while True:
            try:
                line = input("you> ").strip()
            except EOFError:
                print()
                return 0
            except KeyboardInterrupt:
                now = time.monotonic()
                if now - last_ctrlc < _DOUBLE_CTRLC_WINDOW_S:
                    console.print()
                    return 0
                last_ctrlc = now
                console.print("\n[dim](press Ctrl-C again to quit)[/]")
                continue
            if not line:
                continue
            lowered = line.lower()
            if lowered in {"exit", "quit", ":q"}:
                return 0
            if lowered in {"/deep_research", "/research", "/knowledge"}:
                _run_deep_research(console)
                console.print()
                continue

            result, exc = _run_turn_with_cancel(agent, line, console)
            if exc is not None:
                if isinstance(exc, TernoError):
                    console.print(f"[bold red]error:[/] {exc}")
                else:
                    console.print(f"[bold red]error:[/] {exc!r}")
                continue
            if renderer is not None:
                renderer.finalize()
                renderer.reset()
            if result is not None and result.cancelled:
                console.print("[dim](turn cancelled)[/]")
            if args.quiet and result is not None and not result.cancelled:
                print(f"terno> {result.answer}\n")
            elif not args.quiet:
                console.print()
    finally:
        _shutdown_mcp(agent)
    return 0


def _run_turn_with_cancel(
    agent: TernoAgent,
    task: str,
    console: Console,
) -> tuple[AgentRun | None, BaseException | None]:
    """Run one agent turn on a worker thread with Ctrl-C → cancel wired up.

    Returns ``(result, None)`` on success, ``(None, exception)`` on failure.
    Cancellation produces an `AgentRun(cancelled=True)`, not an exception.
    """
    holder: dict[str, AgentRun | BaseException | None] = {"result": None, "exc": None}

    def _worker() -> None:
        try:
            holder["result"] = agent.run(task)
        except BaseException as exc:
            holder["exc"] = exc

    thread = threading.Thread(target=_worker, name="terno-turn", daemon=True)
    cancel_pressed = {"count": 0, "last": 0.0}

    def _on_sigint(signum, frame):  # noqa: ARG001 - signal handler signature
        cancel_pressed["count"] += 1
        cancel_pressed["last"] = time.monotonic()
        try:
            agent.cancel()
        except Exception:
            pass
        console.print("\n[dim](stopping… press again to force-kill the chat)[/]")

    previous = signal.signal(signal.SIGINT, _on_sigint)
    try:
        thread.start()
        # Watchdog: if the user presses Ctrl-C twice while the worker is
        # ignoring cooperative cancellation, restore the default handler
        # so a third press kills the whole process.
        force_threshold = 2
        while thread.is_alive():
            thread.join(timeout=0.2)
            if cancel_pressed["count"] >= force_threshold:
                # Hand control back to the OS — user really wants out.
                signal.signal(signal.SIGINT, signal.default_int_handler)
                # Keep waiting; next ^C will raise KeyboardInterrupt.
                force_threshold = 10**9  # only restore once
    finally:
        signal.signal(signal.SIGINT, previous)
        agent.reset_cancel()

    return (holder["result"], holder["exc"])


def _build_agent(args: argparse.Namespace, *, on_event=None) -> TernoAgent:
    """Build a TernoAgent from env, honoring CLI flags like ``--no-memory``."""
    if not getattr(args, "no_memory", False):
        return TernoAgent.from_env(on_event=on_event)
    config = Config.from_env()
    config.memory_enabled = False
    return TernoAgent.from_config(config, on_event=on_event)


def _shutdown_mcp(agent: TernoAgent) -> None:
    manager = getattr(agent, "mcp_manager", None)
    if manager is not None:
        try:
            manager.shutdown()
        except Exception:  # pragma: no cover - defensive
            pass


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
