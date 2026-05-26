"""argparse-based CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import difflib
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
from terno_agent.attachments import AttachmentInput
from terno_agent.config import Config
from terno_agent.core.events import (
    AgentEvent,
    IterationStart,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    TurnEnd,
)
from terno_agent.core.exceptions import TernoError, ToolError
from terno_agent.core.permissions import (
    PermissionDecision,
    PermissionMode,
    PermissionPolicy,
    PermissionRequest,
)
from terno_agent.knowledge.cli import run_knowledge_extraction
from terno_agent.memory.extractor import ExtractionResult
from terno_agent.tools.ask_user import Answer, Question

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
    p.add_argument(
        "--sandbox",
        metavar="NAME",
        default=None,
        help=(
            "Override the sandbox backend for this invocation. Built-ins: "
            "docker, local, none. Also accepts plugin names registered via "
            "the 'terno_agent.sandboxes' entry-point group, or "
            "'package.module:ClassName' for ad-hoc backends."
        ),
    )
    p.add_argument(
        "--sandbox-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Pass a key=value option through to the sandbox constructor. "
            "May be passed multiple times. Merges with TERNO_SANDBOX_OPTIONS."
        ),
    )
    sub = p.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="Run the agent on a single task and exit.")
    ask.add_argument(
        "--attach",
        action="append",
        default=[],
        metavar="PATH",
        help="Attach a file to this turn. May be passed multiple times.",
    )
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
        result, _exc = _run_turn_with_cancel(
            agent,
            task,
            console,
            attachments=args.attach,
        )
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
        "Use [bold]/deep_research[/] to launch knowledge extraction, "
        "[bold]/clear[/] to reset the conversation. "
        "Hit Ctrl-C once to stop a running turn; twice in 2s to quit.\n"
    )
    last_ctrlc = 0.0
    pending_attachments: list[str] = []
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
            if lowered in {"/clear", "/reset"}:
                agent.clear_history()
                console.print("[dim](conversation cleared)[/]\n")
                continue
            if lowered.startswith("/attach "):
                path = line[len("/attach ") :].strip()
                if path:
                    pending_attachments.append(path)
                    console.print(f"[dim]attached for next turn:[/] {path}")
                continue
            if lowered == "/attachments":
                if pending_attachments:
                    for path in pending_attachments:
                        console.print(f"[dim]- {path}[/]")
                else:
                    console.print("[dim](no pending attachments)[/]")
                continue
            if lowered in {"/usage", "/tokens"}:
                u = agent.usage
                console.print(
                    f"[dim]usage: {u.total_input_tokens} in / "
                    f"{u.total_output_tokens} out across {u.llm_calls} calls "
                    f"(last: {u.last_input_tokens} in / {u.last_output_tokens} out)[/]\n"
                )
                continue

            turn_attachments: list[AttachmentInput] = list(pending_attachments)
            pending_attachments = []
            result, exc = _run_turn_with_cancel(
                agent,
                line,
                console,
                attachments=turn_attachments,
            )
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
    *,
    attachments: list[AttachmentInput] | None = None,
) -> tuple[AgentRun | None, BaseException | None]:
    """Run one agent turn on a worker thread with Ctrl-C → cancel wired up.

    Returns ``(result, None)`` on success, ``(None, exception)`` on failure.
    Cancellation produces an `AgentRun(cancelled=True)`, not an exception.
    """
    holder: dict[str, AgentRun | BaseException | None] = {"result": None, "exc": None}

    def _worker() -> None:
        try:
            holder["result"] = agent.run(task, attachments=attachments)
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
    config = Config.from_env()
    overridden = False

    if getattr(args, "no_memory", False):
        config.memory_enabled = False
        overridden = True

    sandbox_arg = getattr(args, "sandbox", None)
    if sandbox_arg:
        config.sandbox = sandbox_arg if ":" in sandbox_arg else sandbox_arg.strip().lower()
        overridden = True

    cli_sandbox_opts = _parse_cli_kv(getattr(args, "sandbox_option", []) or [])
    if cli_sandbox_opts:
        merged = dict(config.sandbox_options)
        merged.update(cli_sandbox_opts)
        config.sandbox_options = merged
        overridden = True

    renderer = on_event if isinstance(on_event, AgentRenderer) else None
    console = renderer.console if renderer is not None else Console()
    ask_callback = CliPrompter(console, renderer=renderer)
    on_memory_event = _make_memory_notifier(console)
    permission_policy = _build_cli_permission_policy(console, renderer=renderer)

    if overridden:
        return TernoAgent.from_config(
            config,
            on_event=on_event,
            ask_callback=ask_callback,
            on_memory_event=on_memory_event,
            permission_policy=permission_policy,
        )
    return TernoAgent.from_env(
        on_event=on_event,
        ask_callback=ask_callback,
        on_memory_event=on_memory_event,
        permission_policy=permission_policy,
    )


def _make_memory_notifier(console: Console):
    """Print a single ``memory updated`` line when the extractor finishes."""

    def notify(result: ExtractionResult) -> None:
        if not result.changed:
            return
        console.print("[dim]memory updated[/]")

    return notify


def _parse_cli_kv(entries: list[str]) -> dict[str, str]:
    """Parse repeated ``--sandbox-option key=value`` flags into a dict."""
    out: dict[str, str] = {}
    for raw in entries:
        if "=" not in raw:
            raise TernoError(f"--sandbox-option entry {raw!r} must be 'key=value'")
        key, _, value = raw.partition("=")
        key = key.strip()
        if not key:
            raise TernoError(f"--sandbox-option entry {raw!r} has empty key")
        out[key] = value.strip()
    return out


def _shutdown_mcp(agent: TernoAgent) -> None:
    """Tear down the agent's per-session resources.

    Despite the legacy name this closes the sandbox container (unless
    `sandbox_persist=True`) as well as MCP servers. Kept under the old
    function name so the two existing call sites don't need to be
    touched.
    """
    sandbox = getattr(agent, "sandbox", None)
    if sandbox is not None:
        closer = getattr(sandbox, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # pragma: no cover - defensive
                pass
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
        # The CliPrompter prints questions itself; skip the JSON args panel
        # so the user sees a clean question-by-question flow.
        if call.name == "ask_user":
            count = len(call.arguments.get("questions") or [])
            self.console.print(
                Text(f"[{event.agent}] asking the user {count} question(s)…", style=_AGENT_STYLE)
            )
            return
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
        return _format_write_body(
            path=str(args.get("path", "")),
            content=args["content"],
            overwrite=bool(args.get("overwrite", False)),
        )
    if name == "edit_file":
        return _format_edit_diff(
            path=str(args.get("path", "")),
            old=str(args.get("old_string", "")),
            new=str(args.get("new_string", "")),
            replace_all=bool(args.get("replace_all", False)),
        )
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


_DIFF_MAX_LINES = 200


def _format_write_body(*, path: str, content: str, overwrite: bool) -> object:
    """For new files render the content; for overwrites show a real diff."""
    existing = _read_existing(path) if path else None
    if existing is not None:
        return _format_edit_diff(
            path=path,
            old=existing,
            new=content,
            replace_all=overwrite,
        )
    return Syntax(
        _truncate(content, 2000),
        _lang_for_path(path),
        theme="ansi_dark",
        word_wrap=True,
    )


def _read_existing(path: str) -> str | None:
    from pathlib import Path

    try:
        p = Path(path).expanduser()
        if not p.is_file():
            return None
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _format_edit_diff(*, path: str, old: str, new: str, replace_all: bool) -> Text:
    """Colour a unified diff of an ``edit_file`` call's old → new strings."""
    label = path or "(unspecified)"
    diff_lines = list(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
            lineterm="",
            n=3,
        )
    )

    body = Text()
    body.append(f"path: {label}", style="bold")
    if replace_all:
        body.append("  (replace_all)", style="dim")
    body.append("\n")

    if not diff_lines:
        body.append("(no textual difference)", style="dim italic")
        return body

    truncated = False
    if len(diff_lines) > _DIFF_MAX_LINES:
        diff_lines = diff_lines[:_DIFF_MAX_LINES]
        truncated = True

    for line in diff_lines:
        style = _diff_line_style(line)
        body.append(line, style=style)
        body.append("\n")

    if truncated:
        body.append(
            f"... [diff truncated at {_DIFF_MAX_LINES} lines]",
            style="dim italic",
        )

    return body


def _diff_line_style(line: str) -> str:
    if line.startswith(("+++", "---")):
        return "bold"
    if line.startswith("@@"):
        return "cyan"
    if line.startswith("+"):
        return "green"
    if line.startswith("-"):
        return "red"
    return ""


# --------------------------------------------------------------------------- #
# Human-in-the-loop prompter (ask_user tool callback)
# --------------------------------------------------------------------------- #


class CliPrompter:
    """Renders ``ask_user`` questions one at a time and reads stdin replies.

    Wired into ``TernoAgent`` via ``ask_callback``. Invoked from the agent's
    worker thread; stdin reads block that thread until the user answers,
    while the main thread keeps handling Ctrl-C → cancel signals.
    """

    def __init__(self, console: Console, *, renderer: AgentRenderer | None = None) -> None:
        self.console = console
        self.renderer = renderer

    def __call__(self, questions: list[Question]) -> list[Answer]:
        if not sys.stdin.isatty():
            raise ToolError(
                "ask_user is unavailable: stdin is not a TTY. "
                "Make a reasonable assumption and state it in your response."
            )
        # Close any streaming text panel so prompts don't collide with output.
        if self.renderer is not None:
            self.renderer.finalize()

        total = len(questions)
        answers: list[Answer] = []
        for idx, question in enumerate(questions, start=1):
            answers.append(self._ask_one(idx, total, question))
        return answers

    # ----- internals ---------------------------------------------------- #

    def _ask_one(self, idx: int, total: int, q: Question) -> Answer:
        header = f"Question {idx}/{total}"
        body = Text()
        body.append(q.question, style="bold")
        if q.multi_select:
            body.append("\n(multi-select — comma-separated, e.g. 1,3)", style="dim")
        self.console.print(
            Panel(body, title=header, title_align="left", border_style="cyan", padding=(0, 1))
        )

        other_idx = len(q.options) + 1
        for i, opt in enumerate(q.options, start=1):
            line = Text(f"  [{i}] ", style="cyan")
            line.append(opt.label, style="bold")
            if opt.description:
                line.append(f" — {opt.description}", style="dim")
            self.console.print(line)
        self.console.print(
            Text(f"  [{other_idx}] ", style="cyan").append("Other (custom text)", style="bold")
        )

        prompt = "select> "
        while True:
            try:
                raw = input(prompt).strip()
            except EOFError as exc:
                raise ToolError("ask_user cancelled: stdin closed.") from exc
            tokens = _parse_selection(raw, other_idx, q.multi_select, self.console)
            if tokens is None:
                continue

            labels: list[str] = []
            other_text: str | None = None
            need_reprompt = False
            for t in tokens:
                if t == other_idx:
                    try:
                        custom = input("  other> ").strip()
                    except EOFError as exc:
                        raise ToolError("ask_user cancelled: stdin closed.") from exc
                    if not custom:
                        self.console.print("[dim](other requires text — try again)[/]")
                        need_reprompt = True
                        break
                    other_text = custom
                else:
                    labels.append(q.options[t - 1].label)
            if need_reprompt:
                continue
            return Answer(question=q.question, selected=labels, other_text=other_text)


def _parse_selection(
    raw: str, max_idx: int, multi_select: bool, console: Console
) -> list[int] | None:
    if not raw:
        console.print("[dim](please enter a selection)[/]")
        return None
    try:
        tokens = [int(t.strip()) for t in raw.split(",") if t.strip()]
    except ValueError:
        console.print("[dim](enter numbers separated by commas)[/]")
        return None
    if not tokens:
        console.print("[dim](please enter a selection)[/]")
        return None
    if any(t < 1 or t > max_idx for t in tokens):
        console.print(f"[dim](numbers must be between 1 and {max_idx})[/]")
        return None
    if not multi_select and len(tokens) != 1:
        console.print("[dim](single-select — pick exactly one)[/]")
        return None
    # De-dupe while preserving order
    seen: set[int] = set()
    deduped: list[int] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


# --------------------------------------------------------------------------- #
# Permission prompter (front-end for PermissionPolicy)
# --------------------------------------------------------------------------- #


def _build_cli_permission_policy(
    console: Console,
    *,
    renderer: AgentRenderer | None = None,
) -> PermissionPolicy:
    """Build the default CLI permission policy.

    ASK mode with a Rich-driven prompter. Read-only / user-driven tools
    are auto-allowed via the policy's ``always_allow_tools`` default.
    """
    prompter = CliPermissionPrompter(console, renderer=renderer)
    return PermissionPolicy(
        mode=PermissionMode.ASK,
        on_request=prompter,
    )


class CliPermissionPrompter:
    """ASK-mode prompter for the CLI.

    Three options (modelled on Claude Code):
      1. Allow once.
      2. Allow this tool for the rest of the session.
      3. Deny + tell the agent what to do instead — the reason is
         surfaced to the LLM as a tool error.

    Returns a ``PermissionDecision`` rather than mutating a hook
    context, so the same callable shape works for any front-end
    (CLI, web UI, SDK consumer).
    """

    def __init__(self, console: Console, *, renderer: AgentRenderer | None = None) -> None:
        self.console = console
        self.renderer = renderer

    def __call__(self, request: PermissionRequest) -> PermissionDecision:
        if not sys.stdin.isatty():
            # Non-interactive — default to allow (CLI was piped).
            return PermissionDecision.allow_once()
        # Close any streaming text panel so the prompt isn't tangled with output.
        if self.renderer is not None:
            self.renderer.finalize()

        self._render_request(request)
        choice = self._read_choice()
        if choice == "1":
            return PermissionDecision.allow_once()
        if choice == "2":
            self.console.print(
                f"[dim]({request.tool_name!r} allowed for the rest of this session)[/]"
            )
            return PermissionDecision.allow_always(tool=request.tool_name)
        reason = self._read_reason()
        self.console.print("[dim](denied — feedback sent to the agent)[/]")
        return PermissionDecision.deny(reason)

    # ----- rendering ---------------------------------------------------- #

    def _render_request(self, request: PermissionRequest) -> None:
        name = request.tool_name
        try:
            preview = json.dumps(request.arguments, indent=2, default=str)
        except Exception:
            preview = str(request.arguments)
        body = Text()
        body.append("Tool: ", style="dim")
        body.append(name, style="bold")
        body.append("\n\n")
        body.append("Arguments:\n", style="dim")
        body.append(_truncate(preview, 1200))
        self.console.print(
            Panel(
                body,
                title="permission required",
                title_align="left",
                border_style="yellow",
                padding=(0, 1),
            )
        )
        self.console.print(
            Text("  [1] ", style="cyan").append("Allow once", style="bold")
        )
        self.console.print(
            Text("  [2] ", style="cyan").append(
                f"Allow {name!r} for the rest of this session", style="bold"
            )
        )
        self.console.print(
            Text("  [3] ", style="cyan").append(
                "Deny and tell the agent what to do instead", style="bold"
            )
        )

    def _read_choice(self) -> str:
        while True:
            try:
                raw = input("permission> ").strip()
            except EOFError:
                # No more input — default to deny so we don't run unattended tools.
                return "3"
            if raw in {"1", "2", "3"}:
                return raw
            self.console.print("[dim](enter 1, 2, or 3)[/]")

    def _read_reason(self) -> str:
        try:
            return input("  feedback> ").strip() or "Denied by user."
        except EOFError:
            return "Denied by user."


if __name__ == "__main__":
    raise SystemExit(main())
