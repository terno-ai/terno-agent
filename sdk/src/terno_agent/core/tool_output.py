"""Keep oversized tool output out of the LLM context.

Every tool result is appended to the agent's history and re-sent on every
later LLM call of the turn, so one large result is paid for many times over.
:class:`ToolOutputLimiter` sits in the agent loop (``BaseAgent._run_tool_call``)
and applies to every tool alike — built-ins, MCP tools, subagents.

When a result is over budget, the full text is saved to a file and the model
gets the head and tail plus that path. It can grep the file or read a slice of
it instead of carrying the whole text in its context.

Where the file goes is decided by :func:`make_output_saver`: inside the sandbox
when ``tool_output_dir`` is a sandbox path and the sandbox has the
``write_file`` op, otherwise on the host under the agent's ``workdir`` (which
the file tools read on the host too — see ``tools.files._use_sandbox``).
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# (limit, head, tail) in characters.
Budget = tuple[int, int, int]

# ``(filename, text) -> path the model can read`` or ``None`` when not saved.
OutputSaver = Callable[[str, str], "str | None"]

DEFAULT_OUTPUT_DIR = ".terno/tool_outputs"

# Their output must reach the model whole: a skill body is instructions, and
# ask_user returns the user's own answers.
_EXEMPT_TOOLS = frozenset({"activate_skill", "ask_user"})

# Their output is already a file the model chose a slice of; saving it again
# would only make a second copy. Over budget they are cut and pointed back at
# offset/limit instead.
_FILE_READING_TOOLS = frozenset({"read_file"})

# Package installs and training loops print pages of progress nobody reads.
_INSTALL_ARGS_RE = re.compile(r"\b(pip|uv|conda|apt(-get)?|npm|yarn)\b.*\binstall\b", re.I | re.S)
_NOISE_LINE_RE = re.compile(
    r"^(Collecting |Downloading |Installing collected packages|Successfully installed|"
    r"Requirement already satisfied|Building wheel for|Preparing (metadata|wheel)|"
    r"Epoch\s+\d+/\d+|\s*\d+/\d+\s*\[.*]\s*-?\s*loss:)",
    re.I,
)

_NOISE_BUDGET: Budget = (1000, 300, 300)
# Reading saved output back, or a call the host marked as needing more room
# (``large_patterns``). Mirrors terno-ai's legacy 15,000-char cap.
_LARGE_BUDGET: Budget = (15000, 6000, 6000)


@dataclass(slots=True)
class ToolOutputLimiter:
    save: OutputSaver
    max_chars: int = 5000
    head_chars: int = 1500
    tail_chars: int = 1500
    # Where saved output lives; a call whose arguments mention it is the model
    # reading saved output back, which is never saved again.
    output_dir: str = ""
    # Regexes over a call's JSON arguments that earn the large budget, e.g.
    # schema listings the next steps depend on.
    large_patterns: tuple[re.Pattern[str], ...] = ()

    @classmethod
    def build(
        cls,
        save: OutputSaver,
        *,
        max_chars: int = 5000,
        head_chars: int = 1500,
        tail_chars: int = 1500,
        output_dir: str = "",
        large_patterns: Iterable[str] = (),
    ) -> ToolOutputLimiter:
        return cls(
            save=save,
            max_chars=max_chars,
            head_chars=head_chars,
            tail_chars=tail_chars,
            output_dir=output_dir,
            large_patterns=tuple(re.compile(p) for p in large_patterns),
        )

    def apply(self, tool_name: str, arguments: dict[str, Any], text: str) -> str:
        if not text or tool_name in _EXEMPT_TOOLS:
            return text
        args_text = _args_text(arguments)
        rereading = tool_name in _FILE_READING_TOOLS or bool(
            self.output_dir and self.output_dir in args_text
        )
        limit, head, tail = self._budget(args_text, text, rereading)
        if len(text) <= limit:
            return text

        omitted = len(text) - head - tail
        preview = (
            f"{text[:head]}\n\n...[{omitted:,} characters omitted]...\n\n{text[-tail:]}"
        )
        shown = f"{head + tail:,} of {len(text):,} characters shown"
        if rereading:
            return preview + (
                f"\n\n[Output shortened: {shown}. Read a smaller range "
                "(read_file offset/limit) or grep for what you need.]"
            )

        name = f"{uuid.uuid4().hex[:10]}-{_safe(tool_name)}.txt"
        try:
            path = self.save(name, text)
        except Exception as exc:  # a failed save must never fail the tool call
            print(f"warning: could not save tool output: {exc}", file=sys.stderr)
            path = None
        if path is None:
            return preview + (
                f"\n\n[Output shortened: {shown}; the full text could not be saved. "
                "Re-run asking only for what you need (aggregates, filters, the first "
                "rows).]"
            )
        lines = text.count("\n") + 1
        return preview + (
            f"\n\n[Output shortened: {shown}. Full output ({lines:,} lines) saved to "
            f"{path} — grep it, read a slice with read_file offset/limit, or load it "
            "in code. Don't print it back whole.]"
        )

    def _budget(self, args_text: str, text: str, rereading: bool) -> Budget:
        if rereading or any(p.search(args_text) for p in self.large_patterns):
            return _LARGE_BUDGET
        if _INSTALL_ARGS_RE.search(args_text):
            return _NOISE_BUDGET
        lines = text.splitlines()
        noisy = sum(bool(_NOISE_LINE_RE.match(line)) for line in lines)
        if lines and (noisy >= 3 or noisy / len(lines) >= 0.30):
            return _NOISE_BUDGET
        return (self.max_chars, self.head_chars, self.tail_chars)


def make_output_saver(
    output_dir: str, *, workdir: Path, sandbox: Any = None
) -> tuple[OutputSaver, str]:
    """Pick where saved output goes. Returns ``(saver, resolved_dir)``.

    An absolute ``output_dir`` outside ``workdir`` is a sandbox path: written
    with the sandbox's ``write_file`` op when it has one. Anything else is
    written on the host under ``workdir``.
    """
    directory = Path(output_dir or DEFAULT_OUTPUT_DIR)
    in_sandbox = (
        sandbox is not None
        and callable(getattr(sandbox, "write_file", None))
        and directory.is_absolute()
        and not _is_under(directory, workdir)
    )
    if in_sandbox:
        base = directory.as_posix().rstrip("/")

        def save_in_sandbox(name: str, text: str) -> str | None:
            path = f"{base}/{name}"
            result = sandbox.write_file(path, text, overwrite=True)
            if isinstance(result, dict) and result.get("ok"):
                return path
            error = result.get("error") if isinstance(result, dict) else result
            print(f"warning: could not save tool output to {path}: {error}", file=sys.stderr)
            return None

        return save_in_sandbox, base

    host_dir = directory if directory.is_absolute() else workdir / directory

    def save_on_host(name: str, text: str) -> str | None:
        host_dir.mkdir(parents=True, exist_ok=True)
        path = host_dir / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    return save_on_host, str(host_dir)


def _args_text(arguments: dict[str, Any]) -> str:
    try:
        return json.dumps(arguments, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(arguments)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)[:40]


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "OutputSaver",
    "ToolOutputLimiter",
    "make_output_saver",
]
