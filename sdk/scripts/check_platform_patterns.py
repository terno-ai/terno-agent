"""Flags 2 known Windows-breaking patterns already found in this codebase:
unconditional Unix-only imports, and hardcoded sh/python3 subprocess commands
with no platform-awareness nearby. Not a general-purpose platform linter -
scoped to exactly what has already broken this codebase once (see the
fix/cross-platform branch).

Known limits, accepted on purpose:
- only sees real Python syntax, not commands embedded in a string that gets
  exec'd elsewhere (e.g. _driver.py's DRIVER_SOURCE)
- only recognizes subprocess.Popen/run/call/check_call/check_output directly,
  not a custom sandbox.run_shell()-style wrapper
- "unconditional" for the command check means: no sys.platform/_IS_WINDOWS-style
  name anywhere in the same function - a coarse, function-scoped heuristic
  rather than real control-flow analysis, chosen specifically to avoid false
  positives on the early-return platform branches this codebase already uses.

Run: python scripts/check_platform_patterns.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

UNIX_ONLY_MODULES = {"fcntl"}
HARDCODED_COMMANDS = {"sh", "python3"}
SUBPROCESS_CALLEES = {"Popen", "run", "call", "check_call", "check_output"}
PLATFORM_CHECK_NAMES = {"platform", "sys", "_IS_WINDOWS", "_IS_POSIX"}

SRC = Path(__file__).resolve().parent.parent / "src"


def _imported_modules(node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name.split(".")[0] for alias in node.names}
    if node.module:
        return {node.module.split(".")[0]}
    return set()


def _is_subprocess_call(node: ast.Call) -> bool:
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    return name in SUBPROCESS_CALLEES


def _first_command_word(node: ast.Call) -> str | None:
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        parts = first.value.split()
        return parts[0] if parts else None
    if isinstance(first, (ast.List, ast.Tuple)) and first.elts:
        elt = first.elts[0]
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            return elt.value
    return None


def _mentions_platform_check(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in PLATFORM_CHECK_NAMES:
            return True
        if isinstance(child, ast.Attribute) and child.attr in {"platform", "system"}:
            return True
    return False


def check_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bad = _imported_modules(node) & UNIX_ONLY_MODULES
            if bad:
                problems.append(
                    f"{path}:{node.lineno}: unconditional import of {sorted(bad)} - "
                    "Unix-only, guard with `if sys.platform != \"win32\":`"
                )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _mentions_platform_check(node):
                continue
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and _is_subprocess_call(call):
                    word = _first_command_word(call)
                    if word in HARDCODED_COMMANDS:
                        problems.append(
                            f"{path}:{call.lineno}: hardcoded '{word}' command with no "
                            "platform check in this function - doesn't exist on Windows"
                        )
    return problems


def main() -> int:
    problems: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        problems.extend(check_file(path))

    if problems:
        print("Found known Windows-breaking patterns:\n")
        for p in problems:
            print(" ", p)
        print(f"\n{len(problems)} problem(s) found.")
        return 1

    print("No known Windows-breaking patterns found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
