"""Long-running Python driver that powers session-state sandboxes.

The host (DockerSandbox / LocalSandbox) launches one ``python -u -c <source>``
process per session and talks to it over stdin/stdout:

- The host writes one JSON-encoded request per line to stdin::
      {"code": "x = 1"}\\n
- The driver compiles + execs the snippet against a shared globals dict
  (so imports, variables, and function defs persist across calls).
- The driver captures the snippet's stdout/stderr and emits one response
  line to stdout, prefixed by `SENTINEL` so the host can locate it amid
  any free-form output the snippet itself produced::
      \\x1e__TERNO_SANDBOX_RESPONSE__\\x1e{"stdout":...,"stderr":...,"exit_code":0}\\n

Exceptions inside snippets are caught and surface as `exit_code=1` with a
traceback in `stderr`. `SystemExit(N)` produces `exit_code=N` but does NOT
terminate the driver — the session continues. The driver only exits if
stdin closes (host disconnects) or an unrecoverable error fires.
"""

from __future__ import annotations

# Marker the host scans for in the driver's stdout. Using a control char
# (RS / 0x1E) makes false-positive matches in snippet output unlikely.
SENTINEL = "\x1e__TERNO_SANDBOX_RESPONSE__\x1e"

# Sent verbatim into the container / subprocess as ``python -u -c``. Kept
# as a module-level string so tests can also exec it locally.
DRIVER_SOURCE = '''
import sys, io, json, traceback, contextlib

SENTINEL = "\\x1e__TERNO_SANDBOX_RESPONSE__\\x1e"

ns = {"__name__": "__terno_sandbox__", "__builtins__": __builtins__}


def _emit(response):
    sys.__stdout__.write(SENTINEL + json.dumps(response) + "\\n")
    sys.__stdout__.flush()


def _run_one(code):
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    exit_code = 0
    try:
        compiled = compile(code, "<snippet>", "exec")
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            exec(compiled, ns)
    except SystemExit as exc:
        value = exc.code
        if value is None:
            exit_code = 0
        elif isinstance(value, int):
            exit_code = value
        else:
            err_buf.write(str(value))
            exit_code = 1
    except BaseException:
        traceback.print_exc(file=err_buf)
        exit_code = 1
    return {
        "stdout": out_buf.getvalue(),
        "stderr": err_buf.getvalue(),
        "exit_code": exit_code,
    }


def _loop():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception as exc:
            _emit({"stdout": "", "stderr": f"driver protocol error: {exc}", "exit_code": 1})
            continue
        code = request.get("code", "")
        if not isinstance(code, str):
            _emit({"stdout": "", "stderr": "driver: 'code' must be a string", "exit_code": 1})
            continue
        _emit(_run_one(code))


try:
    _loop()
except KeyboardInterrupt:
    pass
'''


__all__ = ["DRIVER_SOURCE", "SENTINEL"]
