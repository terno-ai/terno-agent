"""Background process operations, run inside the sandbox.

The SDK cannot install anything in a sandbox, so :class:`SandboxBackgroundRunner`
sends this module's source through ``run_python`` and calls :func:`dispatch`.
It must therefore stay self-contained: standard library only, nothing from
``terno_agent``, and syntax the sandbox's Python can run (3.8+).

Each task has its own directory::

    <root>/<task id>/
        _sandbox_ops.py   copy of this module; runs the task detached
        spec.json         what to run
        pid               pid of the detached wrapper (leads its process group)
        output.log        combined stdout/stderr, trimmed to its tail if large
        exit_code         written when the work ends

The wrapper process stays up for the task's lifetime and only reads output
and writes files, so polling, reading and stopping just look at the
directory: nothing stays connected to the sandbox while the work runs.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

MARKER = "__terno_bg__"

LOG_FILE = "output.log"
PID_FILE = "pid"
EXIT_FILE = "exit_code"
SPEC_FILE = "spec.json"
MODULE_FILE = "_sandbox_ops.py"

DEFAULT_ROOT_NAME = "terno-background"
# Finished task directories older than this are removed when a task starts.
RETENTION_S = 7 * 24 * 3600
_CHUNK = 64 * 1024
_IS_WINDOWS = os.name == "nt"
# Windows: CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS.
_WINDOWS_DETACHED = 0x00000200 | 0x00000008


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #


def start(
    *,
    root: str,
    task_id: str,
    kind: str,
    payload: str,
    cwd: str,
    max_log_bytes: int,
) -> dict:
    """Start the work detached and return its handle (the task directory)."""
    root = root or os.path.join(tempfile.gettempdir(), DEFAULT_ROOT_NAME)
    os.makedirs(root, exist_ok=True)
    _prune(root)
    # A fresh directory every time: task ids can repeat across sessions, and
    # an old exit_code must never be read as this task's.
    task_dir = tempfile.mkdtemp(prefix=f"{task_id}-", dir=root)

    spec = {
        "argv": _command(task_dir, kind, payload),
        "cwd": cwd or None,
        "max_log_bytes": max_log_bytes,
    }
    _write_atomic(os.path.join(task_dir, SPEC_FILE), json.dumps(spec))
    _write_atomic(os.path.join(task_dir, MODULE_FILE), _own_source())

    options = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
               "stderr": subprocess.DEVNULL, "close_fds": True}
    if _IS_WINDOWS:
        options["creationflags"] = _WINDOWS_DETACHED
    else:
        # Own session, so the wrapper leads a process group that stop() can
        # kill as a whole, and the sandbox call that started it can exit.
        options["start_new_session"] = True
    wrapper = subprocess.Popen(
        [sys.executable, os.path.join(task_dir, MODULE_FILE), "wrap", task_dir], **options
    )
    _write_atomic(os.path.join(task_dir, PID_FILE), str(wrapper.pid))
    return {"handle": task_dir}


def poll(*, handle: str) -> dict:
    exit_code = _read_exit_code(handle)
    if exit_code is not None:
        return {"running": False, "exit_code": exit_code}
    pid = _read_pid(handle)
    if pid is None:
        return {"running": False, "error": "task files are missing"}
    if _alive(pid):
        return {"running": True}
    # It may have finished between the two checks above.
    exit_code = _read_exit_code(handle)
    if exit_code is not None:
        return {"running": False, "exit_code": exit_code}
    return {"running": False, "error": "the process ended without an exit code"}


def read_output(*, handle: str, max_chars: int) -> dict:
    path = os.path.join(handle, LOG_FILE)
    try:
        size = os.path.getsize(path)
    except OSError:
        return {"output": "", "truncated": False}
    start_at = max(0, size - max_chars * 4)
    with open(path, "rb") as log:
        log.seek(start_at)
        text = log.read().decode("utf-8", "replace")
    tail = text[-max_chars:]
    truncated = _was_trimmed(handle) or start_at > 0 or len(text) > len(tail)
    return {"output": tail, "truncated": truncated}


def stop(*, handle: str, grace_s: float = 2.0) -> dict:
    """Stop the task's whole process group: politely, then by force."""
    pid = _read_pid(handle)
    # Once it has finished, its pid may belong to another process.
    if pid is None or _read_exit_code(handle) is not None or not _alive(pid):
        return {}
    if _IS_WINDOWS:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        return {}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except OSError:
            break
        if _wait_until_dead(pid, grace_s):
            break
    return {}


OPERATIONS = {"start": start, "poll": poll, "read_output": read_output, "stop": stop}


def dispatch(operation: str, args_json: str) -> None:
    """Entry point for the host: run one operation and print its reply."""
    reply = OPERATIONS[operation](**json.loads(args_json))
    print(MARKER + json.dumps(reply), flush=True)


# --------------------------------------------------------------------------- #
# Detached wrapper
# --------------------------------------------------------------------------- #


def wrap(task_dir: str) -> int:
    """Run the task, copy its output to the log (keeping only the tail when it
    grows past the limit) and record the exit code."""
    spec = _read_spec(task_dir)
    log_path = os.path.join(task_dir, LOG_FILE)
    max_bytes = int(spec["max_log_bytes"])
    try:
        child = subprocess.Popen(
            spec["argv"], cwd=spec["cwd"], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        with open(log_path, "ab") as log:
            log.write(f"could not start: {exc}\n".encode())
        exit_code = 127
    else:
        log = open(log_path, "ab")
        try:
            for chunk in iter(lambda: child.stdout.read1(_CHUNK), b""):
                log.write(chunk)
                log.flush()
                if log.tell() > max_bytes:
                    log.close()
                    _keep_tail(log_path, max_bytes // 2)
                    log = open(log_path, "ab")
        finally:
            log.close()
        exit_code = child.wait()
    _write_atomic(os.path.join(task_dir, EXIT_FILE), str(exit_code))
    return exit_code


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _command(task_dir: str, kind: str, payload: str) -> list:
    if kind == "python":
        script = os.path.join(task_dir, "script.py")
        _write_atomic(script, payload)
        return [sys.executable, "-u", script]
    if kind == "shell":
        return ["cmd", "/c", payload] if _IS_WINDOWS else ["/bin/sh", "-c", payload]
    raise ValueError(f"unsupported background task kind: {kind!r}")


def _own_source() -> str:
    # Sent through run_python, this module has no file; the host puts its
    # source in the namespace it runs it in.
    source = globals().get("__terno_source__")
    if source is not None:
        return source
    with open(__file__, encoding="utf-8") as module:
        return module.read()


def _read_spec(task_dir: str) -> dict:
    try:
        with open(os.path.join(task_dir, SPEC_FILE), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _read_pid(task_dir: str) -> int | None:
    return _read_int(os.path.join(task_dir, PID_FILE))


def _read_exit_code(task_dir: str) -> int | None:
    return _read_int(os.path.join(task_dir, EXIT_FILE))


def _read_int(path: str) -> int | None:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _write_atomic(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _keep_tail(path: str, keep_bytes: int) -> None:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - keep_bytes))
        tail = f.read()
    with open(path + ".tmp", "wb") as f:
        f.write(tail)
    os.replace(path + ".tmp", path)
    # Remember that the start of the output is gone.
    open(path + ".trimmed", "w").close()


def _prune(root: str) -> None:
    """Remove finished task directories older than RETENTION_S."""
    cutoff = time.time() - RETENTION_S
    for name in os.listdir(root):
        exit_file = os.path.join(root, name, EXIT_FILE)
        try:
            if os.path.getmtime(exit_file) < cutoff:
                shutil.rmtree(os.path.join(root, name), ignore_errors=True)
        except OSError:
            continue


def _was_trimmed(task_dir: str) -> bool:
    return os.path.exists(os.path.join(task_dir, LOG_FILE + ".trimmed"))


def _alive(pid: int) -> bool:
    if _IS_WINDOWS:
        listing = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
        ).stdout
        return f'"{pid}"' in listing
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return not _is_zombie(pid)


def _is_zombie(pid: int) -> bool:
    # A finished process whose parent never reaps it (e.g. when the sandbox's
    # PID 1 is not an init) still answers kill(0). Linux shows this in /proc;
    # elsewhere the exit_code file is the signal that it finished.
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().rsplit(")", 1)[-1].split()[0]
    except (OSError, IndexError):
        return False
    return state == "Z"


def _wait_until_dead(pid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return not _alive(pid)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "wrap":
        sys.exit("usage: _sandbox_ops.py wrap <task dir>")
    wrap(sys.argv[2])
