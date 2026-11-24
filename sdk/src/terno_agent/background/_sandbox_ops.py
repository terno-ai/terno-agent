"""Background process operations, run inside the sandbox.

The SDK cannot install anything in a sandbox, so :class:`SandboxBackgroundRunner`
sends this module's source through ``run_python`` and calls :func:`dispatch`.
It must therefore stay self-contained: standard library only, nothing from
``terno_agent``, and syntax the sandbox's Python can run (3.8+).

Each task has its own directory, made by :func:`prepare` and named by the
host::

    <root>/<name>/
        _sandbox_ops.py   copy of this module; runs the task detached
        spec.json         what to run, where, and until when
        pid               pid of the detached wrapper (leads its process group)
        output.log        combined stdout/stderr, trimmed to its tail if large
        timed_out         written when the wrapper stopped the work at its deadline
        stopped           written by stop(): before it kills the work, or so
                          work not started yet never starts
        exit_code         written when the work ends (always, even on errors)

The wrapper stays up for the task's lifetime: it copies output to the log,
enforces the deadline itself (so the limit holds even when nobody polls), and
records the exit code. Polling, reading and stopping only look at the
directory, so nothing stays connected to the sandbox while the work runs.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

MARKER = "__terno_bg__"

LOG_FILE = "output.log"
PID_FILE = "pid"
EXIT_FILE = "exit_code"
TIMED_OUT_FILE = "timed_out"
STOPPED_FILE = "stopped"
SPEC_FILE = "spec.json"
MODULE_FILE = "_sandbox_ops.py"

DEFAULT_ROOT_NAME = "terno-background"
# Finished task directories older than this are removed when a task starts.
RETENTION_S = 7 * 24 * 3600
# How long stopped work gets to exit before it is killed, and how long output
# still arriving after the work exited is read.
STOP_GRACE_S = 2.0
DRAIN_S = 2.0
_CHUNK = 64 * 1024
_IS_WINDOWS = os.name == "nt"
_WINDOWS_DETACHED = 0x00000200 | 0x00000008
_WINDOWS_NO_WINDOW = 0x08000000


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #


def prepare(*, root: str, name: str) -> dict:
    """Make the task's directory and return its handle. Nothing runs yet: the
    host saves the handle first, so it can always stop what it starts."""
    root = _prepare_root(root)
    _prune(root)
    task_dir = os.path.join(root, name)
    os.mkdir(task_dir)
    return {"handle": task_dir}


def start(
    *,
    handle: str,
    kind: str,
    payload: str,
    cwd: str,
    max_log_bytes: int,
    max_runtime_s: float,
) -> dict:
    """Start the work detached in its prepared directory, unless it was
    stopped already (a start that arrives late, after the host gave up)."""
    task_dir = handle
    if os.path.exists(os.path.join(task_dir, STOPPED_FILE)):
        return {"started": False}

    spec = {
        "argv": _command(task_dir, kind, payload),
        "cwd": cwd or os.getcwd(),
        "max_log_bytes": max_log_bytes,
        "deadline": time.time() + max_runtime_s,
    }
    _write_atomic(os.path.join(task_dir, SPEC_FILE), json.dumps(spec))
    _write_atomic(os.path.join(task_dir, MODULE_FILE), _own_source())

    options = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
               "stderr": subprocess.DEVNULL, "close_fds": True, "cwd": task_dir}
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
    return {"started": True}


def poll(*, handle: str) -> dict:
    if os.path.exists(os.path.join(handle, TIMED_OUT_FILE)):
        return {"running": False, "timed_out": True, "ended_at": _ended_at(handle, TIMED_OUT_FILE)}
    exit_code = _read_exit_code(handle)
    if exit_code is None:
        pid = _read_pid(handle)
        if pid is None:
            if os.path.exists(os.path.join(handle, STOPPED_FILE)):
                return {"running": False, "stopped": True, "ended_at": _ended_at(handle, STOPPED_FILE)}
            return {"running": False, "error": "task files are missing (was the sandbox restarted?)"}
        if _wrapper_alive(pid, handle):
            return {"running": True}
        exit_code = _read_exit_code(handle)
    if os.path.exists(os.path.join(handle, STOPPED_FILE)):
        reply = {"running": False, "stopped": True, "ended_at": _ended_at(handle, STOPPED_FILE)}
        if exit_code is not None:
            reply["exit_code"] = exit_code
        return reply
    if exit_code is None:
        return {"running": False, "error": "the process ended without an exit code"}
    return {"running": False, "exit_code": exit_code, "ended_at": _ended_at(handle, EXIT_FILE)}


def poll_many(*, handles: list) -> dict:
    """:func:`poll` for several tasks in one call. A task that cannot be
    read now is left out, to be checked again next time."""
    states = {}
    for handle in handles:
        try:
            states[handle] = poll(handle=handle)
        except OSError:
            continue
    return {"states": states}


def read_output(*, handle: str, max_chars: int) -> dict:
    try:
        log = open(os.path.join(handle, LOG_FILE), "rb")
    except OSError:
        return {"output": "", "truncated": False}
    with log:
        size = os.fstat(log.fileno()).st_size
        start_at = max(0, size - max_chars * 4)
        log.seek(start_at)
        text = log.read().decode("utf-8", "replace")
    tail = text[-max_chars:]
    truncated = _was_trimmed(handle) or start_at > 0 or len(text) > len(tail)
    return {"output": tail, "truncated": truncated}


def stop(*, handle: str, grace_s: float = STOP_GRACE_S) -> dict:
    """Stop the task's whole process group: politely, then by force. Work
    that has not started yet is marked so it never does."""
    task_dir = handle
    # Once it has finished, its pid may belong to another process.
    if _read_exit_code(task_dir) is not None:
        return {}
    pid = _read_pid(task_dir)
    if pid is None:
        if os.path.isdir(task_dir):
            _write_atomic(os.path.join(task_dir, STOPPED_FILE), "1")
        return {}
    if not _wrapper_alive(pid, task_dir):
        return {}

    _write_atomic(os.path.join(task_dir, STOPPED_FILE), "1")
    if _IS_WINDOWS:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        return {}
    _kill_group(pid, grace_s)
    return {}


OPERATIONS = {
    "prepare": prepare,
    "start": start,
    "poll": poll,
    "poll_many": poll_many,
    "read_output": read_output,
    "stop": stop,
}


def dispatch(operation: str, args_json: str) -> None:
    """Entry point for the host: run one operation and print its reply."""
    reply = OPERATIONS[operation](**json.loads(args_json))
    print(MARKER + json.dumps(reply), flush=True)


# --------------------------------------------------------------------------- #
# Detached wrapper
# --------------------------------------------------------------------------- #


def wrap(task_dir: str) -> None:
    """Run the task, copy its output to the log, enforce its deadline and
    record its exit code. The exit code is written whatever happens."""
    if not _IS_WINDOWS:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    exit_code = 1
    try:
        exit_code = _run_work(task_dir)
    finally:
        _write_atomic(os.path.join(task_dir, EXIT_FILE), str(exit_code))


def _run_work(task_dir: str) -> int:
    spec = _read_spec(task_dir)
    log = _Log(os.path.join(task_dir, LOG_FILE), int(spec["max_log_bytes"]))
    if os.path.exists(os.path.join(task_dir, STOPPED_FILE)):
        log.note("stopped before it started")
        return 1
    cwd = spec["cwd"]
    if not os.path.isdir(cwd):
        log.note(f"working directory {cwd} is gone; running in {task_dir}")
        cwd = task_dir
    options = {"cwd": cwd, "stdin": subprocess.DEVNULL,
               "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
    if _IS_WINDOWS:
        options["creationflags"] = _WINDOWS_NO_WINDOW
    else:
        options["preexec_fn"] = _default_sigterm
    try:
        child = subprocess.Popen(spec["argv"], **options)
    except OSError as exc:
        log.note(f"could not start: {exc}")
        return 127

    reader = threading.Thread(target=log.copy_from, args=(child.stdout,), daemon=True)
    reader.start()
    deadline = float(spec["deadline"])
    while True:
        try:
            exit_code = child.wait(timeout=1)
            break
        except subprocess.TimeoutExpired:
            if time.time() >= deadline:
                _write_atomic(os.path.join(task_dir, TIMED_OUT_FILE), "1")
                log.note("stopped: reached its time limit")
                _stop_work(child)
    reader.join(DRAIN_S)
    return exit_code


def _default_sigterm() -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _stop_work(child: subprocess.Popen) -> None:
    if _IS_WINDOWS:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(child.pid)], capture_output=True)
        return
    group = os.getpgrp()
    os.killpg(group, signal.SIGTERM)  # the wrapper ignores it
    try:
        child.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        os.killpg(group, signal.SIGKILL)


class _Log:
    """The task's output file, kept under its size limit by dropping the
    oldest part. Write errors (a full disk) never stop the work: output is
    then discarded and the problem noted once."""

    def __init__(self, path: str, max_bytes: int) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.failed = False

    def note(self, line: str) -> None:
        self._write((line + "\n").encode())

    def copy_from(self, stream) -> None:
        for chunk in iter(lambda: stream.read1(_CHUNK), b""):
            self._write(chunk)

    def _write(self, data: bytes) -> None:
        if self.failed:
            return
        try:
            with open(self.path, "ab") as f:
                f.write(data)
                size = f.tell()
            if size > self.max_bytes:
                _keep_tail(self.path, self.max_bytes // 2)
        except OSError as exc:
            self.failed = True
            try:
                with open(self.path, "ab") as f:
                    f.write(f"\n[output no longer saved: {exc}]\n".encode())
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _command(task_dir: str, kind: str, payload: str):
    if kind == "python":
        script = os.path.join(task_dir, "script.py")
        _write_atomic(script, payload)
        return [sys.executable, "-u", script]
    if kind == "shell":
        return "cmd /c " + payload if _IS_WINDOWS else ["/bin/sh", "-c", payload]
    raise ValueError(f"unsupported background task kind: {kind!r}")


def _prepare_root(root: str) -> str:
    """The directory holding task directories, private to this user."""
    if not root:
        owner = str(os.getuid()) if hasattr(os, "getuid") else os.environ.get("USERNAME", "user")
        root = os.path.join(tempfile.gettempdir(), f"{DEFAULT_ROOT_NAME}-{owner}")
    os.makedirs(root, mode=0o700, exist_ok=True)
    if hasattr(os, "getuid") and os.stat(root).st_uid != os.getuid():
        # Someone else could swap the code a task runs.
        raise PermissionError(f"{root} belongs to another user")
    return root


def _own_source() -> str:
    # Sent through run_python, this module has no file; the host puts its
    # source in the namespace it runs it in.
    source = globals().get("__terno_source__")
    if source is not None:
        return source
    with open(__file__, encoding="utf-8") as module:
        return module.read()


def _read_spec(task_dir: str) -> dict:
    with open(os.path.join(task_dir, SPEC_FILE), encoding="utf-8") as f:
        return json.load(f)


def _read_pid(task_dir: str) -> int | None:
    pid = _read_int(os.path.join(task_dir, PID_FILE))
    return pid if pid is not None and pid > 1 else None


def _ended_at(task_dir: str, name: str) -> float | None:
    """When the work ended: the file recording its end is written right then."""
    try:
        return os.path.getmtime(os.path.join(task_dir, name))
    except OSError:
        return None


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
    try:
        os.replace(path + ".tmp", path)
    except PermissionError:
        # Windows refuses while someone is reading the log; trim next time.
        os.remove(path + ".tmp")
        return
    # Remember that the start of the output is gone.
    open(path + ".trimmed", "w").close()


def _prune(root: str) -> None:
    """Remove task directories untouched for RETENTION_S whose work is over."""
    cutoff = time.time() - RETENTION_S
    for name in os.listdir(root):
        task_dir = os.path.join(root, name)
        try:
            if os.path.getmtime(task_dir) >= cutoff:
                continue
        except OSError:
            continue
        pid = _read_pid(task_dir)
        over = _read_exit_code(task_dir) is not None or pid is None
        if over or not _wrapper_alive(pid, task_dir):
            shutil.rmtree(task_dir, ignore_errors=True)


def _was_trimmed(task_dir: str) -> bool:
    return os.path.exists(os.path.join(task_dir, LOG_FILE + ".trimmed"))


def _wrapper_alive(pid: int, task_dir: str) -> bool:
    """The task's wrapper is still running (and the pid is still it)."""
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
    proc = _proc_info(pid)
    if proc is None:
        return True  # no /proc (macOS): the exit_code file tells us it ended
    state, _group, cmdline = proc
    # A zombie has exited; a different command line means the pid was reused.
    return state != "Z" and task_dir in cmdline


def _proc_info(pid: int):
    """(state, process group, command line) from Linux /proc, or None."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[-1].split()
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
        return fields[0], int(fields[2]), cmdline
    except (OSError, IndexError, ValueError):
        return None


def _kill_group(group: int, grace_s: float) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except OSError:
            return
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if not _group_alive(group):
                return
            time.sleep(0.1)


def _group_alive(group: int) -> bool:
    """Any process of the group is still running (zombies do not count)."""
    if os.path.isdir("/proc"):
        for entry in os.listdir("/proc"):
            if entry.isdigit():
                info = _proc_info(int(entry))
                if info is not None and info[1] == group and info[0] != "Z":
                    return True
        return False
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "wrap":
        sys.exit("usage: _sandbox_ops.py wrap <task dir>")
    wrap(sys.argv[2])
