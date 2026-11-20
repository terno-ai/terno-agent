"""``_send_all`` writes a whole request to a non-blocking socket."""

from __future__ import annotations

import socket
import threading

import pytest

from terno_agent.sandbox.docker import _send_all


def test_sends_more_than_the_socket_buffer():
    writer, reader = socket.socketpair()
    writer.setblocking(False)
    payload = b"x" * 2_000_000 + b"\n"
    received = bytearray()

    def drain():
        while len(received) < len(payload):
            chunk = reader.recv(65536)
            if not chunk:
                break
            received.extend(chunk)

    thread = threading.Thread(target=drain)
    thread.start()
    try:
        _send_all(writer, payload, timeout_s=10)
        thread.join(timeout=10)
        assert bytes(received) == payload
    finally:
        writer.close()
        reader.close()


def test_times_out_when_the_socket_never_becomes_writable(monkeypatch):
    # How much an OS buffers on a socket varies (Windows takes far more), so
    # simulate a full buffer instead of trying to fill a real one.
    import terno_agent.sandbox.docker as docker_module

    monkeypatch.setattr(docker_module.select, "select", lambda r, w, x, t: ([], [], []))
    writer, reader = socket.socketpair()
    writer.setblocking(False)
    try:
        with pytest.raises(TimeoutError):
            _send_all(writer, b"x" * 100, timeout_s=0.3)
    finally:
        writer.close()
        reader.close()
