from __future__ import annotations

import gzip
import io
from typing import Any

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.tools import web


class _FakeResponse:
    def __init__(
        self,
        body: bytes,
        content_type: str = "text/html; charset=utf-8",
        content_encoding: str = "",
    ) -> None:
        self._body = body
        self.headers = {"Content-Type": content_type}
        if content_encoding:
            self.headers["Content-Encoding"] = content_encoding
        self._buf = io.BytesIO(body)

    def read(self, size: int | None = None) -> bytes:
        if size is None:
            return self._buf.read()
        return self._buf.read(size)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None


@pytest.fixture
def stub_urlopen(monkeypatch):
    captured: dict[str, Any] = {}

    def make(
        body: bytes,
        content_type: str = "text/html; charset=utf-8",
        content_encoding: str = "",
    ):
        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.headers)
            return _FakeResponse(body, content_type, content_encoding)

        monkeypatch.setattr(web, "urlopen", fake_urlopen)
        return captured

    return make


def test_web_fetch_strips_html(stub_urlopen):
    stub_urlopen(
        b"<html><head><title>x</title><style>.a{}</style></head>"
        b"<body><h1>Hello</h1><p>World <b>!</b></p>"
        b"<script>alert('no')</script></body></html>",
    )
    out = web.WebFetchTool().run(url="https://example.com")
    assert "Hello" in out
    assert "World" in out
    assert "!" in out
    assert "alert" not in out  # script body dropped
    assert ".a{}" not in out  # style body dropped


def test_web_fetch_rejects_non_http():
    with pytest.raises(ToolError, match="absolute"):
        web.WebFetchTool().run(url="file:///etc/passwd")


def test_web_fetch_truncates_long_text(stub_urlopen):
    long_body = ("a" * 50_000).encode()
    stub_urlopen(long_body, content_type="text/plain")
    out = web.WebFetchTool().run(url="https://example.com", max_chars=100)
    assert "truncated" in out
    # Strip the truncation suffix, body should now be exactly 100 chars
    head = out.splitlines()[0]
    assert len(head) == 100


def test_web_search_parses_ddg_results(stub_urlopen):
    html_body = (
        b'<html><body>'
        b'<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">'
        b'First <b>Result</b></a>'
        b'<a class="result__snippet">Snippet one.</a>'
        b'<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fb">'
        b'Second Result</a>'
        b'<a class="result__snippet">Snippet two.</a>'
        b'</body></html>'
    )
    captured = stub_urlopen(html_body)
    out = web.WebSearchTool().run(query="example query", limit=5)
    assert "1. First Result" in out
    assert "https://example.com/a" in out
    assert "Snippet one." in out
    assert "2. Second Result" in out
    assert "https://example.com/b" in out
    assert "q=example+query" in captured["url"]


def test_web_search_no_results(stub_urlopen):
    stub_urlopen(b"<html><body>nothing here</body></html>")
    out = web.WebSearchTool().run(query="zqzqzq")
    assert "no results" in out


def test_web_search_requires_query(stub_urlopen):
    stub_urlopen(b"")
    with pytest.raises(ToolError):
        web.WebSearchTool().run(query="")


def test_web_fetch_sends_browser_headers(stub_urlopen):
    captured = stub_urlopen(b"<html><body>ok</body></html>")
    web.WebFetchTool().run(url="https://example.com")
    sent = captured["headers"]
    # urllib lowercases header keys in Request.headers
    expected = {
        "User-agent",
        "Accept",
        "Accept-language",
        "Accept-encoding",
        "Upgrade-insecure-requests",
        "Sec-fetch-dest",
        "Sec-fetch-mode",
        "Sec-fetch-site",
        "Sec-fetch-user",
        "Sec-ch-ua",
        "Sec-ch-ua-mobile",
        "Sec-ch-ua-platform",
    }
    missing = expected - set(sent)
    assert not missing, f"missing browser headers: {missing}"
    assert "Chrome" in sent["User-agent"]
    assert sent["Accept-encoding"] == "gzip, deflate"


def test_web_fetch_decompresses_gzip(stub_urlopen):
    payload = b"<html><body><p>compressed body</p></body></html>"
    stub_urlopen(gzip.compress(payload), content_encoding="gzip")
    out = web.WebFetchTool().run(url="https://example.com")
    assert "compressed body" in out


def test_web_fetch_unknown_encoding_falls_through(stub_urlopen):
    # If a server ignores Accept-Encoding and sends br anyway, we don't
    # crash — we just hand the raw bytes through UTF-8-with-replacement.
    stub_urlopen(b"plain text body", content_encoding="br", content_type="text/plain")
    out = web.WebFetchTool().run(url="https://example.com")
    assert "plain text body" in out
