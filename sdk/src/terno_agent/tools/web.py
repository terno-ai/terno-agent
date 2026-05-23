"""Web tools: web_fetch (read a URL) and web_search (DuckDuckGo HTML).

Both rely only on the standard library so they work without optional
extras. HTML is stripped to visible text via ``html.parser``; search
results come from DuckDuckGo's HTML endpoint (no API key required).
"""

from __future__ import annotations

import gzip
import html.parser
import re
import zlib
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)

# Browser-like headers reduce false-positive bot blocks on
# Cloudflare / Akamai / DataDome stacks. We only advertise compression
# schemes we can actually decode (no brotli without an extra dep).
_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": _USER_AGENT,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Sec-CH-UA": (
        '"Chromium";v="134", "Google Chrome";v="134", "Not.A/Brand";v="24"'
    ),
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"Windows"',
}

_MAX_RESPONSE_BYTES = 2_000_000
_REQUEST_TIMEOUT_S = 30
_FETCH_DEFAULT_MAX_CHARS = 20_000
_SEARCH_DEFAULT_LIMIT = 5
_SEARCH_MAX_LIMIT = 20


def _decompress(raw: bytes, encoding: str) -> bytes:
    enc = encoding.lower().strip()
    if not enc or enc == "identity":
        return raw
    if enc == "gzip":
        return gzip.decompress(raw)
    if enc == "deflate":
        # Servers send either zlib-wrapped or raw deflate; try both.
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return zlib.decompress(raw, -zlib.MAX_WBITS)
    # Unknown encoding (e.g. br) — leave bytes untouched and let the
    # caller's UTF-8 replacement decode produce something readable
    # rather than failing the whole fetch.
    return raw


def _fetch(url: str) -> tuple[str, str]:
    """Return (body_text, content_type) for ``url``.

    Raises :class:`ToolError` for non-http(s) URLs, network failures, or
    non-2xx responses.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ToolError(f"only absolute http(s) URLs are supported: {url!r}")

    req = Request(url, headers=dict(_BROWSER_HEADERS))
    try:
        with urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
            ctype = resp.headers.get("Content-Type", "") or ""
            encoding = resp.headers.get("Content-Encoding", "") or ""
            raw = resp.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise ToolError(f"HTTP {exc.code} for {url}: {exc.reason}") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise ToolError(f"could not reach {url}: {reason}") from exc
    except TimeoutError as exc:
        raise ToolError(f"request to {url} timed out") from exc
    except OSError as exc:
        raise ToolError(f"network error fetching {url}: {exc}") from exc

    truncated = len(raw) > _MAX_RESPONSE_BYTES
    payload = raw[:_MAX_RESPONSE_BYTES]
    if encoding:
        try:
            payload = _decompress(payload, encoding)
        except (OSError, zlib.error, EOFError) as exc:
            raise ToolError(
                f"failed to decode {encoding!r} response from {url}: {exc}"
            ) from exc
    text = payload.decode("utf-8", errors="replace")
    if truncated:
        text += "\n... [response body truncated at 2 MB]"
    return text, ctype


class _HtmlToText(html.parser.HTMLParser):
    """Minimal HTML→text: drop script/style, treat block tags as newlines."""

    _SKIP = frozenset({"script", "style", "noscript", "svg", "iframe", "head"})
    _BLOCK = frozenset(
        {
            "p", "div", "section", "article", "li", "tr", "br",
            "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote",
            "ul", "ol", "table",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:  # noqa: D401
        if tag in self._SKIP:
            self._skip_depth += 1
            return
        if tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if data:
            self._parts.append(data)

    def to_text(self) -> str:
        raw = "".join(self._parts)
        raw = re.sub(r"[ \t\f\v]+", " ", raw)
        raw = re.sub(r"\n[ \t]+", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _html_to_text(body: str) -> str:
    parser = _HtmlToText()
    try:
        parser.feed(body)
        parser.close()
    except Exception:
        # html.parser is permissive but defensive against pathological input.
        return body
    return parser.to_text()


@dataclass
class WebFetchTool:
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_fetch",
            description=(
                "Fetch the content at an http(s) URL and return it as text. "
                "HTML is stripped to its visible text; other content types "
                "are returned as-is (text decoded as UTF-8 with replacement). "
                "Response bodies are capped at ~2 MB and the returned text "
                "is further trimmed to `max_chars` (default 20 000)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute http(s) URL to fetch.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": (
                            "Trim the returned text to at most this many "
                            f"characters (default {_FETCH_DEFAULT_MAX_CHARS})."
                        ),
                    },
                },
                "required": ["url"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        url = (kwargs.get("url") or "").strip()
        if not url:
            raise ToolError("web_fetch requires a 'url'.")
        max_chars = int(kwargs.get("max_chars") or _FETCH_DEFAULT_MAX_CHARS)
        if max_chars <= 0:
            raise ToolError("max_chars must be positive.")

        body, ctype = _fetch(url)
        text = _html_to_text(body) if "html" in ctype.lower() else body
        if len(text) > max_chars:
            dropped = len(text) - max_chars
            text = text[:max_chars] + f"\n... [truncated {dropped} chars]"
        return text


@dataclass
class WebSearchTool:
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_search",
            description=(
                "Search the web for current information using DuckDuckGo's "
                "HTML endpoint (no API key required). Returns a numbered "
                "list of results as 'title / URL / snippet'. For deeper "
                "content on a specific result, follow up with `web_fetch`."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-form search query.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            f"Maximum number of results "
                            f"(default {_SEARCH_DEFAULT_LIMIT}, "
                            f"max {_SEARCH_MAX_LIMIT})."
                        ),
                    },
                },
                "required": ["query"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        query = (kwargs.get("query") or "").strip()
        if not query:
            raise ToolError("web_search requires a non-empty 'query'.")
        limit_raw = int(kwargs.get("limit") or _SEARCH_DEFAULT_LIMIT)
        limit = max(1, min(limit_raw, _SEARCH_MAX_LIMIT))

        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        body, _ctype = _fetch(url)
        results = _parse_ddg(body, limit=limit)
        if not results:
            return f"(no results for {query!r})"

        rendered = []
        for i, r in enumerate(results, start=1):
            rendered.append(
                f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}"
            )
        return "\n\n".join(rendered)


_RESULT_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
    r'(?:.*?<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>)?',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _parse_ddg(html_text: str, *, limit: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for match in _RESULT_RE.finditer(html_text):
        href_raw, title_raw, snippet_raw = match.group(1), match.group(2), match.group(3)
        resolved = _resolve_ddg_href(href_raw)
        if not resolved:
            continue
        title = _TAG_RE.sub("", title_raw).strip()
        snippet = _TAG_RE.sub("", snippet_raw or "").strip()
        out.append({"url": resolved, "title": title, "snippet": snippet})
        if len(out) >= limit:
            break
    return out


def _resolve_ddg_href(href: str) -> str:
    """DDG wraps result URLs in /l/?uddg=<encoded>. Unwrap when possible."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        target = qs.get("uddg")
        if target:
            return unquote(target[0])
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return href
    return ""


__all__ = ["WebFetchTool", "WebSearchTool"]
