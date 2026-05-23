import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.mcp.bridge import AsyncBridge
from terno_agent.mcp.tool import McpTool, format_content, format_tool_name
from tests.mcp.conftest import FakeCallResult, FakeContentBlock, FakeSession, FakeTool


@pytest.fixture()
def bridge():
    b = AsyncBridge()
    b.start()
    yield b
    b.stop()


def test_format_tool_name_basic():
    assert format_tool_name("fetch", "fetch") == "mcp__fetch__fetch"


def test_format_tool_name_sanitizes():
    name = format_tool_name("my server", "weird.tool!")
    assert name.startswith("mcp__my_server__weird_tool_")


def test_format_tool_name_truncates_long_names():
    long_tool = "a" * 100
    name = format_tool_name("srv", long_tool)
    assert len(name) <= 64


def test_format_content_text_blocks():
    out = format_content(
        [FakeContentBlock(type="text", text="line 1"), FakeContentBlock(type="text", text="line 2")]
    )
    assert out == "line 1\nline 2"


def test_format_content_empty():
    assert format_content([]) == ""


def test_mcp_tool_run_success(bridge):
    session = FakeSession([FakeTool(name="echo")])
    tool = McpTool(
        server="fake",
        tool_name="echo",
        description="echo",
        input_schema={"type": "object"},
        session=session,
        bridge=bridge,
        timeout_s=2,
    )
    out = tool.run(value="hi")
    assert "echo:{'value': 'hi'}" in out
    assert session.calls == [("echo", {"value": "hi"})]


def test_mcp_tool_run_is_error(bridge):
    def _call(name, args):
        return FakeCallResult(
            content=[FakeContentBlock(type="text", text="boom")], isError=True
        )

    session = FakeSession([FakeTool(name="x")], call_impl=_call)
    tool = McpTool(
        server="fake",
        tool_name="x",
        description="",
        input_schema={"type": "object"},
        session=session,
        bridge=bridge,
        timeout_s=2,
    )
    with pytest.raises(ToolError) as exc:
        tool.run()
    assert "boom" in str(exc.value)


def test_mcp_tool_run_session_raises(bridge):
    def _call(name, args):
        raise RuntimeError("kaboom")

    session = FakeSession([FakeTool(name="x")], call_impl=_call)
    tool = McpTool(
        server="fake",
        tool_name="x",
        description="",
        input_schema={"type": "object"},
        session=session,
        bridge=bridge,
        timeout_s=2,
    )
    with pytest.raises(ToolError) as exc:
        tool.run()
    assert "kaboom" in str(exc.value)


def test_mcp_tool_schema_passes_through(bridge):
    schema_in = {"type": "object", "properties": {"q": {"type": "string"}}}
    tool = McpTool(
        server="fake",
        tool_name="echo",
        description="d",
        input_schema=schema_in,
        session=FakeSession(),
        bridge=bridge,
    )
    schema = tool.schema
    assert schema.name == "mcp__fake__echo"
    assert schema.parameters is schema_in
    assert schema.description.startswith("[mcp:fake]")
