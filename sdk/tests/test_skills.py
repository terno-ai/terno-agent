from __future__ import annotations

from pathlib import Path

from terno_agent.agents.terno import TernoAgent
from terno_agent.core.messages import AssistantMessage
from terno_agent.llm.base import LLMResponse
from terno_agent.skills import ActivateSkillTool, discover_skills


class _CapturingLLM:
    model = "dummy"

    def __init__(self) -> None:
        self.messages = []
        self.tools = []

    def complete(self, messages, tools=None, **kwargs):
        self.messages = messages
        self.tools = tools or []
        return LLMResponse(message=AssistantMessage(content="done"), stop_reason="stop")


def _write_skill(root: Path, name: str, description: str, body: str = "Do the thing.") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_discover_skills_reads_metadata_and_body(tmp_path):
    root = tmp_path / ".agents" / "skills"
    skill_file = _write_skill(
        root,
        "code-review",
        "Review code for regressions. Use when the user asks for review.",
        "Check behavior first.",
    )
    (skill_file.parent / "references").mkdir()
    (skill_file.parent / "references" / "checklist.md").write_text("Checklist", encoding="utf-8")

    catalog = discover_skills(tmp_path, include_builtin=False, include_user=False)

    assert list(catalog.skills) == ["code-review"]
    skill = catalog.skills["code-review"]
    assert skill.description.startswith("Review code")
    assert "Check behavior first." in skill.body
    assert not catalog.diagnostics


def test_activate_skill_returns_a_stub_and_defers_the_body(tmp_path):
    root = tmp_path / ".agents" / "skills"
    _write_skill(root, "docs", "Write docs. Use for documentation tasks.", "Prefer examples.")
    (root / "docs" / "references").mkdir()
    (root / "docs" / "references" / "style.md").write_text("Style", encoding="utf-8")
    catalog = discover_skills(tmp_path, include_builtin=False, include_user=False)
    tool = ActivateSkillTool(catalog)

    result = tool.run(skill="docs")

    # The tool RESULT is a stub, matching the reference harness. Keeping the
    # body out of the tool result is what protects it from compaction.
    assert result == "Launching skill: docs"
    assert "Prefer examples." not in result

    body = tool.take_pending_context()
    assert '<skill_content name="docs">' in body
    assert "Prefer examples." in body
    assert "Skill directory:" in body
    assert "<file>references/style.md</file>" in body

    # Drained once, so a later tool call can't re-attach a stale body.
    assert tool.take_pending_context() == ""


def test_scripts_are_listed_separately_from_references(tmp_path):
    root = tmp_path / ".agents" / "skills"
    _write_skill(root, "viz", "Charts.", "Body.")
    (root / "viz" / "references").mkdir()
    (root / "viz" / "references" / "palette.md").write_text("hexes", encoding="utf-8")
    (root / "viz" / "scripts").mkdir()
    (root / "viz" / "scripts" / "validate.js").write_text("//", encoding="utf-8")
    catalog = discover_skills(tmp_path, include_builtin=False, include_user=False)

    tool = ActivateSkillTool(catalog)
    tool.run(skill="viz")
    body = tool.take_pending_context()

    # A reference is Read; a script is executed and never needs reading. Listing
    # them together invites the model to Read a script it should have run.
    refs = body.split("<skill_references>")[1].split("</skill_references>")[0]
    scripts = body.split("<skill_scripts>")[1].split("</skill_scripts>")[0]
    assert "references/palette.md" in refs and "validate.js" not in refs
    assert "scripts/validate.js" in scripts and "palette.md" not in scripts
    assert "Do not Read them." in scripts


def test_body_reaches_the_wire_as_a_sibling_text_part(tmp_path):
    from terno_agent.core.messages import ToolResult, ToolResultMessage
    from terno_agent.llm.anthropic_client import _to_anthropic

    payload = _to_anthropic(
        ToolResultMessage(
            results=[
                ToolResult(call_id="c1", content="Launching skill: docs",
                           followup_text="<skill_content name=\"docs\">body</skill_content>")
            ]
        )
    )

    kinds = [p["type"] for p in payload["content"]]
    assert kinds == ["tool_result", "text"]
    assert payload["content"][0]["content"] == "Launching skill: docs"
    assert "body" in payload["content"][1]["text"]


def test_terno_agent_exposes_skill_catalog_and_tool(tmp_path):
    root = tmp_path / ".agents" / "skills"
    _write_skill(root, "testing", "Improve tests. Use when adding or fixing tests.")
    catalog = discover_skills(tmp_path, include_builtin=False, include_user=False)
    llm = _CapturingLLM()
    agent = TernoAgent(llm, workdir=tmp_path, skill_catalog=catalog)

    result = agent.run("please fix the tests")

    assert result.answer == "done"
    assert "Skill" in agent.tools
    system_prompt = llm.messages[0].content
    assert "<available_skills>" in system_prompt
    assert "<name>testing</name>" in system_prompt
    assert any(tool.name == "Skill" for tool in llm.tools)


def test_builtin_skills_are_available_by_default(tmp_path):
    catalog = discover_skills(tmp_path, include_user=False)

    for name in (
        "code-review",
        "data-analysis",
        "data-cleaning",
        "data-visualization",
        "debugging",
        "documentation",
        "machine-learning",
        "python-data",
        "research-synthesis",
        "sql-analysis",
        "task-planning",
    ):
        assert name in catalog.skills
    assert "Explore, clean, summarize" in catalog.skills["data-analysis"].description


def test_project_skill_overrides_builtin(tmp_path):
    root = tmp_path / ".agents" / "skills"
    _write_skill(root, "data-analysis", "Project-specific data workflow.")

    catalog = discover_skills(tmp_path, include_user=False)

    assert catalog.skills["data-analysis"].path == (root / "data-analysis" / "SKILL.md").resolve()
    assert any("shadows" in diagnostic.message for diagnostic in catalog.diagnostics)


def test_skill_tool_matches_the_captured_schema(tmp_path):
    _write_skill(tmp_path / ".agents" / "skills", "docs", "Write documentation.")
    catalog = discover_skills(tmp_path, include_user=False)
    schema = ActivateSkillTool(catalog).schema

    assert schema.name == "Skill"
    # Param is `skill`, per the capture — not Terno's old `name`.
    assert schema.parameters["required"] == ["skill"]
    # Kept from Terno: the enum makes "do not guess names" enforceable. Built-in
    # skills are discovered alongside the project one, so just check membership.
    assert "docs" in schema.parameters["properties"]["skill"]["enum"]
    assert schema.description.startswith("Invoke a skill.")


def test_catalog_advertises_the_current_tool_name(tmp_path):
    # The tool was renamed activate_skill -> Skill; the catalog blurb has to
    # match, or the prompt points the model at a tool that does not exist.
    _write_skill(tmp_path / ".agents" / "skills", "docs", "Write documentation.")
    section = discover_skills(tmp_path, include_user=False).prompt_section()

    assert "`Skill`" in section
    assert "activate_skill" not in section
    # Metadata only — the body is loaded on activation, not advertised.
    assert "docs" in section
    assert "Do the thing." not in section


# ----- slash invocation ---------------------------------------------------- #


def test_slash_wrapper_matches_the_capture_byte_for_byte():
    from terno_agent.skills.slash import wrapper_text

    assert wrapper_text("simplify") == (
        "<command-message>simplify</command-message>\n"
        "<command-name>/simplify</command-name>\n"
    )


def test_slash_parses_name_and_arguments():
    from terno_agent.skills.slash import parse

    assert parse("/docs") == ("docs", "")
    assert parse("/docs some args here") == ("docs", "some args here")
    # Not slash invocations.
    assert parse("docs") is None
    assert parse("/") is None
    assert parse("/a/b") is None


def test_slash_yields_two_parts_wrapper_then_body(tmp_path):
    from terno_agent.core.messages import TextPart
    from terno_agent.skills.slash import resolve

    _write_skill(tmp_path / ".agents" / "skills", "docs", "Write docs.", "Prefer examples.")
    catalog = discover_skills(tmp_path, include_builtin=False, include_user=False)

    parts = resolve("/docs", catalog)

    assert [type(p) for p in parts] == [TextPart, TextPart]
    assert parts[0].text.startswith("<command-message>docs</command-message>")
    assert "Prefer examples." in parts[1].text
    # No Skill tool call is involved on this path.
    assert "Launching skill" not in parts[1].text


def test_slash_arguments_are_appended_to_the_body(tmp_path):
    from terno_agent.skills.slash import resolve

    _write_skill(tmp_path / ".agents" / "skills", "docs", "Write docs.")
    catalog = discover_skills(tmp_path, include_builtin=False, include_user=False)

    parts = resolve("/docs focus on the CLI", catalog)
    assert parts[1].text.endswith("focus on the CLI")


def test_unknown_and_reserved_slashes_fall_through(tmp_path):
    from terno_agent.skills.slash import resolve

    _write_skill(tmp_path / ".agents" / "skills", "clear", "Shadows a builtin.")
    catalog = discover_skills(tmp_path, include_builtin=False, include_user=False)

    # Unknown -> None, so the caller can treat it as ordinary chat rather than
    # erroring on a typo.
    assert resolve("/nosuchskill", catalog) is None
    assert resolve("not a slash command", catalog) is None
    # A skill may not hijack a command the CLI handles itself.
    assert resolve("/clear", catalog) is None
