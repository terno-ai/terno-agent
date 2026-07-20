from __future__ import annotations

from pathlib import Path

from terno_agent.agents.terno import TernoAgent
from terno_agent.config import Config
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


def test_activate_skill_returns_wrapped_body_and_resources(tmp_path):
    root = tmp_path / ".agents" / "skills"
    _write_skill(root, "docs", "Write docs. Use for documentation tasks.", "Prefer examples.")
    (root / "docs" / "references").mkdir()
    (root / "docs" / "references" / "style.md").write_text("Style", encoding="utf-8")
    catalog = discover_skills(tmp_path, include_builtin=False, include_user=False)

    result = ActivateSkillTool(catalog).run(name="docs")

    assert '<skill_content name="docs">' in result
    assert "Prefer examples." in result
    assert "Skill directory:" in result
    assert "<file>references/style.md</file>" in result


def test_terno_agent_exposes_skill_catalog_and_tool(tmp_path):
    root = tmp_path / ".agents" / "skills"
    _write_skill(root, "testing", "Improve tests. Use when adding or fixing tests.")
    catalog = discover_skills(tmp_path, include_builtin=False, include_user=False)
    llm = _CapturingLLM()
    agent = TernoAgent(llm, workdir=tmp_path, skill_catalog=catalog)

    result = agent.run("please fix the tests")

    assert result.answer == "done"
    assert "activate_skill" in agent.tools
    system_prompt = llm.messages[0].content
    assert "<available_skills>" in system_prompt
    assert "<name>testing</name>" in system_prompt
    assert any(tool.name == "activate_skill" for tool in llm.tools)


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


def test_config_can_scope_skills_to_host_owned_paths_only(tmp_path):
    """A host embedding the SDK (e.g. terno-ai) with its own curated skill
    set can set skill_include_builtin/skill_include_user False so only its
    own skill_paths are discovered — none of the SDK's generic builtins."""
    host_skills = tmp_path / "host-skills"
    _write_skill(host_skills, "data-visualization", "Host-specific viz instructions.")

    config = Config(
        llm_provider="terno",
        llm_model="dummy",
        llm_api_key="test-key",
        provisioner_url="https://example.invalid",
        sandbox="none",
        sandbox_fallback="none",
        mcp_enabled=False,
        memory_enabled=False,
        file_memory_enabled=False,
        skills_enabled=True,
        skill_paths=[str(host_skills)],
        skill_include_builtin=False,
        skill_include_user=False,
    )

    agent = TernoAgent.from_config(config, workdir=tmp_path / "workdir")

    assert list(agent.skill_catalog.skills) == ["data-visualization"]
    assert (
        agent.skill_catalog.skills["data-visualization"].description
        == "Host-specific viz instructions."
    )
    assert "activate_skill" in agent.tools
