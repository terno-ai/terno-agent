from __future__ import annotations

import difflib
import re
from pathlib import Path

from terno_agent.prompts import (
    PromptContext,
    blocks,
    build_system_prompt,
    render_session_block,
    render_system_prompt,
)
from terno_agent.prompts.context import GitSnapshot

# The reference corpus these blocks were ported from. Absent on machines that
# only have the SDK, so fidelity tests skip rather than fail.
CORPUS = Path("/Users/navin/terno/cc-corpus/system")

# Located by content hash, not by file name: names carry an ordinal prefix that
# shifts whenever another capture run is merged into the corpus.
CORE_HASH = "3b27271fa44c"
SESSION_HASH = "67e35e4f1c02"


def _captured(block_hash: str) -> str | None:
    """The captured block with this hash, or None if the corpus isn't here."""
    if not CORPUS.is_dir():
        return None
    matches = sorted(CORPUS.glob(f"*_{block_hash}.md"))
    return matches[0].read_text() if matches else None


def _ctx(tmp_path: Path, **kwargs: object) -> PromptContext:
    return PromptContext(
        cwd=tmp_path,
        session_id="sess-1",
        model_name="Opus 5",
        model_id="claude-opus-5",
        knowledge_cutoff="May 2026",
        memory_path=tmp_path / ".terno" / "memory",
        scratchpad_path=tmp_path / "scratch",
        **kwargs,  # type: ignore[arg-type]
    )


# ----- block structure ---------------------------------------------------- #


def test_blocks_carry_the_captured_cache_breakpoints(tmp_path: Path) -> None:
    sp = build_system_prompt(_ctx(tmp_path))

    assert len(sp.blocks) == 3
    # Identity is small and never worth a breakpoint.
    assert sp.blocks[0].cache_control is None
    # Static across every session and user, so the cache entry is shared.
    assert sp.blocks[1].cache_control == {
        "type": "ephemeral",
        "ttl": "1h",
        "scope": "global",
    }
    # Session-local: embeds cwd and git state.
    assert sp.blocks[2].cache_control == {"type": "ephemeral", "ttl": "1h"}


def test_extra_prompt_becomes_its_own_trailing_block(tmp_path: Path) -> None:
    sp = build_system_prompt(_ctx(tmp_path), extra="TOOLS GO HERE")

    # Appended, not merged — merging would invalidate the cached blocks above.
    assert len(sp.blocks) == 4
    assert sp.blocks[3].text == "TOOLS GO HERE"
    assert sp.blocks[3].cache_control is None
    assert sp.blocks[1].cache_control is not None


def test_to_anthropic_emits_text_blocks(tmp_path: Path) -> None:
    payload = build_system_prompt(_ctx(tmp_path)).to_anthropic()

    assert [b["type"] for b in payload] == ["text"] * 3
    assert "cache_control" not in payload[0]
    assert payload[1]["cache_control"]["scope"] == "global"


def test_render_flattens_in_block_order(tmp_path: Path) -> None:
    sp = build_system_prompt(_ctx(tmp_path))

    assert sp.render() == "\n\n".join(b.text for b in sp.blocks)
    assert sp.render().startswith(blocks.IDENTITY)


# ----- runtime injection -------------------------------------------------- #


def test_session_block_injects_runtime_paths(tmp_path: Path) -> None:
    text = render_session_block(_ctx(tmp_path))

    assert str(tmp_path / ".terno" / "memory") in text
    assert str(tmp_path / "scratch") in text
    assert f"Primary working directory: {tmp_path}" in text
    assert "You are powered by the model named Opus 5." in text
    assert "The exact model ID is claude-opus-5." in text
    # No placeholder may survive rendering.
    assert "{{" not in text


def test_capability_flags_gate_their_sections(tmp_path: Path) -> None:
    off = render_session_block(
        _ctx(tmp_path, supports_bang_prefix=False, supports_skills=False,
             has_agent_tool=False, has_workflow_tool=False)
    )
    assert "# Session-specific guidance" not in off
    assert blocks.NO_UNPROMPTED_AGENT not in off
    assert blocks.NO_UNPROMPTED_WORKFLOW not in off

    on = render_session_block(
        _ctx(tmp_path, supports_bang_prefix=True, supports_skills=True,
             has_agent_tool=True, has_workflow_tool=True)
    )
    assert "# Session-specific guidance" in on
    assert blocks.GUIDANCE_BANG_PREFIX in on
    assert blocks.NO_UNPROMPTED_AGENT in on
    assert blocks.NO_UNPROMPTED_WORKFLOW in on


def test_git_section_omitted_outside_a_repo(tmp_path: Path) -> None:
    outside = render_session_block(_ctx(tmp_path, git=GitSnapshot(is_repo=False)))
    assert "gitStatus:" not in outside
    assert "Is a git repository: false" in outside

    inside = render_session_block(
        _ctx(
            tmp_path,
            git=GitSnapshot(
                is_repo=True, branch="feature/x", main_branch="main",
                user="someone", status="(clean)", recent_commits="abc123 do a thing",
            ),
        )
    )
    assert "gitStatus:" in inside
    assert "Current branch: feature/x" in inside
    assert "abc123 do a thing" in inside


def test_skill_section_lands_before_git_status(tmp_path: Path) -> None:
    text = render_session_block(
        _ctx(tmp_path, git=GitSnapshot(is_repo=True, branch="main", status="(clean)")),
        skill_section="# Skills\n- thing",
    )
    assert text.index("# Skills") < text.index("gitStatus:")


def test_detect_probes_the_environment(tmp_path: Path) -> None:
    ctx = PromptContext.detect(tmp_path)

    assert ctx.cwd == tmp_path.resolve()
    # Not a repo, and path helpers still resolve without explicit overrides.
    assert ctx.git.is_repo is False
    assert ctx.scratchpad_dir.name == "scratchpad"
    assert ctx.memory_dir.is_absolute()


def test_render_system_prompt_needs_no_arguments() -> None:
    # Used by the back-compat `prompts.SYSTEM_PROMPT` accessor.
    assert blocks.IDENTITY in render_system_prompt()


# ----- fidelity to the captured reference --------------------------------- #


def test_core_block_matches_the_capture_byte_for_byte() -> None:
    captured = _captured(CORE_HASH)
    if captured is None:
        return  # corpus not present on this machine
    assert blocks.CORE == captured


def test_session_sections_appear_verbatim_in_the_capture() -> None:
    text = _captured(SESSION_HASH)
    if text is None:
        return
    for section in (
        blocks.CODE_STYLE,
        blocks.PRONOUNS,
        blocks.CARE_AND_REPORTING,
        blocks.GUIDANCE_BANG_PREFIX,
        blocks.GUIDANCE_SKILLS,
        blocks.CONTEXT_MANAGEMENT,
        blocks.DELIVERING_WORK,
        blocks.CORRECTIONS,
        blocks.NO_UNPROMPTED_AGENT,
        blocks.NO_UNPROMPTED_WORKFLOW,
    ):
        assert section in text

    # Templated sections match up to their first placeholder.
    for template in (
        blocks.MEMORY,
        blocks.LANGUAGE,
        blocks.SCRATCHPAD,
        blocks.ENVIRONMENT,
        blocks.GIT_STATUS,
    ):
        prefix = template.split("{{")[0]
        assert prefix in text


# Second capture run, from a different repo with no MCP attached.
SESSION_HASH_ALT = "abafc2b89634"


def test_only_the_templated_parts_differ_between_captured_sessions() -> None:
    """Two sessions' block 3 should differ *only* where we inject.

    This is what justifies treating the rest as static text: if a section we
    hardcode turned out to vary by session, it would show up here.
    """
    a, b = _captured(SESSION_HASH), _captured(SESSION_HASH_ALT)
    if a is None or b is None:
        return

    changed = [
        line[1:]
        for line in difflib.unified_diff(a.splitlines(), b.splitlines(), n=0)
        if line[:1] in "+-" and line[:3] not in ("---", "+++")
    ]
    assert changed, "expected the two sessions to differ somewhere"

    # Every differing line must belong to an injected field: the memory dir, the
    # working directory, the scratchpad path, or the git snapshot.
    for line in changed:
        assert (
            "/memory/" in line
            or "Primary working directory:" in line
            or "/scratchpad" in line
            # git status lines: porcelain flags, or a `<sha> <subject>` commit
            or line.startswith(("?? ", " M ", "M ", "A ", "D ", "(clean)"))
            or re.match(r"^[0-9a-f]{7,40} ", line)
        ), f"unexpected session-varying line: {line!r}"
