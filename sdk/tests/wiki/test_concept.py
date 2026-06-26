"""Concept + frontmatter render/parse roundtrips."""

from __future__ import annotations

import pytest

from terno_agent.wiki import frontmatter
from terno_agent.wiki.concept import Concept, ConceptError


def test_render_parse_roundtrip():
    c = Concept(
        concept_id="tables/users",
        title="users",
        type="table",
        summary="Registered users.",
        body="## Overview\n\nThe users table.",
        metadata={"source": "introspection+llm", "tags": ["core", "pii"]},
    )
    text = c.render()
    assert text.startswith("---\n")
    parsed = Concept.parse("tables/users", text)
    assert parsed.title == "users"
    assert parsed.type == "table"
    assert parsed.summary == "Registered users."
    assert "The users table." in parsed.body
    assert parsed.metadata["tags"] == ["core", "pii"]


def test_preferred_key_order():
    text = frontmatter.render(
        {"source": "x", "title": "T", "type": "table", "summary": "s"}, "body"
    )
    # title/type/summary come before the alphabetical remainder.
    head = text.split("---")[1]
    assert head.index("title") < head.index("type") < head.index("summary")
    assert head.index("summary") < head.index("source")


def test_missing_required_fields_raises():
    with pytest.raises(ConceptError):
        Concept.parse("x", "---\ntitle: only title\n---\n\nbody")
    with pytest.raises(ConceptError):
        Concept(concept_id="x", title="", type="table")


def test_parse_without_frontmatter_recovers_body():
    fm, body = frontmatter.parse("no frontmatter here")
    assert fm == {}
    assert body == "no frontmatter here"
