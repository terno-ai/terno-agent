"""A single OKF concept — one unit of knowledge, one markdown document.

A concept's *id* is the path of its file within the bundle, with the ``.md``
suffix removed (e.g. ``tables/users.md`` → ``tables/users``). The required
frontmatter fields are ``title`` and ``type``; everything else is optional
metadata the producer chooses to attach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from terno_agent.wiki.frontmatter import parse as parse_frontmatter
from terno_agent.wiki.frontmatter import render as render_frontmatter

REQUIRED_FIELDS = ("title", "type")


class ConceptError(ValueError):
    """Raised when a concept is missing required frontmatter."""


@dataclass(slots=True)
class Concept:
    concept_id: str
    title: str
    type: str
    summary: str = ""
    body: str = ""
    # Any additional frontmatter (updated, source, tags, links, ...).
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.concept_id = self.concept_id.strip().strip("/")
        if not self.concept_id:
            raise ConceptError("concept_id must be non-empty.")
        if not self.title.strip():
            raise ConceptError(f"concept {self.concept_id!r} is missing a title.")
        if not self.type.strip():
            raise ConceptError(f"concept {self.concept_id!r} is missing a type.")

    # ----- serialization ------------------------------------------------- #

    def frontmatter(self) -> dict[str, Any]:
        fm: dict[str, Any] = {"title": self.title, "type": self.type}
        if self.summary:
            fm["summary"] = self.summary
        for key, value in self.metadata.items():
            if key not in ("title", "type", "summary"):
                fm[key] = value
        return fm

    def render(self) -> str:
        return render_frontmatter(self.frontmatter(), self.body)

    @classmethod
    def parse(cls, concept_id: str, text: str) -> Concept:
        fm, body = parse_frontmatter(text)
        title = str(fm.get("title", "")).strip()
        type_ = str(fm.get("type", "")).strip()
        if not title or not type_:
            raise ConceptError(
                f"concept {concept_id!r} is missing required frontmatter "
                f"({', '.join(REQUIRED_FIELDS)})."
            )
        summary = str(fm.get("summary", "")).strip()
        metadata = {
            k: v for k, v in fm.items() if k not in ("title", "type", "summary")
        }
        return cls(
            concept_id=concept_id,
            title=title,
            type=type_,
            summary=summary,
            body=body,
            metadata=metadata,
        )


__all__ = ["Concept", "ConceptError", "REQUIRED_FIELDS"]
