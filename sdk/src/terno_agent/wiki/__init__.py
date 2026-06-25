"""Open Knowledge Format (OKF) — datasource knowledge bundles.

A *bundle* is a directory of markdown files (with YAML frontmatter) that
describes one datasource: an ``index.md`` for progressive disclosure plus
one concept document per table, organized in subdirectories.

  - Curate:   KnowledgeAgent (per-turn loop; import from terno_agent.okf.agent)
  - Build:    DatasourceKnowledgeAgent (introspection + optional LLM enrichment)
  - Format:   Concept, KnowledgeBundle (read/write + index generation)
  - Consume:  KnowledgeContextProvider (per-turn injection into the main agent)

``KnowledgeAgent`` lives in ``terno_agent.okf.agent`` and is intentionally not
re-exported here: it imports ``terno_agent.agents.base``, so importing it from
this package would create an import cycle. Import it from its submodule.
"""

from terno_agent.wiki.builder import BuildReport, DatasourceKnowledgeAgent
from terno_agent.wiki.bundle import KnowledgeBundle
from terno_agent.wiki.concept import Concept, ConceptError
from terno_agent.wiki.context import KnowledgeContextProvider
from terno_agent.wiki.paths import bundle_dir, knowledge_root, slugify
from terno_agent.wiki.tools import (
    BuildDatasourceKnowledgeTool,
    ListKnowledgeTool,
    ReadConceptTool,
    WriteConceptTool,
    knowledge_agent_tools,
)

__all__ = [
    "BuildDatasourceKnowledgeTool",
    "BuildReport",
    "Concept",
    "ConceptError",
    "DatasourceKnowledgeAgent",
    "KnowledgeBundle",
    "KnowledgeContextProvider",
    "ListKnowledgeTool",
    "ReadConceptTool",
    "WriteConceptTool",
    "bundle_dir",
    "knowledge_agent_tools",
    "knowledge_root",
    "slugify",
]
