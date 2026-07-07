"""Wiki memory — file-based memory bundles curated by a background agent.

A *bundle* is a directory of markdown files (with YAML frontmatter) that holds
memory for one datasource: an ``index.md`` for progressive disclosure plus one
file per fact, organized in subdirectories. One file = one fact. Facts carry a
``type`` (user|feedback|project|reference, or table|domain|metric|datasource)
and a ``scope`` (``global`` or ``datasource:<id>``), mirroring terno-ai's
memory format.

  - Curate:   MemoryAgent (per-turn background loop; import from
              ``terno_agent.wiki.agent`` — it depends on
              ``terno_agent.agents.base``, so it is not re-exported here to
              avoid an import cycle).
  - Build:    DatasourceKnowledgeAgent (DB introspection + optional enrichment).
  - Format:   Concept, KnowledgeBundle (read/write + index generation).
  - Consume:  MemoryContextProvider (per-turn injection into the main agent).
  - Tools:    memory_read_tools (main agent) / memory_agent_tools (curator).
"""

from terno_agent.wiki.builder import BuildReport, DatasourceKnowledgeAgent
from terno_agent.wiki.bundle import KnowledgeBundle
from terno_agent.wiki.concept import Concept, ConceptError
from terno_agent.wiki.context import MemoryContextProvider
from terno_agent.wiki.paths import bundle_dir, knowledge_root, slugify
from terno_agent.wiki.tools import (
    MemoryEditTool,
    MemoryListTool,
    MemoryReadTool,
    MemorySearchTool,
    MemoryWriteTool,
    memory_agent_tools,
    memory_read_tools,
)

__all__ = [
    "BuildReport",
    "Concept",
    "ConceptError",
    "DatasourceKnowledgeAgent",
    "KnowledgeBundle",
    "MemoryContextProvider",
    "MemoryEditTool",
    "MemoryListTool",
    "MemoryReadTool",
    "MemorySearchTool",
    "MemoryWriteTool",
    "bundle_dir",
    "knowledge_root",
    "memory_agent_tools",
    "memory_read_tools",
    "slugify",
]
